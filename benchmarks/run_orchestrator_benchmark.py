from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import tracemalloc
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastmdanalysis import FastMDAnalysis  # noqa: E402
from dataset_config import get_dataset_config, list_datasets  # noqa: E402
from instrumentation import Instrumentation  # noqa: E402

TOOL_ID = "fastmdanalysis"
FRAME_STRIDE = 10
MAX_FRAMES = 120
FRAME_SLICE = (0, FRAME_STRIDE * MAX_FRAMES, FRAME_STRIDE)
CLUSTER_ATOM_SELECTION = "protein and name CA"
ANALYSES = ("rmsd", "rmsf", "rg", "cluster")
ANALYZE_OPTIONS = {
    "rmsd": {"ref": 0, "align": True},
    "rg": {"by_chain": False},
    "cluster": {
        "methods": ["dbscan"],
        "eps": 0.3,
        "min_samples": 5,
        "atoms": CLUSTER_ATOM_SELECTION,
    },
}
SLIDES_ENABLED = True
REPEATS = 5

DEFAULT_DATASET = "trpcage"
DATASET_SLUG = DEFAULT_DATASET
DATASET_LABEL = ""
OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results" / f"orchestrator_{DATASET_SLUG}"


@dataclass
class RunMetrics:
    calc_s: float
    plot_s: float
    total_s: float
    calc_mem_mb: float
    plot_mem_mb: float
    peak_mem_mb: float
    per_analysis_s: dict[str, float] | None = None

    @property
    def elapsed_s(self) -> float:
        return self.total_s


class _PlottingMemoryTracker:
    """Track calculation vs plotting memory usage using tracemalloc."""

    def __init__(self) -> None:
        self.plot_started = False
        self.calc_peak_bytes = 0
        self._plot_increment_bytes = 0
        self._baseline_current = 0
        self._overall_peak_bytes = 0

    def observe(self) -> None:
        if not tracemalloc.is_tracing():
            return
        current, peak = tracemalloc.get_traced_memory()
        if peak > self._overall_peak_bytes:
            self._overall_peak_bytes = peak
        if self.plot_started:
            increment = peak - self._baseline_current
            if increment > self._plot_increment_bytes:
                self._plot_increment_bytes = max(0, increment)
        else:
            if peak > self.calc_peak_bytes:
                self.calc_peak_bytes = peak

    def mark_plot_start(self) -> None:
        if self.plot_started or not tracemalloc.is_tracing():
            return
        current, peak = tracemalloc.get_traced_memory()
        if peak > self._overall_peak_bytes:
            self._overall_peak_bytes = peak
        if peak > self.calc_peak_bytes:
            self.calc_peak_bytes = peak
        self._baseline_current = current
        self.plot_started = True
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()

    def calc_bytes(self, fallback_peak: int) -> int:
        return self.calc_peak_bytes or fallback_peak

    def plot_bytes(self) -> int:
        return max(0, self._plot_increment_bytes)

    def overall_bytes(self, fallback_peak: int) -> int:
        return max(fallback_peak, self._overall_peak_bytes)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values) if len(values) > 1 else 0.0),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _summarize(metrics: list[RunMetrics]) -> dict[str, dict[str, float]]:
    fields = {
        "calc_s": [m.calc_s for m in metrics],
        "plot_s": [m.plot_s for m in metrics],
        "total_s": [m.total_s for m in metrics],
        "elapsed_s": [m.elapsed_s for m in metrics],
        "calc_mem_mb": [m.calc_mem_mb for m in metrics],
        "plot_mem_mb": [m.plot_mem_mb for m in metrics],
        "peak_mem_mb": [m.peak_mem_mb for m in metrics],
    }
    summary = {name: _stats(values) for name, values in fields.items()}

    per_analysis: dict[str, list[float]] = {}
    for metric in metrics:
        if not metric.per_analysis_s:
            continue
        for name, seconds in metric.per_analysis_s.items():
            per_analysis.setdefault(name, []).append(float(seconds))
    if per_analysis:
        summary["per_analysis_s"] = {name: _stats(values) for name, values in per_analysis.items()}

    return summary


def _collect_loc_metrics() -> dict[str, int]:
    # Single-line snippet: one line to instantiate, one line to run analyze (plotting)
    return {"calc": 1, "plot": 1, "total": 2}


def _set_dataset(slug: str) -> None:
    global DATASET_SLUG, DATASET_LABEL, OUTPUT_ROOT
    config = get_dataset_config(slug)
    DATASET_SLUG = config.slug
    DATASET_LABEL = config.label
    OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results" / f"orchestrator_{DATASET_SLUG}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def _measure_single_run(traj: Path, top: Path, rep_dir: Path, instrument: Instrumentation) -> RunMetrics:
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    tracker: _PlottingMemoryTracker | None = None
    analyze_module = None
    original_slide_show = None
    original_gather_figures = None
    if SLIDES_ENABLED:
        from fastmdanalysis.analysis import analyze as analyze_module  # noqa: WPS433

        tracker = _PlottingMemoryTracker()
        original_slide_show = analyze_module.slide_show
        original_gather_figures = analyze_module.gather_figures

        # Wrap plotting helpers so we can capture a pre-plot peak and the slide-specific peak.
        def _wrapped_gather_figures(*args, **kwargs):  # type: ignore[no-untyped-def]
            if tracker:
                tracker.mark_plot_start()
            try:
                return original_gather_figures(*args, **kwargs)
            finally:
                if tracker:
                    tracker.observe()

        def _wrapped_slide_show(*args, **kwargs):  # type: ignore[no-untyped-def]
            if tracker:
                tracker.mark_plot_start()
            try:
                return original_slide_show(*args, **kwargs)
            finally:
                if tracker:
                    tracker.observe()

        analyze_module.gather_figures = _wrapped_gather_figures
        analyze_module.slide_show = _wrapped_slide_show

    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    instrument.record_attempt(TOOL_ID)
    calc_time = 0.0
    total_time = 0.0
    plot_time = 0.0
    peak_bytes = 0
    calc_peak_bytes = 0
    plot_peak_bytes = 0
    per_analysis_breakdown: dict[str, float] = {}
    slides_seconds = 0.0
    try:
        fastmda = FastMDAnalysis(str(traj), str(top), frames=FRAME_SLICE)
        start_total = time.perf_counter()
        start_calc = time.perf_counter()
        results = fastmda.analyze(
            include=ANALYSES,
            options=ANALYZE_OPTIONS,
            slides=SLIDES_ENABLED,
            output=rep_dir,
            verbose=False,
        )
        calc_time = time.perf_counter() - start_calc
        total_time = time.perf_counter() - start_total
        if tracker:
            tracker.observe()
        _, peak = tracemalloc.get_traced_memory()
        if isinstance(results, dict):
            for name, result in results.items():
                seconds = float(getattr(result, "seconds", 0.0) or 0.0)
                if name == "slides":
                    slides_seconds = seconds
                    continue
                per_analysis_breakdown[name] = seconds
        analysis_sum = sum(per_analysis_breakdown.values())
        if analysis_sum > 0.0:
            calc_time = analysis_sum
        plot_time = slides_seconds
        instrument.record_success(TOOL_ID)
    except Exception:
        instrument.record_exception(TOOL_ID)
        raise
    finally:
        if tracker:
            tracker.observe()
        peak_bytes = tracemalloc.get_traced_memory()[1]
        if tracker:
            calc_peak_bytes = tracker.calc_bytes(peak_bytes)
            plot_peak_bytes = tracker.plot_bytes()
            peak_bytes = tracker.overall_bytes(peak_bytes)
        else:
            calc_peak_bytes = peak_bytes
            plot_peak_bytes = 0
        tracemalloc.stop()
        if analyze_module is not None:
            # Restore original helpers to avoid side-effects on subsequent runs.
            analyze_module.gather_figures = original_gather_figures
            analyze_module.slide_show = original_slide_show

    peak_mb = max(peak_bytes / (1024 ** 2), 0.0)
    calc_mem_mb = max(calc_peak_bytes / (1024 ** 2), 0.0)
    plot_mem_mb = max(plot_peak_bytes / (1024 ** 2), 0.0)
    return RunMetrics(
        calc_s=calc_time,
        plot_s=plot_time,
        total_s=total_time,
        calc_mem_mb=calc_mem_mb,
        plot_mem_mb=plot_mem_mb,
        peak_mem_mb=peak_mb,
        per_analysis_s=per_analysis_breakdown or None,
    )


def _write_outputs(metrics: list[RunMetrics], instrument: Instrumentation, config) -> None:
    summary = {TOOL_ID: _summarize(metrics)}
    loc_metrics = _collect_loc_metrics()
    summary[TOOL_ID]["loc"] = loc_metrics

    per_analysis_names: list[str] = []
    for m in metrics:
        if not m.per_analysis_s:
            continue
        for name in m.per_analysis_s.keys():
            if name not in per_analysis_names:
                per_analysis_names.append(name)
    if per_analysis_names:
        # Preserve user-facing order matching ANALYSES when possible
        ordered = [name for name in ANALYSES if name in per_analysis_names]
        ordered.extend(name for name in per_analysis_names if name not in ordered)
        per_analysis_names = ordered

    summary_csv = OUTPUT_ROOT / "metrics_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tool", "metric", "mean", "stdev", "min", "max"])
        for metric_name, stats in summary[TOOL_ID].items():
            if metric_name in {"loc", "per_analysis_s"}:
                continue
            writer.writerow([
                TOOL_ID,
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['stdev']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
            ])
        writer.writerow([TOOL_ID, "calc_loc", loc_metrics["calc"], loc_metrics["calc"], loc_metrics["calc"], loc_metrics["calc"]])
        writer.writerow([TOOL_ID, "plot_loc", loc_metrics["plot"], loc_metrics["plot"], loc_metrics["plot"], loc_metrics["plot"]])
        writer.writerow([TOOL_ID, "total_loc", loc_metrics["total"], loc_metrics["total"], loc_metrics["total"], loc_metrics["total"]])
        if per_analysis_names:
            for name in per_analysis_names:
                values = [float(m.per_analysis_s.get(name, 0.0)) for m in metrics if m.per_analysis_s]
                if not values:
                    continue
                stats = _stats(values)
                writer.writerow([
                    TOOL_ID,
                    f"analysis:{name}",
                    f"{stats['mean']:.6f}",
                    f"{stats['stdev']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                ])

    breakdown_csv = OUTPUT_ROOT / "metrics_summary_breakdown.csv"
    with breakdown_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tool", "metric", "mean", "stdev", "min", "max"])
        for metric_name in ("calc_s", "plot_s", "total_s", "calc_mem_mb", "plot_mem_mb", "peak_mem_mb"):
            stats = summary[TOOL_ID][metric_name]
            writer.writerow([
                TOOL_ID,
                metric_name,
                f"{stats['mean']:.6f}",
                f"{stats['stdev']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
            ])

    runs_csv = OUTPUT_ROOT / "metrics_runs.csv"
    with runs_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tool", "repeat", "elapsed_s", "calc_mem_mb", "plot_mem_mb", "peak_mem_mb"])
        for idx, m in enumerate(metrics, start=1):
            writer.writerow([
                TOOL_ID,
                idx,
                f"{m.elapsed_s:.6f}",
                f"{m.calc_mem_mb:.6f}",
                f"{m.plot_mem_mb:.6f}",
                f"{m.peak_mem_mb:.6f}",
            ])

    runs_breakdown_csv = OUTPUT_ROOT / "metrics_runs_breakdown.csv"
    with runs_breakdown_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "tool",
            "repeat",
            "calc_s",
            "plot_s",
            "total_s",
            "calc_mem_mb",
            "plot_mem_mb",
            "peak_mem_mb",
        ]
        if per_analysis_names:
            header.extend([f"analysis:{name}" for name in per_analysis_names])
        writer.writerow(header)
        for idx, m in enumerate(metrics, start=1):
            row = [
                TOOL_ID,
                idx,
                f"{m.calc_s:.6f}",
                f"{m.plot_s:.6f}",
                f"{m.total_s:.6f}",
                f"{m.calc_mem_mb:.6f}",
                f"{m.plot_mem_mb:.6f}",
                f"{m.peak_mem_mb:.6f}",
            ]
            if per_analysis_names:
                for name in per_analysis_names:
                    value = 0.0
                    if m.per_analysis_s and name in m.per_analysis_s:
                        value = float(m.per_analysis_s[name])
                    row.append(f"{value:.6f}")
            writer.writerow(row)

    if per_analysis_names:
        analysis_csv = OUTPUT_ROOT / "metrics_analysis_breakdown.csv"
        with analysis_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tool", "repeat", *per_analysis_names])
            for idx, m in enumerate(metrics, start=1):
                row = [TOOL_ID, idx]
                for name in per_analysis_names:
                    value = 0.0
                    if m.per_analysis_s and name in m.per_analysis_s:
                        value = float(m.per_analysis_s[name])
                    row.append(f"{value:.6f}")
                writer.writerow(row)

    payload = {
        "metadata": {
            "trajectory": str(config.traj),
            "topology": str(config.top),
            "dataset": DATASET_SLUG,
            "dataset_label": DATASET_LABEL,
            "repeats": len(metrics),
            "analyses": list(ANALYSES),
            "slides": SLIDES_ENABLED,
        },
        "summary": {
            TOOL_ID: {
                **summary[TOOL_ID],
                "loc": loc_metrics,
            }
        },
        "runs": {
            TOOL_ID: [
                {
                    "calc_s": m.calc_s,
                    "plot_s": m.plot_s,
                    "total_s": m.total_s,
                    "calc_mem_mb": m.calc_mem_mb,
                    "plot_mem_mb": m.plot_mem_mb,
                    "peak_mem_mb": m.peak_mem_mb,
                    "elapsed_s": m.elapsed_s,
                    "analysis_breakdown": m.per_analysis_s or {},
                }
                for m in metrics
            ]
        },
        "instrumentation": {
            TOOL_ID: instrument.to_dict().get(TOOL_ID, {})
        },
    }

    summary_path = OUTPUT_ROOT / "metrics.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    touchpoints_json = OUTPUT_ROOT / "touchpoints.json"
    touchpoints_md = OUTPUT_ROOT / "touchpoints.md"
    touchpoints_json.write_text(json.dumps(instrument.to_dict(), indent=2))
    touchpoints_md.write_text(instrument.to_markdown())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark a single FastMDAnalysis.analyze run over multiple analyses.")
    available_datasets = sorted({name.lower() for name in list_datasets()})
    parser.add_argument(
        "--dataset",
        default=DATASET_SLUG,
        type=str.lower,
        choices=available_datasets,
        help="Dataset identifier to process.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="Number of times to repeat the single-run benchmark.",
    )
    args = parser.parse_args(argv)

    _set_dataset(args.dataset)
    config = get_dataset_config(args.dataset)

    instrument = Instrumentation([TOOL_ID])
    instrument.record_modules(
        TOOL_ID,
        [
            "fastmdanalysis.analysis.rmsd",
            "fastmdanalysis.analysis.rmsf",
            "fastmdanalysis.analysis.rg",
            "fastmdanalysis.analysis.cluster",
            "fastmdanalysis.analysis.analyze",
        ],
    )
    instrument.record_functions(
        TOOL_ID,
        [
            "FastMDAnalysis.analyze",
            "FastMDAnalysis.rmsd",
            "FastMDAnalysis.rmsf",
            "FastMDAnalysis.rg",
            "FastMDAnalysis.cluster",
        ],
    )

    metrics: list[RunMetrics] = []
    for idx in range(1, args.repeats + 1):
        rep_dir = OUTPUT_ROOT / TOOL_ID / f"rep_{idx:02d}"
        metrics.append(_measure_single_run(config.traj, config.top, rep_dir, instrument))

    _write_outputs(metrics, instrument, config)
    print(f"[+] Wrote orchestrator benchmark results to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
