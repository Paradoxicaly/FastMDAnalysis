from __future__ import annotations

import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import rms

from fastmdanalysis.analysis import rmsf as fast_rmsf
from fastmdanalysis.datasets import TrpCage
from instrumentation import Instrumentation


REPEATS = 10
ATOM_SELECTION = None
TOOL_ORDER = ["fastmdanalysis", "mdtraj", "mdanalysis"]

SNIPPET_ROOT = PROJECT_ROOT / "benchmarks" / "snippets" / "rmsf"
SNIPPET_FILES = {
    "fastmdanalysis": SNIPPET_ROOT / "fastmdanalysis_script.py",
    "mdtraj": SNIPPET_ROOT / "mdtraj_script.py",
    "mdanalysis": SNIPPET_ROOT / "mdanalysis_script.py",
}

TRAJ_FILE = Path(TrpCage.traj)
TOPOLOGY_FILE = Path(TrpCage.top)

OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results" / "rmsf_trpcage"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


INSTRUMENT = Instrumentation(TOOL_ORDER)


@dataclass
class RunMetrics:
    elapsed_s: float
    peak_mem_mb: float


def _measure_execution(tool: str, func):
    INSTRUMENT.record_attempt(tool)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
    except Exception:
        INSTRUMENT.record_exception(tool)
        tracemalloc.stop()
        raise
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 ** 2)
    INSTRUMENT.record_success(tool)
    return result, RunMetrics(elapsed_s=elapsed, peak_mem_mb=peak_mb)


def _save_bar_plot(values: np.ndarray, out_path: Path, title: str) -> None:
    atoms = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(atoms, values.flatten())
    ax.set_title(title)
    ax.set_xlabel("Atom Index")
    ax.set_ylabel("RMSF (nm)")
    ax.grid(alpha=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _run_fastmdanalysis(traj, rep_dir: Path) -> np.ndarray:
    INSTRUMENT.record_modules("fastmdanalysis", [
        "fastmdanalysis.analysis.rmsf",
        "fastmdanalysis.datasets",
    ])
    INSTRUMENT.record_functions("fastmdanalysis", [
        "RMSFAnalysis.__init__",
        "RMSFAnalysis.run",
        "RMSFAnalysis.plot",
    ])
    analysis = fast_rmsf.RMSFAnalysis(traj, atoms=ATOM_SELECTION, output=str(rep_dir))
    before = {p.name for p in rep_dir.glob("*")}
    analysis.run()
    after = {p.name for p in rep_dir.glob("*")}
    new_files = [rep_dir / name for name in after - before]
    if new_files:
        INSTRUMENT.record_files("fastmdanalysis", new_files)
    return analysis.data.flatten()


def _run_mdtraj(traj, rep_dir: Path) -> np.ndarray:
    INSTRUMENT.record_modules("mdtraj", [
        "mdtraj",
        "numpy",
        "matplotlib.pyplot",
    ])
    INSTRUMENT.record_functions("mdtraj", [
        "md.load",
        "md.rmsf",
        "numpy.savetxt",
        "_save_bar_plot",
    ])
    avg_xyz = traj.xyz.mean(axis=0, keepdims=True)
    reference = md.Trajectory(avg_xyz, traj.topology)
    values = md.rmsf(traj, reference)
    out_data = rep_dir / "rmsf.dat"
    np.savetxt(out_data, values.reshape(-1, 1), header="rmsf", fmt="%.6f")
    plot_path = rep_dir / "rmsf.png"
    _save_bar_plot(values, plot_path, "RMSF per Atom (MDTraj)")
    INSTRUMENT.record_files("mdtraj", [out_data, plot_path])
    return values


def _run_mdanalysis(universe, rep_dir: Path) -> np.ndarray:
    INSTRUMENT.record_modules("mdanalysis", [
        "MDAnalysis",
        "MDAnalysis.analysis.rms",
        "numpy",
        "matplotlib.pyplot",
    ])
    INSTRUMENT.record_functions("mdanalysis", [
        "rms.RMSF",
        "numpy.savetxt",
        "_save_bar_plot",
    ])
    universe.trajectory[0]
    analysis = rms.RMSF(universe.atoms).run()
    values = analysis.results.rmsf / 10.0
    data_path = rep_dir / "rmsf.dat"
    plot_path = rep_dir / "rmsf.png"
    np.savetxt(data_path, values.reshape(-1, 1), header="rmsf", fmt="%.6f")
    _save_bar_plot(values, plot_path, "RMSF per Atom (MDAnalysis)")
    INSTRUMENT.record_files("mdanalysis", [data_path, plot_path])
    return values


def _summarize(metrics: list[RunMetrics]) -> dict:
    elapsed = [m.elapsed_s for m in metrics]
    memory = [m.peak_mem_mb for m in metrics]
    return {
        "runs": len(metrics),
        "elapsed_s": {
            "mean": statistics.mean(elapsed),
            "stdev": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            "min": min(elapsed),
            "max": max(elapsed),
        },
        "peak_mem_mb": {
            "mean": statistics.mean(memory),
            "stdev": statistics.stdev(memory) if len(memory) > 1 else 0.0,
            "min": min(memory),
            "max": max(memory),
        },
    }


def _count_loc_sections(snippet_path: Path) -> dict[str, int]:
    sections = {"calc": 0, "plot": 0}
    current = None
    if not snippet_path.exists():
        return {**sections, "total": 0}
    calc_markers = {"# calculation", "'CALCULATION_SECTION'", '"CALCULATION_SECTION"'}
    plot_markers = {"# plotting", "'PLOTTING_SECTION'", '"PLOTTING_SECTION"'}
    for line in snippet_path.read_text().splitlines():
        stripped = line.strip()
        if stripped in calc_markers:
            current = "calc"
            continue
        if stripped in plot_markers:
            current = "plot"
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if current in sections:
            sections[current] += 1
    sections["total"] = sections["calc"] + sections["plot"]
    return sections


def _collect_loc_metrics() -> dict[str, dict[str, int]]:
    return {tool: _count_loc_sections(path) for tool, path in SNIPPET_FILES.items()}


def _plot_summary(summary: dict[str, dict], runs: dict[str, list[RunMetrics]], output_dir: Path) -> None:
    labels = [tool.capitalize() if tool != "fastmdanalysis" else "FastMDAnalysis" for tool in TOOL_ORDER]
    bar_positions = np.arange(len(TOOL_ORDER))

    runtimes = [summary[tool]["elapsed_s"]["mean"] for tool in TOOL_ORDER]
    runtime_err = [summary[tool]["elapsed_s"]["stdev"] for tool in TOOL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bar_positions, runtimes, yerr=runtime_err, capsize=5, color=["#4472C4", "#ED7D31", "#A5A5A5"])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("RMSF Runtime Comparison (Trp-cage)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_summary.png", bbox_inches="tight")
    plt.close(fig)

    memories = [summary[tool]["peak_mem_mb"]["mean"] for tool in TOOL_ORDER]
    memory_err = [summary[tool]["peak_mem_mb"]["stdev"] for tool in TOOL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bar_positions, memories, yerr=memory_err, capsize=5, color=["#4472C4", "#ED7D31", "#A5A5A5"])
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Mean Peak Memory (MB)")
    ax.set_title("RMSF Peak Memory Comparison (Trp-cage)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_data = [[m.elapsed_s for m in runs[tool]] for tool in TOOL_ORDER]
    boxplot_kwargs = dict(
        patch_artist=True,
        boxprops=dict(facecolor="#D9E1F2", color="#2F5597"),
        medianprops=dict(color="#C00000"),
    )
    try:
        ax.boxplot(runtime_data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(runtime_data, labels=labels, **boxplot_kwargs)
    ax.set_ylabel("Runtime per Run (s)")
    ax.set_title("RMSF Runtime Distribution (Trp-cage)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_distribution.png", bbox_inches="tight")
    plt.close(fig)

    loc_calc = [summary[tool]["loc"]["calc"] for tool in TOOL_ORDER]
    loc_plot = [summary[tool]["loc"]["plot"] for tool in TOOL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    calc_bars = ax.bar(bar_positions, loc_calc, color="#4472C4", label="Calculation")
    plot_bars = ax.bar(bar_positions, loc_plot, bottom=loc_calc, color="#ED7D31", label="Plotting")
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Non-comment Lines")
    ax.set_title("RMSF Reference Snippet Size")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for idx, bar in enumerate(plot_bars):
        total = loc_calc[idx] + loc_plot[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            total + 0.1,
            f"{total}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "loc_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mdtraj_traj = md.load(str(TRAJ_FILE), top=str(TOPOLOGY_FILE))
    fast_traj = mdtraj_traj
    universe = mda.Universe(str(TOPOLOGY_FILE), str(TRAJ_FILE))

    results: dict[str, list[RunMetrics]] = {
        "fastmdanalysis": [],
        "mdtraj": [],
        "mdanalysis": [],
    }

    for idx in range(1, REPEATS + 1):
        fast_dir = OUTPUT_ROOT / "fastmdanalysis" / f"rep_{idx:02d}"
        fast_dir.mkdir(parents=True, exist_ok=True)
        _, fast_metrics = _measure_execution("fastmdanalysis", lambda: _run_fastmdanalysis(fast_traj, fast_dir))
        results["fastmdanalysis"].append(fast_metrics)

        mdtraj_dir = OUTPUT_ROOT / "mdtraj" / f"rep_{idx:02d}"
        mdtraj_dir.mkdir(parents=True, exist_ok=True)
        _, mdtraj_metrics = _measure_execution("mdtraj", lambda: _run_mdtraj(mdtraj_traj, mdtraj_dir))
        results["mdtraj"].append(mdtraj_metrics)

        mda_dir = OUTPUT_ROOT / "mdanalysis" / f"rep_{idx:02d}"
        mda_dir.mkdir(parents=True, exist_ok=True)
        _, mda_metrics = _measure_execution("mdanalysis", lambda: _run_mdanalysis(universe, mda_dir))
        results["mdanalysis"].append(mda_metrics)

    summary = {tool: _summarize(metrics) for tool, metrics in results.items()}

    loc_metrics = _collect_loc_metrics()
    for tool, loc in loc_metrics.items():
        summary.setdefault(tool, {})["loc"] = loc

    _plot_summary(summary, results, OUTPUT_ROOT)

    summary_csv_path = OUTPUT_ROOT / "metrics_summary.csv"
    with summary_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "metric", "mean", "stdev", "min", "max"])
        for tool in TOOL_ORDER:
            elapsed = summary[tool]["elapsed_s"]
            writer.writerow([
                tool,
                "elapsed_s",
                f"{elapsed['mean']:.6f}",
                f"{elapsed['stdev']:.6f}",
                f"{elapsed['min']:.6f}",
                f"{elapsed['max']:.6f}",
            ])
            memory = summary[tool]["peak_mem_mb"]
            writer.writerow([
                tool,
                "peak_mem_mb",
                f"{memory['mean']:.6f}",
                f"{memory['stdev']:.6f}",
                f"{memory['min']:.6f}",
                f"{memory['max']:.6f}",
            ])
            loc = summary[tool]["loc"]
            writer.writerow([
                tool,
                "calc_loc",
                loc["calc"],
                loc["calc"],
                loc["calc"],
                loc["calc"],
            ])
            writer.writerow([
                tool,
                "plot_loc",
                loc["plot"],
                loc["plot"],
                loc["plot"],
                loc["plot"],
            ])
            writer.writerow([
                tool,
                "total_loc",
                loc["total"],
                loc["total"],
                loc["total"],
                loc["total"],
            ])

    runs_csv_path = OUTPUT_ROOT / "metrics_runs.csv"
    with runs_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "repeat", "elapsed_s", "peak_mem_mb"])
        for tool in TOOL_ORDER:
            for idx, metrics in enumerate(results[tool], start=1):
                writer.writerow([
                    tool,
                    idx,
                    f"{metrics.elapsed_s:.6f}",
                    f"{metrics.peak_mem_mb:.6f}",
                ])

    payload = {
        "metadata": {
            "trajectory": str(TRAJ_FILE),
            "topology": str(TOPOLOGY_FILE),
            "repeats": REPEATS,
            "atom_selection": ATOM_SELECTION,
        },
        "summary": summary,
        "runs": {
            tool: [m.__dict__ for m in metrics] for tool, metrics in results.items()
        },
        "instrumentation": INSTRUMENT.to_dict(),
    }

    summary_path = OUTPUT_ROOT / "metrics.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    touchpoints_json = OUTPUT_ROOT / "touchpoints.json"
    touchpoints_md = OUTPUT_ROOT / "touchpoints.md"
    touchpoints_json.write_text(json.dumps(INSTRUMENT.to_dict(), indent=2))
    touchpoints_md.write_text(INSTRUMENT.to_markdown())
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
