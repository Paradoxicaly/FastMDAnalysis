from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset_config import get_dataset_config, list_datasets
from palette import color_for, colors_for, label_for_tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "benchmarks" / "results"

BENCHMARK_KEYS = ("rmsd", "rmsf", "rg", "cluster")
TOOLS = ["fastmdanalysis", "mdtraj", "mdanalysis"]

DEFAULT_DATASET = "trpcage"
DATASET_SLUG = DEFAULT_DATASET
DATASET_LABEL = ""
BENCHMARK_DIRS: Dict[str, Path] = {}
OUTPUT_ROOT = RESULTS_ROOT / "overview"


def _set_dataset(slug: str) -> None:
    global DATASET_SLUG, DATASET_LABEL, BENCHMARK_DIRS, OUTPUT_ROOT
    config = get_dataset_config(slug)
    DATASET_SLUG = config.slug
    DATASET_LABEL = config.label
    BENCHMARK_DIRS = {
        key: RESULTS_ROOT / f"{key}_{DATASET_SLUG}"
        for key in BENCHMARK_KEYS
    }
    OUTPUT_ROOT = RESULTS_ROOT / f"overview_{DATASET_SLUG}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


_set_dataset(DEFAULT_DATASET)


def _load_touchpoints(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _select_external_modules(tool: str, modules: Iterable[str]) -> set[str]:
    external: set[str] = set()
    tool_lower = tool.lower()
    for module in modules:
        module_lower = module.lower()
        if tool_lower == "fastmdanalysis":
            if not module_lower.startswith("fastmdanalysis"):
                external.add(module)
        elif tool_lower == "mdtraj":
            if not module_lower.startswith("mdtraj"):
                external.add(module)
        elif tool_lower == "mdanalysis":
            if not module_lower.startswith("mdanalysis"):
                external.add(module)
        else:
            if not module_lower.startswith(tool_lower):
                external.add(module)
    return external


def _render_bar_chart(
    title: str,
    ylabel: str,
    labels: list[str],
    values: Dict[str, list[float]],
    filename: str,
    legend_title: str | None = None,
    stacked: bool = False,
    show_labels: bool = False,
    series_colors: Dict[str, list[str]] | None = None,
) -> None:
    x = np.arange(len(labels))
    width = 0.25 if not stacked else 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_info: list[tuple] = []
    legend_entries: list[tuple] = []

    items = list(values.items())

    if stacked:
        bottom = np.zeros(len(labels))
        for series_name, series_values in items:
            baseline = bottom.copy()
            color = None
            if series_colors and series_name in series_colors:
                color = series_colors[series_name]
            container = ax.bar(x, series_values, width=width, bottom=bottom, label=series_name, color=color)
            bar_info.append((container, baseline))
            bottom += np.array(series_values)
            legend_entries.append((container, series_name, any(abs(val) > 1e-9 for val in series_values)))
    else:
        series_count = len(items)
        offsets = [0.0] if series_count == 1 else np.linspace(-width, width, series_count)
        for offset, (series_name, series_values) in zip(offsets, items):
            color = None
            if series_colors and series_name in series_colors:
                color = series_colors[series_name]
            container = ax.bar(x + offset, series_values, width=width, label=series_name, color=color)
            bar_info.append((container, None))
            legend_entries.append((container, series_name, any(abs(val) > 1e-9 for val in series_values)))

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    active_entries = [(container, name) for container, name, active in legend_entries if active]
    should_show_legend = False
    if legend_title is not None:
        should_show_legend = bool(active_entries) or len(legend_entries) > 0
    elif len(active_entries) > 1:
        should_show_legend = True

    if should_show_legend:
        handles, labels_for_legend = zip(*active_entries) if active_entries else zip(*[(legend_entries[0][0], legend_entries[0][1])])
        legend = ax.legend(handles, labels_for_legend)
        if legend_title:
            legend.set_title(legend_title)

    if show_labels:
        ax.relim()
        ax.autoscale_view()
        ylim_min, ylim_max = ax.get_ylim()
        upward_offset = max((ylim_max - ylim_min) * 0.02, 0.05)
        downward_offset = max((ylim_max - ylim_min) * 0.03, 0.06)

        for container, baseline in bar_info:
            bars = list(container)
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                base = baseline[idx] if baseline is not None else 0.0

                if height > 0:
                    offset = min(height * 0.15, downward_offset)
                    y = base + height - offset
                    if y <= base:
                        y = base + height + upward_offset
                        va = "bottom"
                    else:
                        va = "top"
                else:
                    y = base + upward_offset
                    va = "bottom"

                value = height
                label = f"{value:.2f}" if abs(value - round(value)) > 1e-6 else str(int(round(value)))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    label,
                    ha="center",
                    va=va,
                    fontsize=10,
                )

    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / filename, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate instrumentation metrics across benchmarks.")
    available_datasets = sorted({name.lower() for name in list_datasets()})
    parser.add_argument(
        "--dataset",
        default=DATASET_SLUG,
        type=str.lower,
        choices=available_datasets,
        help="Dataset identifier whose benchmark outputs should be processed.",
    )
    args = parser.parse_args(argv)

    _set_dataset(args.dataset)

    missing = [str(path) for path in BENCHMARK_DIRS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing benchmark outputs for dataset '{DATASET_SLUG}'. Expected directories: {', '.join(missing)}"
        )

    per_benchmark: Dict[str, Dict[str, Dict[str, object]]] = {}
    aggregated_modules: Dict[str, set[str]] = {tool: set() for tool in TOOLS}
    aggregated_functions: Dict[str, set[str]] = {tool: set() for tool in TOOLS}
    external_modules: Dict[str, set[str]] = {tool: set() for tool in TOOLS}
    attempts: Dict[str, int] = defaultdict(int)
    successes: Dict[str, int] = defaultdict(int)
    exceptions: Dict[str, int] = defaultdict(int)
    loc_totals: Dict[str, Dict[str, int]] = {tool: {"calc": 0, "plot": 0} for tool in TOOLS}
    runtime_totals: Dict[str, Dict[str, float]] = {
        tool: {"calc": 0.0, "plot": 0.0, "overhead": 0.0, "total": 0.0}
        for tool in TOOLS
    }
    peak_mem_totals: Dict[str, Dict[str, float]] = {
        tool: {"calc": 0.0, "plot": 0.0, "total": 0.0}
        for tool in TOOLS
    }

    for benchmark, bench_dir in BENCHMARK_DIRS.items():
        touchpoints = _load_touchpoints(bench_dir / "touchpoints.json")
        metrics = _load_metrics(bench_dir / "metrics.json")
        benchmark_entry: Dict[str, Dict[str, object]] = {}

        for tool in TOOLS:
            tp = touchpoints.get(tool, {})
            modules = tp.get("modules", [])
            functions = tp.get("functions", [])
            attempts[tool] += tp.get("attempts", 0)
            successes[tool] += tp.get("successes", 0)
            exceptions[tool] += tp.get("exceptions", 0)

            aggregated_modules[tool].update(modules)
            aggregated_functions[tool].update(functions)
            external_modules[tool].update(_select_external_modules(tool, modules))

            benchmark_entry[tool] = {
                "modules": len(modules),
                "functions": len(functions),
                "attempts": tp.get("attempts", 0),
                "successes": tp.get("successes", 0),
                "exceptions": tp.get("exceptions", 0),
            }

            summary = metrics.get("summary", {}).get(tool, {})
            loc = summary.get("loc", {})
            loc_totals[tool]["calc"] += int(loc.get("calc", 0))
            loc_totals[tool]["plot"] += int(loc.get("plot", 0))

            calc_runtime = float(summary.get("calc_s", {}).get("mean", 0.0))
            plot_runtime = float(summary.get("plot_s", {}).get("mean", 0.0))
            total_runtime = float(summary.get("total_s", {}).get("mean", calc_runtime + plot_runtime))
            overhead_runtime = max(total_runtime - calc_runtime - plot_runtime, 0.0)
            runtime_totals[tool]["calc"] += calc_runtime
            runtime_totals[tool]["plot"] += plot_runtime
            runtime_totals[tool]["overhead"] += overhead_runtime
            runtime_totals[tool]["total"] += total_runtime

            calc_mem = float(summary.get("calc_mem_mb", {}).get("mean", 0.0))
            peak_mem = float(summary.get("peak_mem_mb", {}).get("mean", calc_mem))
            plot_mem = max(peak_mem - calc_mem, 0.0)
            peak_mem_totals[tool]["calc"] += calc_mem
            peak_mem_totals[tool]["plot"] += plot_mem
            peak_mem_totals[tool]["total"] += peak_mem

        per_benchmark[benchmark] = benchmark_entry

    aggregated_summary = {
        tool: {
            "modules": len(aggregated_modules[tool]),
            "functions": len(aggregated_functions[tool]),
            "external_modules": len(external_modules[tool]),
            "attempts": attempts[tool],
            "successes": successes[tool],
            "exceptions": exceptions[tool],
        }
        for tool in TOOLS
    }

    aggregated_details = {
        tool: {
            "external_modules": sorted(external_modules[tool]),
        }
        for tool in TOOLS
    }

    payload = {
        "per_benchmark": per_benchmark,
        "aggregated": aggregated_summary,
        "loc_totals": loc_totals,
        "runtime_totals": runtime_totals,
        "peak_mem_totals": peak_mem_totals,
        "aggregated_details": aggregated_details,
        "metadata": {
            "benchmarks": list(BENCHMARK_DIRS.keys()),
            "tools": TOOLS,
            "dataset": DATASET_SLUG,
            "dataset_label": DATASET_LABEL,
        },
    }

    summary_path = OUTPUT_ROOT / "instrumentation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    for metric_key in ("modules", "functions"):
        values = {tool: [] for tool in TOOLS}
        labels = []
        for benchmark in BENCHMARK_DIRS.keys():
            labels.append(benchmark.upper())
            entry = per_benchmark.get(benchmark, {})
            for tool in TOOLS:
                values[tool].append(entry.get(tool, {}).get(metric_key, 0))
        display_values = {label_for_tool(tool): values[tool] for tool in TOOLS}
        series_colors = {
            label_for_tool(tool): [color_for(tool)] * len(labels)
            for tool in TOOLS
        }
        _render_bar_chart(
            title=f"{metric_key.title()} touched per benchmark",
            ylabel=f"Distinct {metric_key}",
            labels=labels,
            values=display_values,
            filename=f"{metric_key}_per_benchmark.png",
            legend_title="Tool",
            series_colors=series_colors,
        )

    agg_labels = [label_for_tool(tool) for tool in TOOLS]
    metric_schema = [
        ("modules", "Modules"),
        ("functions", "Functions"),
        ("external_modules", "External imports"),
    ]
    metric_labels = [label for _, label in metric_schema]
    overview_values: Dict[str, list[float]] = {}
    series_colors_overview: Dict[str, list[str]] = {}
    for tool, display in zip(TOOLS, agg_labels):
        tool_metrics = aggregated_summary.get(tool, {})
        overview_values[display] = [float(tool_metrics.get(metric_key, 0)) for metric_key, _ in metric_schema]
        series_colors_overview[display] = [color_for(tool)] * len(metric_schema)

    _render_bar_chart(
        title="Instrumentation overview metrics",
        ylabel="Count",
        labels=metric_labels,
        values=overview_values,
        filename="instrumentation_overview.png",
        legend_title="Tool",
        show_labels=True,
        series_colors=series_colors_overview,
    )

    for benchmark, entry in per_benchmark.items():
        local_labels = [label for _, label in metric_schema[:2]]
        local_values: Dict[str, list[float]] = {}
        local_colors: Dict[str, list[str]] = {}
        for tool, display in zip(TOOLS, agg_labels):
            stats = entry.get(tool, {})
            local_values[display] = [float(stats.get("modules", 0)), float(stats.get("functions", 0))]
            local_colors[display] = [color_for(tool)] * len(local_labels)

        _render_bar_chart(
            title=f"{benchmark.upper()} instrumentation overview",
            ylabel="Count",
            labels=local_labels,
            values=local_values,
            filename=f"instrumentation_{benchmark}.png",
            legend_title="Tool",
            show_labels=True,
            series_colors=local_colors,
        )

    _render_bar_chart(
        title="External module dependencies",
        ylabel="Modules",
        labels=agg_labels,
        values={"External": [aggregated_summary[tool]["external_modules"] for tool in TOOLS]},
        filename="external_modules.png",
        show_labels=True,
        series_colors={
            "External": colors_for(TOOLS, variant="primary"),
        },
    )

    loc_values = {
        "Calculation": [loc_totals[tool]["calc"] for tool in TOOLS],
        "Plotting": [loc_totals[tool]["plot"] for tool in TOOLS],
    }
    _render_bar_chart(
        title="Snippet lines of code (calc vs plot) across benchmarks",
        ylabel="Lines",
        labels=agg_labels,
        values=loc_values,
        filename="loc_totals.png",
        legend_title="Section",
        stacked=True,
        series_colors={
            "Calculation": colors_for(TOOLS, variant="primary"),
            "Plotting": colors_for(TOOLS, variant="secondary"),
        },
    )

    runtime_values = {
        "Computation": [runtime_totals[tool]["calc"] for tool in TOOLS],
        "Plotting": [runtime_totals[tool]["plot"] for tool in TOOLS],
    }
    runtime_colors = {
        "Computation": colors_for(TOOLS, variant="primary"),
        "Plotting": colors_for(TOOLS, variant="secondary"),
    }
    if any(runtime_totals[tool]["overhead"] > 1e-6 for tool in TOOLS):
        runtime_values["Overhead"] = [runtime_totals[tool]["overhead"] for tool in TOOLS]
        runtime_colors["Overhead"] = ["#7F7F7F"] * len(TOOLS)
    _render_bar_chart(
        title="Aggregate runtime footprint",
        ylabel="Seconds",
        labels=agg_labels,
        values=runtime_values,
        filename="runtime_totals.png",
        legend_title="Component",
        stacked=True,
        series_colors=runtime_colors,
    )

    peak_mem_values = {
        "Computation": [peak_mem_totals[tool]["calc"] for tool in TOOLS],
        "Plotting": [peak_mem_totals[tool]["plot"] for tool in TOOLS],
    }
    peak_mem_colors = {
        "Computation": colors_for(TOOLS, variant="primary"),
        "Plotting": colors_for(TOOLS, variant="secondary"),
    }
    _render_bar_chart(
        title="Aggregate peak memory footprint",
        ylabel="Megabytes",
        labels=agg_labels,
        values=peak_mem_values,
        filename="peak_mem_totals.png",
        legend_title="Component",
        stacked=True,
        series_colors=peak_mem_colors,
    )

    print(f"Wrote instrumentation overview to {summary_path}")


if __name__ == "__main__":
    main()
