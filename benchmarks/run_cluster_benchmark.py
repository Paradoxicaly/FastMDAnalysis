from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
import types
import argparse
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from sklearn.cluster import DBSCAN

from fastmdanalysis.analysis import cluster as fast_cluster
from dataset_config import get_dataset_config, list_datasets
from instrumentation import Instrumentation
from palette import colors_for, label_for_tool

AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
TITLE_FONT_SIZE = 20
LEGEND_FONT_SIZE = 15
plt.rcParams.update({
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "figure.titlesize": TITLE_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
})


REPEATS = 5
FRAME_STRIDE = 10
MAX_FRAMES = 120
ATOM_SELECTION = "protein and name CA"
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 5
TOOL_ORDER = ["fastmdanalysis", "mdtraj", "mdanalysis"]

SNIPPET_ROOT = PROJECT_ROOT / "benchmarks" / "snippets" / "cluster"
SNIPPET_FILES = {
    "fastmdanalysis": SNIPPET_ROOT / "fastmdanalysis_script.py",
    "mdtraj": SNIPPET_ROOT / "mdtraj_script.py",
    "mdanalysis": SNIPPET_ROOT / "mdanalysis_script.py",
}

DEFAULT_DATASET = "trpcage"
DATASET_SLUG = DEFAULT_DATASET
DATASET_LABEL = ""
TRAJ_FILE = Path()
TOPOLOGY_FILE = Path()
OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results"


def _set_dataset(slug: str) -> None:
    global DATASET_SLUG, DATASET_LABEL, TRAJ_FILE, TOPOLOGY_FILE, OUTPUT_ROOT
    config = get_dataset_config(slug)
    DATASET_SLUG = config.slug
    DATASET_LABEL = config.label
    TRAJ_FILE = Path(config.traj)
    TOPOLOGY_FILE = Path(config.top)
    OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results" / f"cluster_{DATASET_SLUG}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


_set_dataset(DEFAULT_DATASET)


INSTRUMENT = Instrumentation(TOOL_ORDER)


@dataclass
class RunMetrics:
    calc_s: float
    plot_s: float
    total_s: float
    calc_mem_mb: float
    plot_mem_mb: float
    peak_mem_mb: float

    @property
    def elapsed_s(self) -> float:
        return self.total_s


def _adjust_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels
    minimum = int(labels.min())
    if minimum < 1:
        labels = labels + (1 - minimum)
    return labels


def _noop_method(self, *args, **kwargs):
    return None


def _measure_tool(tool: str, runner):
    INSTRUMENT.record_attempt(tool)
    tracemalloc.start()
    start_total = time.perf_counter()
    try:
        values, calc_time, plot_time, generated, calc_mem_mb, plot_mem_mb = runner()
    except Exception:
        INSTRUMENT.record_exception(tool)
        tracemalloc.stop()
        raise
    total_elapsed = time.perf_counter() - start_total
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = max(peak / (1024 ** 2), calc_mem_mb, plot_mem_mb)
    INSTRUMENT.record_success(tool)
    if generated:
        INSTRUMENT.record_files(tool, generated)
    return values, RunMetrics(
        calc_s=calc_time,
        plot_s=plot_time,
        total_s=total_elapsed,
        calc_mem_mb=calc_mem_mb,
        plot_mem_mb=plot_mem_mb,
        peak_mem_mb=peak_mb,
    )


def _plot_population(labels: np.ndarray, dataset_label: str, out_path: Path) -> Path:
    unique = np.sort(np.unique(labels))
    counts = np.array([np.sum(labels == u) for u in unique])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique, counts, color="#4472C4")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Frames")
    ax.set_title(f"Cluster Populations ({dataset_label})")
    ax.set_xticks(unique)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_histogram(labels: np.ndarray, dataset_label: str, out_path: Path) -> Path:
    unique = np.sort(np.unique(labels))
    cmap = plt.get_cmap("tab10", len(unique))
    norm = plt.Normalize(vmin=unique.min() - 0.5, vmax=unique.max() + 0.5)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    data = labels.reshape(1, -1)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(f"Cluster Assignment vs Frame ({dataset_label})")
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=unique)
    cbar.ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_scatter(labels: np.ndarray, dataset_label: str, out_path: Path) -> Path:
    frames = np.arange(len(labels))
    unique = np.sort(np.unique(labels))
    cmap = plt.get_cmap("tab10", len(unique))
    norm = plt.Normalize(vmin=unique.min() - 0.5, vmax=unique.max() + 0.5)
    fig, ax = plt.subplots(figsize=(10, 3))
    scatter = ax.scatter(frames, np.zeros_like(frames), c=labels, cmap=cmap, norm=norm, s=60)
    ax.set_title(f"Cluster Timeline Scatter ({dataset_label})")
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique)
    cbar.ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_distance_matrix(distances: np.ndarray, dataset_label: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(distances, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_title(f"RMSD Distance Matrix ({dataset_label})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("RMSD (nm)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_cluster_artifacts(
    rep_dir: Path,
    labels: np.ndarray,
    distances: np.ndarray,
    dataset_label: str,
    *,
    include_plots: bool,
) -> list[Path]:
    generated: list[Path] = []
    labels_path = rep_dir / "dbscan_labels.dat"
    np.savetxt(labels_path, np.column_stack((np.arange(len(labels)), labels)), fmt="%d", header="frame cluster")
    generated.append(labels_path)
    distance_path = rep_dir / "dbscan_distance_matrix.dat"
    np.savetxt(distance_path, distances, fmt="%.6f", header="RMSD distance matrix")
    generated.append(distance_path)
    if include_plots:
        generated.append(_plot_population(labels, dataset_label, rep_dir / "dbscan_pop.png"))
        generated.append(_plot_histogram(labels, dataset_label, rep_dir / "dbscan_traj_hist.png"))
        generated.append(_plot_scatter(labels, dataset_label, rep_dir / "dbscan_traj_scatter.png"))
        generated.append(_plot_distance_matrix(distances, dataset_label, rep_dir / "dbscan_distance_matrix.png"))
    return generated


def _compute_mdtraj_distances(traj: md.Trajectory) -> np.ndarray:
    n_frames = traj.n_frames
    distances = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        distances[i] = md.rmsd(traj, traj[i])
    return 0.5 * (distances + distances.T)


def _compute_mdanalysis_distances(
    universe: mda.Universe,
    selection: str | None,
    frame_indices: np.ndarray,
) -> np.ndarray:
    atom_group = universe.select_atoms(selection) if selection else universe.atoms
    frame_list = frame_indices.tolist()
    coords = []
    for _ in universe.trajectory[frame_list]:
        coords.append(atom_group.positions.astype(float) / 10.0)
    universe.trajectory[0]
    coords = np.asarray(coords)
    n_frames = coords.shape[0]
    distances = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        for j in range(i, n_frames):
            value = rms.rmsd(coords[i], coords[j], center=True, superposition=True)
            distances[i, j] = distances[j, i] = value
    return distances


def _run_fastmdanalysis(
    traj,
    rep_dir: Path,
    dataset_label: str,
    *,
    timing_only: bool,
    do_plot: bool,
) -> tuple[np.ndarray, float, float, list[Path], float, float]:
    INSTRUMENT.record_modules("fastmdanalysis", [
        "fastmdanalysis.analysis.cluster",
        "fastmdanalysis.datasets",
    ])
    INSTRUMENT.record_functions("fastmdanalysis", [
        "ClusterAnalysis.__init__",
        "ClusterAnalysis.run",
    ])
    analysis = fast_cluster.ClusterAnalysis(
        traj,
        methods="dbscan",
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        atoms=ATOM_SELECTION,
        output=str(rep_dir),
    )
    patched_methods = {}
    for method_name in (
        "_save_data",
        "_plot_population",
        "_plot_cluster_trajectory_histogram",
        "_plot_cluster_trajectory_scatter",
        "_plot_distance_matrix",
        "_plot_dendrogram",
    ):
        if hasattr(analysis, method_name):
            original = getattr(analysis, method_name)
            patched_methods[method_name] = original
            setattr(analysis, method_name, types.MethodType(_noop_method, analysis))

    tracemalloc.reset_peak()
    start_calc = time.perf_counter()
    analysis.run()
    calc_time = time.perf_counter() - start_calc
    calc_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    result = analysis.results.get("dbscan", {})
    labels = np.asarray(result.get("labels", []), dtype=int)
    raw_distances = result.get("distance_matrix")
    if raw_distances is None:
        distances = np.empty((labels.size, labels.size))
    else:
        distances = np.asarray(raw_distances)

    for method_name, original in patched_methods.items():
        setattr(analysis, method_name, original)

    generated: list[Path] = []
    plot_time = 0.0
    plot_peak_mb = 0.0
    if not timing_only and labels.size:
        tracemalloc.reset_peak()
        start_plot = time.perf_counter()
        generated = _save_cluster_artifacts(
            rep_dir,
            labels,
            distances,
            dataset_label,
            include_plots=do_plot,
        )
        plot_time = time.perf_counter() - start_plot
        plot_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)

    return labels, calc_time, plot_time, generated, calc_peak_mb, plot_peak_mb


def _run_mdtraj(
    traj,
    rep_dir: Path,
    dataset_label: str,
    *,
    timing_only: bool,
    do_plot: bool,
) -> tuple[np.ndarray, float, float, list[Path], float, float]:
    INSTRUMENT.record_modules("mdtraj", [
        "mdtraj",
        "numpy",
        "sklearn.cluster",
        "matplotlib.pyplot",
    ])
    INSTRUMENT.record_functions("mdtraj", [
        "md.load",
        "md.rmsd",
        "DBSCAN.fit_predict",
        "_compute_mdtraj_distances",
        "_save_cluster_artifacts",
    ])
    atom_indices = traj.topology.select(ATOM_SELECTION) if ATOM_SELECTION is not None else None
    subtraj = traj.atom_slice(atom_indices) if atom_indices is not None else traj
    tracemalloc.reset_peak()
    start_calc = time.perf_counter()
    distances = _compute_mdtraj_distances(subtraj)
    raw_labels = DBSCAN(metric="precomputed", eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(distances)
    labels = _adjust_labels(raw_labels.astype(int))
    calc_time = time.perf_counter() - start_calc
    calc_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)

    generated: list[Path] = []
    plot_time = 0.0
    plot_peak_mb = 0.0
    if not timing_only:
        tracemalloc.reset_peak()
        start_plot = time.perf_counter()
        generated = _save_cluster_artifacts(
            rep_dir,
            labels,
            distances,
            dataset_label,
            include_plots=do_plot,
        )
        plot_time = time.perf_counter() - start_plot
        plot_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)

    return labels, calc_time, plot_time, generated, calc_peak_mb, plot_peak_mb


def _run_mdanalysis(
    universe,
    frame_indices: np.ndarray,
    rep_dir: Path,
    dataset_label: str,
    *,
    timing_only: bool,
    do_plot: bool,
) -> tuple[np.ndarray, float, float, list[Path], float, float]:
    INSTRUMENT.record_modules("mdanalysis", [
        "MDAnalysis",
        "MDAnalysis.analysis.rms",
        "numpy",
        "sklearn.cluster",
        "matplotlib.pyplot",
    ])
    INSTRUMENT.record_functions("mdanalysis", [
        "rms.rmsd",
        "DBSCAN.fit_predict",
        "_compute_mdanalysis_distances",
        "_save_cluster_artifacts",
    ])
    tracemalloc.reset_peak()
    start_calc = time.perf_counter()
    distances = _compute_mdanalysis_distances(universe, ATOM_SELECTION, frame_indices)
    raw_labels = DBSCAN(metric="precomputed", eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(distances)
    labels = _adjust_labels(raw_labels.astype(int))
    calc_time = time.perf_counter() - start_calc
    calc_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)

    generated: list[Path] = []
    plot_time = 0.0
    plot_peak_mb = 0.0
    if not timing_only:
        tracemalloc.reset_peak()
        start_plot = time.perf_counter()
        generated = _save_cluster_artifacts(
            rep_dir,
            labels,
            distances,
            dataset_label,
            include_plots=do_plot,
        )
        plot_time = time.perf_counter() - start_plot
        plot_peak_mb = tracemalloc.get_traced_memory()[1] / (1024 ** 2)

    return labels, calc_time, plot_time, generated, calc_peak_mb, plot_peak_mb


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "stdev": stdev, "min": min(values), "max": max(values)}


def _summarize(metrics: list[RunMetrics]) -> dict:
    calc = [m.calc_s for m in metrics]
    plot = [m.plot_s for m in metrics]
    total = [m.total_s for m in metrics]
    calc_mem = [m.calc_mem_mb for m in metrics]
    plot_mem = [m.plot_mem_mb for m in metrics]
    memory = [m.peak_mem_mb for m in metrics]
    return {
        "runs": len(metrics),
        "calc_s": _stats(calc),
        "plot_s": _stats(plot),
        "total_s": _stats(total),
        "elapsed_s": _stats(total),
        "calc_mem_mb": _stats(calc_mem),
        "plot_mem_mb": _stats(plot_mem),
        "peak_mem_mb": _stats(memory),
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
    labels = [label_for_tool(tool) for tool in TOOL_ORDER]
    bar_positions = np.arange(len(TOOL_ORDER))
    primary_colors = colors_for(TOOL_ORDER, variant="primary")
    secondary_colors = colors_for(TOOL_ORDER, variant="secondary")

    def _headroom(means: list[float], errs: list[float]) -> float:
        if not means:
            return 1.0
        values = np.asarray(means, dtype=float)
        errors = np.asarray(errs if errs else np.zeros_like(values), dtype=float)
        combined = values + errors
        max_val = float(combined.max()) if combined.size else 0.0
        if max_val <= 0:
            return 1.0
        return max_val * 1.25

    def _nice_ticks(max_value: float) -> list[float]:
        if max_value <= 0:
            return [0.0]
        magnitude = 10 ** math.floor(math.log10(max_value))
        for multiplier in (1, 2, 5, 10):
            step = multiplier * magnitude
            if max_value / step <= 6:
                break
        else:
            step = magnitude
        upper = step * math.ceil(max_value / step)
        count = int(upper / step)
        return [step * idx for idx in range(count + 1)]

    def _apply_yaxis(ax, ymax: float) -> float:
        ticks = _nice_ticks(ymax)
        upper = ticks[-1] if ticks else 0.0
        if upper <= 0:
            upper = 1.0
            ticks = [0.0, upper]
        ax.set_ylim(0, upper)
        ax.set_yticks(ticks)
        return upper

    def _annotate_small(ax, bars, values: list[float], threshold: float = 0.2) -> None:
        for bar, value in zip(bars, values):
            if value <= threshold:
                offset = max(0.01, value * 0.4)
                text_value = f"{value:.2f}"
                if text_value == "0.00" and value > 0:
                    text_value = f"{value:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    text_value,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    def _annotate_small_mem(ax, bars, values: list[float], threshold: float = 1.0) -> None:
        for bar, value in zip(bars, values):
            if value <= threshold and value > 0:
                offset = max(0.04, min(0.2, value * 0.4))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + value + offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    def _edge_color_from_face(bar) -> tuple[float, float, float, float] | str:
        face = bar.get_facecolor()
        if isinstance(face, tuple) and len(face) >= 3:
            rgb = face[:3]
            adjusted = tuple(min(1.0, component * 0.7) for component in rgb)
            return (*adjusted, 1.0)
        return "#666666"

    def _stylize_calc_bars(container) -> None:
        if container is None:
            return
        for bar in container:
            bar.set_linewidth(0.8)
            bar.set_edgecolor(_edge_color_from_face(bar))
            bar.set_alpha(0.9)

    def _stylize_plot_bars(container) -> None:
        if container is None:
            return
        for bar in container:
            bar.set_hatch("//")
            bar.set_edgecolor(_edge_color_from_face(bar))
            bar.set_linewidth(0.8)
            bar.set_alpha(0.65)

    calc_means = [summary[tool]["calc_s"]["mean"] for tool in TOOL_ORDER]
    plot_means = [summary[tool]["plot_s"]["mean"] for tool in TOOL_ORDER]
    plot_display: list[float] = [0.0 for _ in TOOL_ORDER]
    total_display: list[float] = []
    total_err_display: list[float] = []
    for idx, tool in enumerate(TOOL_ORDER):
        total_display.append(calc_means[idx])
        total_err_display.append(summary[tool]["calc_s"]["stdev"])
    limited_ylim = _headroom(total_display, total_err_display)
    total_full_means = [summary[tool]["total_s"]["mean"] for tool in TOOL_ORDER]
    total_full_err = [summary[tool]["total_s"]["stdev"] for tool in TOOL_ORDER]
    shared_ylim = max(limited_ylim, _headroom(total_full_means, total_full_err))
    runtime_means = [summary[tool]["elapsed_s"]["mean"] for tool in TOOL_ORDER]
    runtime_err = [summary[tool]["elapsed_s"]["stdev"] for tool in TOOL_ORDER]
    runtime_ylim = max(_headroom(runtime_means, runtime_err), shared_ylim)

    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_bars = ax.bar(
        bar_positions,
        runtime_means,
        yerr=runtime_err,
        capsize=5,
        color=primary_colors,
        error_kw={"ecolor": "#444444"},
    )
    _stylize_calc_bars(runtime_bars)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title(f"Clustering Runtime Comparison ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    _apply_yaxis(ax, runtime_ylim)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    calc_bar_limited = ax.bar(bar_positions, calc_means, color=primary_colors, label="Computation")
    _stylize_calc_bars(calc_bar_limited)
    legend_handles: list = []
    legend_labels: list[str] = []
    has_plot_component = any(value > 1e-9 for value in plot_display)
    plot_bar_limited = None
    if has_plot_component:
        plot_bar_limited = ax.bar(
            bar_positions,
            plot_display,
            bottom=calc_means,
            color=secondary_colors,
            label="Plotting",
        )
        _stylize_plot_bars(plot_bar_limited)
    if any(total_err_display):
        ax.errorbar(
            bar_positions,
            total_display,
            yerr=np.asarray(total_err_display, dtype=float),
            fmt="none",
            ecolor="#444444",
            capsize=5,
        )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title(f"Clustering Runtime Breakdown ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    if plot_bar_limited is not None:
        legend_handles.append(plot_bar_limited)
        legend_labels.append("Plotting")
    legend_handles.append(calc_bar_limited)
    legend_labels.append("Computation")
    if len(legend_handles) > 1:
        ax.legend(legend_handles, legend_labels)
    _apply_yaxis(ax, runtime_ylim)
    _annotate_small(ax, calc_bar_limited, calc_means)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_summary_split.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    calc_bar_full = ax.bar(bar_positions, calc_means, color=primary_colors, label="Computation")
    _stylize_calc_bars(calc_bar_full)
    full_handles: list = []
    full_labels: list[str] = []
    has_plot_full = any(value > 1e-9 for value in plot_means)
    plot_bar_full = None
    if has_plot_full:
        plot_bar_full = ax.bar(
            bar_positions,
            plot_means,
            bottom=calc_means,
            color=secondary_colors,
            label="Plotting",
        )
        _stylize_plot_bars(plot_bar_full)
        if plot_bar_full is not None:
            full_handles.append(plot_bar_full)
            full_labels.append("Plotting")
    if any(total_full_err):
        total_full_err_array = np.asarray(total_full_err, dtype=float)
        stacked_totals = np.asarray(
            [calc_means[idx] + plot_means[idx] for idx in range(len(calc_means))],
            dtype=float,
        )
        ax.errorbar(
            bar_positions,
            stacked_totals,
            yerr=total_full_err_array,
            fmt="none",
            ecolor="#444444",
            capsize=5,
        )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title(f"Clustering Runtime Breakdown (All Tools) ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    full_handles.append(calc_bar_full)
    full_labels.append("Computation")
    if len(full_handles) > 1:
        ax.legend(full_handles, full_labels)
    _apply_yaxis(ax, runtime_ylim)
    _annotate_small(ax, calc_bar_full, calc_means)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_summary_split_all.png", bbox_inches="tight")
    plt.close(fig)

    calc_mem_means = [summary[tool]["calc_mem_mb"]["mean"] for tool in TOOL_ORDER]
    plot_mem_means = [summary[tool]["plot_mem_mb"]["mean"] for tool in TOOL_ORDER]
    total_mem_means = [summary[tool]["peak_mem_mb"]["mean"] for tool in TOOL_ORDER]
    total_mem_err = [summary[tool]["peak_mem_mb"]["stdev"] for tool in TOOL_ORDER]
    stacked_mem = [calc + plot for calc, plot in zip(calc_mem_means, plot_mem_means)]
    display_envelope = [max(stacked, peak) for stacked, peak in zip(stacked_mem, total_mem_means)]
    fig, ax = plt.subplots(figsize=(8, 5))
    calc_mem_bars = ax.bar(bar_positions, calc_mem_means, color=primary_colors, label="Computation")
    _stylize_calc_bars(calc_mem_bars)
    mem_handles: list = []
    mem_labels: list[str] = []
    has_plot_mem = any(value > 1e-9 for value in plot_mem_means)
    plot_mem_bars = None
    if has_plot_mem:
        plot_mem_bars = ax.bar(
            bar_positions,
            plot_mem_means,
            bottom=calc_mem_means,
            color=secondary_colors,
            label="Plotting",
        )
        _stylize_plot_bars(plot_mem_bars)
        if plot_mem_bars is not None:
            mem_handles.append(plot_mem_bars)
            mem_labels.append("Plotting")
    if any(total_mem_err):
        mem_err_array = np.asarray(total_mem_err, dtype=float)
        symmetric_mem_err = np.vstack([mem_err_array, mem_err_array])
        ax.errorbar(
            bar_positions,
            total_mem_means,
            yerr=symmetric_mem_err,
            fmt="none",
            ecolor="#444444",
            capsize=5,
        )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Peak Memory (MB)")
    ax.set_title(f"Clustering Peak Memory Breakdown ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    mem_handles.append(calc_mem_bars)
    mem_labels.append("Computation")
    peak_markers = ax.scatter(
        bar_positions,
        total_mem_means,
        marker="D",
        color="#333333",
        s=36,
        zorder=5,
        label="Measured peak",
    )
    mem_handles.append(peak_markers)
    mem_labels.append("Measured peak")
    if len(mem_handles) > 1:
        ax.legend(mem_handles, mem_labels)
    ax.set_ylim(0, _headroom(display_envelope, total_mem_err))
    _annotate_small_mem(ax, calc_mem_bars, calc_mem_means)
    if plot_mem_bars is not None:
        _annotate_small_mem(ax, plot_mem_bars, plot_mem_means)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    runtime_data = [[m.total_s for m in runs[tool]] for tool in TOOL_ORDER]
    boxplot_kwargs = dict(
        patch_artist=True,
        boxprops=dict(facecolor="#D9E1F2", color="#2F5597", linewidth=2.0),
        medianprops=dict(color="#C00000", linewidth=2.5),
        whiskerprops=dict(color="#2F5597", linewidth=2.0),
        capprops=dict(color="#2F5597", linewidth=2.0),
        flierprops=dict(
            marker="o",
            markersize=6,
            markerfacecolor="#2F5597",
            markeredgecolor="#2F5597",
            alpha=0.8,
        ),
    )
    try:
        ax.boxplot(runtime_data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(runtime_data, labels=labels, **boxplot_kwargs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Runtime per Run (s)")
    ax.set_title(f"Clustering Runtime Distribution ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    memory_data = [[m.peak_mem_mb for m in runs[tool]] for tool in TOOL_ORDER]
    try:
        ax.boxplot(memory_data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(memory_data, labels=labels, **boxplot_kwargs)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylabel("Peak Memory per Run (MB)")
    ax.set_title(f"Clustering Peak Memory Distribution ({DATASET_LABEL})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / "memory_distribution.png", bbox_inches="tight")
    plt.close(fig)

    loc_calc = [summary[tool]["loc"]["calc"] for tool in TOOL_ORDER]
    loc_plot = [summary[tool]["loc"]["plot"] for tool in TOOL_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.6
    for idx, tool in enumerate(TOOL_ORDER):
        calc_height = loc_calc[idx]
        plot_height = loc_plot[idx]
        calc_container = ax.bar(
            bar_positions[idx],
            calc_height,
            width=bar_width,
            color=primary_colors[idx],
        )
        _stylize_calc_bars(calc_container)
        plot_container = ax.bar(
            bar_positions[idx],
            plot_height,
            width=bar_width,
            bottom=calc_height,
            color=secondary_colors[idx],
        )
        _stylize_plot_bars(plot_container)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Non-comment Lines")
    ax.set_title("Clustering Reference Snippet Size")
    ax.grid(axis="y", alpha=0.3)
    legend_handles = [
        Patch(
            facecolor=secondary_colors[0],
            edgecolor="#444444",
            hatch="//",
            linewidth=0.8,
            label="Plotting",
            alpha=0.65,
        ),
        Patch(
            facecolor=primary_colors[0],
            edgecolor="#444444",
            linewidth=0.8,
            label="Calculation",
            alpha=0.9,
        ),
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(output_dir / "loc_summary.png", bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run clustering benchmark across supported toolkits.")
    available_datasets = sorted({name.lower() for name in list_datasets()})
    parser.add_argument(
        "--dataset",
        default=DATASET_SLUG,
        type=str.lower,
        choices=available_datasets,
        help="Dataset identifier to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help="Number of benchmark repetitions to run.",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="Skip artifact generation and only capture timing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation for individual runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="Maximum number of frames to include after striding.",
    )
    args = parser.parse_args(argv)

    _set_dataset(args.dataset)

    repeats = max(1, args.repeats)
    timing_only = args.timing_only
    do_plot = not args.no_plot
    max_frames = max(1, args.max_frames)

    # Determine frame slicing based on dataset
    frame_limit = None
    if '_500' in DATASET_SLUG:
        # Use first 500 frames
        frame_limit = 500
    elif '_5000' in DATASET_SLUG:
        # Use all frames (no limiting for 5000 frame datasets)
        frame_limit = None
    
    # Load and slice trajectories
    mdtraj_full = md.load(str(TRAJ_FILE), top=str(TOPOLOGY_FILE))
    
    # Apply frame limit before striding if specified
    if frame_limit and mdtraj_full.n_frames > frame_limit:
        mdtraj_full = mdtraj_full[:frame_limit]
    
    frame_indices = np.arange(0, mdtraj_full.n_frames, FRAME_STRIDE)
    if frame_indices.size == 0:
        raise RuntimeError("No frames selected for clustering benchmark; adjust FRAME_STRIDE or dataset")
    if frame_indices.size > max_frames:
        frame_indices = frame_indices[:max_frames]
    mdtraj_traj = mdtraj_full[frame_indices]
    fast_traj = mdtraj_traj
    
    # Load universe with frame limit
    universe = mda.Universe(str(TOPOLOGY_FILE), str(TRAJ_FILE))
    if frame_limit and len(universe.trajectory) > frame_limit:
        universe.trajectory[0:frame_limit]

    run_metrics: dict[str, list[RunMetrics]] = {tool: [] for tool in TOOL_ORDER}
    cluster_counts: dict[str, list[int]] = {tool: [] for tool in TOOL_ORDER}

    for idx in range(1, repeats + 1):
        fast_dir = OUTPUT_ROOT / "fastmdanalysis" / f"rep_{idx:02d}"
        fast_dir.mkdir(parents=True, exist_ok=True)
        fast_labels, fast_metrics = _measure_tool(
            "fastmdanalysis",
            lambda rep_dir=fast_dir: _run_fastmdanalysis(
                fast_traj,
                rep_dir,
                DATASET_LABEL,
                timing_only=timing_only,
                do_plot=do_plot,
            ),
        )
        run_metrics["fastmdanalysis"].append(fast_metrics)
        cluster_counts["fastmdanalysis"].append(int(np.unique(fast_labels).size))

        mdtraj_dir = OUTPUT_ROOT / "mdtraj" / f"rep_{idx:02d}"
        mdtraj_dir.mkdir(parents=True, exist_ok=True)
        mdtraj_labels, mdtraj_metrics = _measure_tool(
            "mdtraj",
            lambda rep_dir=mdtraj_dir: _run_mdtraj(
                mdtraj_traj,
                rep_dir,
                DATASET_LABEL,
                timing_only=timing_only,
                do_plot=do_plot,
            ),
        )
        run_metrics["mdtraj"].append(mdtraj_metrics)
        cluster_counts["mdtraj"].append(int(np.unique(mdtraj_labels).size))

        mda_dir = OUTPUT_ROOT / "mdanalysis" / f"rep_{idx:02d}"
        mda_dir.mkdir(parents=True, exist_ok=True)
        mda_labels, mda_metrics = _measure_tool(
            "mdanalysis",
            lambda rep_dir=mda_dir: _run_mdanalysis(
                universe,
                frame_indices,
                rep_dir,
                DATASET_LABEL,
                timing_only=timing_only,
                do_plot=do_plot,
            ),
        )
        run_metrics["mdanalysis"].append(mda_metrics)
        cluster_counts["mdanalysis"].append(int(np.unique(mda_labels).size))

    summary = {tool: _summarize(metrics) for tool, metrics in run_metrics.items()}

    loc_metrics = _collect_loc_metrics()
    for tool, loc in loc_metrics.items():
        summary.setdefault(tool, {})["loc"] = loc

    for tool in TOOL_ORDER:
        counts = cluster_counts[tool]
        summary[tool]["cluster_count"] = {
            "mean": statistics.mean(counts),
            "stdev": statistics.stdev(counts) if len(counts) > 1 else 0.0,
            "min": min(counts),
            "max": max(counts),
        }

    _plot_summary(summary, run_metrics, OUTPUT_ROOT)

    summary_csv_path = OUTPUT_ROOT / "metrics_summary.csv"
    with summary_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "metric", "mean", "stdev", "min", "max"])
        for tool in TOOL_ORDER:
            for metric_name in ("elapsed_s", "total_s", "calc_mem_mb", "plot_mem_mb", "peak_mem_mb"):
                stats = summary[tool][metric_name]
                writer.writerow([
                    tool,
                    metric_name,
                    f"{stats['mean']:.6f}",
                    f"{stats['stdev']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                ])
            clusters = summary[tool]["cluster_count"]
            writer.writerow([
                tool,
                "cluster_count",
                f"{clusters['mean']:.6f}",
                f"{clusters['stdev']:.6f}",
                f"{clusters['min']:.6f}",
                f"{clusters['max']:.6f}",
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

    breakdown_summary_path = OUTPUT_ROOT / "metrics_summary_breakdown.csv"
    with breakdown_summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "metric", "mean", "stdev", "min", "max"])
        for tool in TOOL_ORDER:
            for metric_name in ("calc_s", "plot_s", "total_s", "calc_mem_mb", "plot_mem_mb"):
                stats = summary[tool][metric_name]
                writer.writerow([
                    tool,
                    metric_name,
                    f"{stats['mean']:.6f}",
                    f"{stats['stdev']:.6f}",
                    f"{stats['min']:.6f}",
                    f"{stats['max']:.6f}",
                ])

    runs_csv_path = OUTPUT_ROOT / "metrics_runs.csv"
    with runs_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "repeat", "elapsed_s", "calc_mem_mb", "plot_mem_mb", "peak_mem_mb", "cluster_count"])
        for tool in TOOL_ORDER:
            for idx, metrics in enumerate(run_metrics[tool], start=1):
                writer.writerow([
                    tool,
                    idx,
                    f"{metrics.elapsed_s:.6f}",
                    f"{metrics.calc_mem_mb:.6f}",
                    f"{metrics.plot_mem_mb:.6f}",
                    f"{metrics.peak_mem_mb:.6f}",
                    cluster_counts[tool][idx - 1],
                ])

    breakdown_runs_path = OUTPUT_ROOT / "metrics_runs_breakdown.csv"
    with breakdown_runs_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "repeat", "calc_s", "plot_s", "total_s", "calc_mem_mb", "plot_mem_mb", "peak_mem_mb", "cluster_count"])
        for tool in TOOL_ORDER:
            for idx, metrics in enumerate(run_metrics[tool], start=1):
                writer.writerow([
                    tool,
                    idx,
                    f"{metrics.calc_s:.6f}",
                    f"{metrics.plot_s:.6f}",
                    f"{metrics.total_s:.6f}",
                    f"{metrics.calc_mem_mb:.6f}",
                    f"{metrics.plot_mem_mb:.6f}",
                    f"{metrics.peak_mem_mb:.6f}",
                    cluster_counts[tool][idx - 1],
                ])

    payload = {
        "metadata": {
            "trajectory": str(TRAJ_FILE),
            "topology": str(TOPOLOGY_FILE),
            "dataset": DATASET_SLUG,
            "dataset_label": DATASET_LABEL,
            "repeats": repeats,
            "atom_selection": ATOM_SELECTION,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "frame_stride": FRAME_STRIDE,
            "max_frames": max_frames,
            "sampled_frames": int(frame_indices.size),
        },
        "summary": summary,
        "runs": {
            tool: [
                {
                    **asdict(metrics),
                    "elapsed_s": metrics.elapsed_s,
                    "cluster_count": cluster_counts[tool][idx],
                }
                for idx, metrics in enumerate(run_metrics[tool])
            ]
            for tool in TOOL_ORDER
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
