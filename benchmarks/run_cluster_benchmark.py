from __future__ import annotations

import csv
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
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
from sklearn.cluster import DBSCAN

from fastmdanalysis.analysis import cluster as fast_cluster
from fastmdanalysis.datasets import TrpCage
from instrumentation import Instrumentation


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

TRAJ_FILE = Path(TrpCage.traj)
TOPOLOGY_FILE = Path(TrpCage.top)

OUTPUT_ROOT = PROJECT_ROOT / "benchmarks" / "results" / "cluster_trpcage"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


INSTRUMENT = Instrumentation(TOOL_ORDER)


@dataclass
class RunMetrics:
    elapsed_s: float
    peak_mem_mb: float


def _adjust_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return labels
    minimum = int(labels.min())
    if minimum < 1:
        labels = labels + (1 - minimum)
    return labels


def _measure_execution(tool: str, func):
    INSTRUMENT.record_attempt(tool)
    tracemalloc.start()
    start = time.perf_counter()
    try:
        labels = func()
    except Exception:
        INSTRUMENT.record_exception(tool)
        tracemalloc.stop()
        raise
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 ** 2)
    INSTRUMENT.record_success(tool)
    return labels, RunMetrics(elapsed_s=elapsed, peak_mem_mb=peak_mb)


def _plot_population(labels: np.ndarray, out_path: Path) -> None:
    unique = np.sort(np.unique(labels))
    counts = np.array([np.sum(labels == u) for u in unique])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique, counts, color="#4472C4")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Frames")
    ax.set_title("Cluster Populations")
    ax.set_xticks(unique)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(labels: np.ndarray, out_path: Path) -> None:
    unique = np.sort(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique))
    norm = plt.Normalize(vmin=unique.min() - 0.5, vmax=unique.max() + 0.5)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    data = labels.reshape(1, -1)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title("Cluster Assignment vs Frame")
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, ticks=unique)
    cbar.ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(labels: np.ndarray, out_path: Path) -> None:
    frames = np.arange(len(labels))
    unique = np.sort(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique))
    norm = plt.Normalize(vmin=unique.min() - 0.5, vmax=unique.max() + 0.5)
    fig, ax = plt.subplots(figsize=(10, 3))
    scatter = ax.scatter(frames, np.zeros_like(frames), c=labels, cmap=cmap, norm=norm, s=60)
    ax.set_title("Cluster Timeline Scatter")
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    cbar = fig.colorbar(scatter, ax=ax, ticks=unique)
    cbar.ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_matrix(distances: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(distances, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_title("RMSD Distance Matrix")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("RMSD (nm)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_cluster_artifacts(rep_dir: Path, labels: np.ndarray, distances: np.ndarray) -> None:
    labels_path = rep_dir / "dbscan_labels.dat"
    np.savetxt(labels_path, np.column_stack((np.arange(len(labels)), labels)), fmt="%d", header="frame cluster")
    distance_path = rep_dir / "dbscan_distance_matrix.dat"
    np.savetxt(distance_path, distances, fmt="%.6f", header="RMSD distance matrix")
    _plot_population(labels, rep_dir / "dbscan_pop.png")
    _plot_histogram(labels, rep_dir / "dbscan_traj_hist.png")
    _plot_scatter(labels, rep_dir / "dbscan_traj_scatter.png")
    _plot_distance_matrix(distances, rep_dir / "dbscan_distance_matrix.png")


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


def _run_fastmdanalysis(traj, rep_dir: Path) -> np.ndarray:
    INSTRUMENT.record_modules("fastmdanalysis", [
        "fastmdanalysis.analysis.cluster",
        "fastmdanalysis.datasets",
    ])
    INSTRUMENT.record_functions("fastmdanalysis", [
        "ClusterAnalysis.__init__",
        "ClusterAnalysis.run",
    ])
    before = {p.name for p in rep_dir.glob("*")}
    analysis = fast_cluster.ClusterAnalysis(
        traj,
        methods="dbscan",
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        atoms=ATOM_SELECTION,
        output=str(rep_dir),
    )
    analysis.run()
    after = {p.name for p in rep_dir.glob("*")}
    new_files = [rep_dir / name for name in after - before]
    if new_files:
        INSTRUMENT.record_files("fastmdanalysis", new_files)
    return np.asarray(analysis.results["dbscan"]["labels"], dtype=int)


def _run_mdtraj(traj, rep_dir: Path) -> np.ndarray:
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
    before = {p.name for p in rep_dir.glob("*")}
    atom_indices = traj.topology.select(ATOM_SELECTION) if ATOM_SELECTION is not None else None
    subtraj = traj.atom_slice(atom_indices) if atom_indices is not None else traj
    distances = _compute_mdtraj_distances(subtraj)
    raw_labels = DBSCAN(metric="precomputed", eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(distances)
    labels = _adjust_labels(raw_labels.astype(int))
    _save_cluster_artifacts(rep_dir, labels, distances)
    after = {p.name for p in rep_dir.glob("*")}
    new_files = [rep_dir / name for name in after - before]
    if new_files:
        INSTRUMENT.record_files("mdtraj", new_files)
    return labels


def _run_mdanalysis(universe, frame_indices: np.ndarray, rep_dir: Path) -> np.ndarray:
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
    before = {p.name for p in rep_dir.glob("*")}
    distances = _compute_mdanalysis_distances(universe, ATOM_SELECTION, frame_indices)
    raw_labels = DBSCAN(metric="precomputed", eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(distances)
    labels = _adjust_labels(raw_labels.astype(int))
    _save_cluster_artifacts(rep_dir, labels, distances)
    after = {p.name for p in rep_dir.glob("*")}
    new_files = [rep_dir / name for name in after - before]
    if new_files:
        INSTRUMENT.record_files("mdanalysis", new_files)
    return labels


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
    ax.set_title("Clustering Runtime Comparison (Trp-cage)")
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
    ax.set_title("Clustering Peak Memory Comparison (Trp-cage)")
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
    ax.set_title("Clustering Runtime Distribution (Trp-cage)")
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
    ax.set_title("Clustering Reference Snippet Size")
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
    mdtraj_full = md.load(str(TRAJ_FILE), top=str(TOPOLOGY_FILE))
    frame_indices = np.arange(0, mdtraj_full.n_frames, FRAME_STRIDE)
    if frame_indices.size == 0:
        raise RuntimeError("No frames selected for clustering benchmark; adjust FRAME_STRIDE or dataset")
    if frame_indices.size > MAX_FRAMES:
        frame_indices = frame_indices[:MAX_FRAMES]
    mdtraj_traj = mdtraj_full[frame_indices]
    fast_traj = mdtraj_traj
    universe = mda.Universe(str(TOPOLOGY_FILE), str(TRAJ_FILE))

    run_metrics: dict[str, list[RunMetrics]] = {tool: [] for tool in TOOL_ORDER}
    cluster_counts: dict[str, list[int]] = {tool: [] for tool in TOOL_ORDER}

    for idx in range(1, REPEATS + 1):
        fast_dir = OUTPUT_ROOT / "fastmdanalysis" / f"rep_{idx:02d}"
        fast_dir.mkdir(parents=True, exist_ok=True)
        fast_labels, fast_metrics = _measure_execution("fastmdanalysis", lambda: _run_fastmdanalysis(fast_traj, fast_dir))
        run_metrics["fastmdanalysis"].append(fast_metrics)
        cluster_counts["fastmdanalysis"].append(int(np.unique(fast_labels).size))

        mdtraj_dir = OUTPUT_ROOT / "mdtraj" / f"rep_{idx:02d}"
        mdtraj_dir.mkdir(parents=True, exist_ok=True)
        mdtraj_labels, mdtraj_metrics = _measure_execution("mdtraj", lambda: _run_mdtraj(mdtraj_traj, mdtraj_dir))
        run_metrics["mdtraj"].append(mdtraj_metrics)
        cluster_counts["mdtraj"].append(int(np.unique(mdtraj_labels).size))

        mda_dir = OUTPUT_ROOT / "mdanalysis" / f"rep_{idx:02d}"
        mda_dir.mkdir(parents=True, exist_ok=True)
        mda_labels, mda_metrics = _measure_execution("mdanalysis", lambda: _run_mdanalysis(universe, frame_indices, mda_dir))
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

    runs_csv_path = OUTPUT_ROOT / "metrics_runs.csv"
    with runs_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tool", "repeat", "elapsed_s", "peak_mem_mb", "cluster_count"])
        for tool in TOOL_ORDER:
            for idx, metrics in enumerate(run_metrics[tool], start=1):
                writer.writerow([
                    tool,
                    idx,
                    f"{metrics.elapsed_s:.6f}",
                    f"{metrics.peak_mem_mb:.6f}",
                    cluster_counts[tool][idx - 1],
                ])

    payload = {
        "metadata": {
            "trajectory": str(TRAJ_FILE),
            "topology": str(TOPOLOGY_FILE),
            "repeats": REPEATS,
            "atom_selection": ATOM_SELECTION,
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "frame_stride": FRAME_STRIDE,
            "max_frames": MAX_FRAMES,
            "sampled_frames": int(frame_indices.size),
        },
        "summary": summary,
        "runs": {
            tool: [
                {
                    "elapsed_s": metrics.elapsed_s,
                    "peak_mem_mb": metrics.peak_mem_mb,
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
