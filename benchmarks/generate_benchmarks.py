from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dataset_config import list_datasets

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmarks"
SLIDES_DIR = ROOT / "slides"

BENCHMARK_TASKS = [
    ("RMSD benchmark", BENCHMARK_DIR / "run_rmsd_benchmark.py"),
    ("RMSF benchmark", BENCHMARK_DIR / "run_rmsf_benchmark.py"),
    ("Radius of gyration benchmark", BENCHMARK_DIR / "run_rg_benchmark.py"),
    ("Clustering benchmark", BENCHMARK_DIR / "run_cluster_benchmark.py"),
]

OVERVIEW_TASK = ("Instrumentation aggregation", BENCHMARK_DIR / "aggregate_instrumentation.py")
SLIDES_TASK = ("Benchmark slideshow", SLIDES_DIR / "generate_benchmark_slides.py")


def _run_step(label: str, script_path: Path, *, dry_run: bool = False, extra_args: list[str] | None = None) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} is missing; cannot run {label}")
    print(f"[+] {label}")
    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)
    if dry_run:
        print("    would run:", " ".join(command))
        return
    subprocess.run(command, check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run FastMDAnalysis benchmark suite and regenerate slides")
    available_datasets = sorted({name.lower() for name in list_datasets()})
    parser.add_argument(
        "--dataset",
        default="trpcage",
        type=str.lower,
        choices=available_datasets,
        help="Dataset identifier to process.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Reuse existing benchmark outputs instead of rerunning the benchmark scripts.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip instrumentation aggregation charts.",
    )
    parser.add_argument(
        "--skip-slides",
        action="store_true",
        help="Skip regenerating the PowerPoint deck.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps without executing them.",
    )
    args = parser.parse_args(argv)

    dataset_args = ["--dataset", args.dataset]

    if not args.skip_benchmarks:
        for label, script in BENCHMARK_TASKS:
            _run_step(label, script, dry_run=args.dry_run, extra_args=dataset_args)
    else:
        print("[!] Skipping benchmark execution")

    if not args.skip_overview:
        _run_step(*OVERVIEW_TASK, dry_run=args.dry_run, extra_args=dataset_args)
    else:
        print("[!] Skipping instrumentation aggregation")

    if not args.skip_slides:
        _run_step(*SLIDES_TASK, dry_run=args.dry_run, extra_args=dataset_args)
    else:
        print("[!] Skipping slideshow generation")

    print(f"[✓] Benchmark pipeline complete for dataset '{args.dataset}'")


if __name__ == "__main__":
    main()
