from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "src" / "fastmdanalysis" / "data"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    slug: str
    label: str
    traj: Path
    top: Path


_DATASETS: Dict[str, DatasetConfig] = {}


def _register_dataset(key: str, slug: str, label: str, traj_rel: str, top_rel: str) -> None:
    traj_path = (DATA_ROOT / traj_rel).resolve()
    top_path = (DATA_ROOT / top_rel).resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found for dataset '{key}': {traj_path}")
    if not top_path.exists():
        raise FileNotFoundError(f"Topology not found for dataset '{key}': {top_path}")
    config = DatasetConfig(
        key=key,
        slug=slug,
        label=label,
        traj=traj_path,
        top=top_path,
    )
    _DATASETS[slug] = config
    _DATASETS[key] = config


_register_dataset(
    key="trpcage",
    slug="trpcage",
    label="Trp-cage",
    traj_rel="trp_cage.dcd",
    top_rel="trp_cage.pdb",
)

_register_dataset(
    key="ubiquitin",
    slug="ubiquitin",
    label="Ubiquitin",
    traj_rel="ubiquitin/Q95.dcd",
    top_rel="ubiquitin/topology.pdb",
)

_register_dataset(
    key="ubiquitin99",
    slug="ubiquitin99",
    label="Ubiquitin 99",
    traj_rel="ubiquitin/Q99.dcd",
    top_rel="ubiquitin/topology.pdb",
)


def get_dataset_config(identifier: str) -> DatasetConfig:
    try:
        return _DATASETS[identifier.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{identifier}'. Available: {', '.join(sorted(set(_DATASETS.keys())))}") from exc


def list_datasets() -> Iterable[str]:
    seen = set()
    for key in _DATASETS:
        if key in seen:
            continue
        seen.add(key)
        yield key
