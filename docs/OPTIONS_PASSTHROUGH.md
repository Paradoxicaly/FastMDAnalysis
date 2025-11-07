# Permissive Options Passthrough

FastMDAnalysis now supports permissive options passthrough with alias mapping, allowing users to supply MDTraj and scikit-learn compatible parameters across all analyses. This feature improves API/CLI parity, usability, and reproducibility.

## Overview

The options passthrough system provides:

1. **Alias Mapping**: Use MDTraj/scikit-learn parameter names directly (e.g., `ref` instead of `reference_frame`)
2. **Permissive Mode**: Unknown options are logged as warnings (default behavior)
3. **Strict Mode**: Unknown options raise errors for strict validation
4. **Pre/Post Hooks**: Framework for custom parameter processing (e.g., per-residue aggregation)

## Quick Start

### Using Aliases in Python API

```python
from fastmdanalysis import FastMDAnalysis

# Load trajectory
fmda = FastMDAnalysis("trajectory.dcd", "topology.pdb")

# Use aliases directly - they "just work"
rmsd = fmda.rmsd(
    ref=0,                      # alias for reference_frame
    selection="protein and name CA",  # alias for atoms
    align=True
)

rmsf = fmda.rmsf(
    selection="protein and name CA",  # alias for atoms
    per_residue=True                  # new feature: aggregate to residues
)
```

### Using Aliases in CLI with YAML

Create an options file (`options.yaml`):

```yaml
rmsd:
  ref: 0
  atoms: "protein and name CA"
  align: true

rmsf:
  selection: "protein and name CA"
  per_residue: true
```

Run with the options file:

```bash
fastmda analyze -traj trajectory.dcd -top topology.pdb --options options.yaml
```

### Using Strict Mode

Enable strict mode to validate all options:

```bash
# CLI
fastmda analyze -traj trajectory.dcd -top topology.pdb --options options.yaml --strict

# Python API
results = fmda.analyze(
    include=["rmsd", "rmsf"],
    options=options_dict,
    strict=True  # Raise errors for unknown options
)
```

## Supported Aliases by Analysis

### RMSD

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `ref` | `reference_frame` | Reference frame index |
| `reference` | `reference_frame` | Reference frame index |
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |

**Example:**
```python
rmsd = fmda.rmsd(ref=0, selection="protein and name CA", align=True)
```

### RMSF

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |

**New Feature: `per_residue`**
```python
rmsf = fmda.rmsf(
    atoms="protein and name CA",
    per_residue=True  # Aggregate to per-residue RMSF
)
```

### Radius of Gyration (RG)

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |

**New Parameter: `mass_weighted`**
```python
rg = fmda.rg(atoms="protein", mass_weighted=False)
```

### Hydrogen Bonds

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |
| `distance_cutoff_nm` | `distance` | Distance cutoff in nm |
| `angle_cutoff_deg` | `angle` | Angle cutoff in degrees |

**New Parameters:**
```python
hbonds = fmda.hbonds(
    atoms="protein",
    distance_cutoff_nm=0.25,  # or just distance=0.25
    angle_cutoff_deg=120,     # or just angle=120
    periodic=True,            # Use PBC
    sidechain_only=False,     # All H-bonds
    exclude_water=True        # Exclude water molecules
)
```

### Secondary Structure (SS)

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |

**New Parameters:**
```python
ss = fmda.ss(
    atoms="protein",
    algorithm="dssp",              # Algorithm (currently only DSSP)
    mkdssp_path="/path/to/mkdssp"  # Optional custom path
)
```

### SASA

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |
| `probe_radius_nm` | `probe_radius` | Probe radius in nm |

**New Parameters:**
```python
sasa = fmda.sasa(
    atoms="protein",
    probe_radius_nm=0.14,   # or probe_radius=0.14
    n_sphere_points=960     # Number of sphere points
)
```

### Dimensionality Reduction

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `method` | `methods` | Single method name |
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |
| `perplexity` | `tsne_perplexity` | t-SNE perplexity |
| `n_iter` | `max_iter` | Maximum iterations |
| `tsne_max_iter` | `max_iter` | Maximum iterations |
| `dissimilarity` | `metric` | MDS metric/dissimilarity |

**Examples:**
```python
# PCA
dimred = fmda.dimred(
    method="pca",              # or methods="pca"
    n_components=2,
    atoms="protein and name CA",
    random_state=42
)

# t-SNE
dimred = fmda.dimred(
    method="tsne",
    n_components=2,
    perplexity=30,             # alias for tsne_perplexity
    max_iter=1000,             # or n_iter=1000
    random_state=42
)

# MDS
dimred = fmda.dimred(
    method="mds",
    n_components=2,
    metric="euclidean",        # or dissimilarity="euclidean"
    random_state=42
)
```

### Clustering

| Alias | Canonical Parameter | Description |
|-------|-------------------|-------------|
| `method` | `methods` | Single method name |
| `atoms` | `atoms` | Atom selection string |
| `selection` | `atoms` | Atom selection string |
| `atom_indices` | `atoms` | Atom selection string |
| `linkage` | `linkage_method` | Hierarchical linkage method |

**Examples:**
```python
# KMeans
cluster = fmda.cluster(
    method="kmeans",           # or methods="kmeans"
    n_clusters=6,
    atoms="protein and name CA",
    random_state=42,
    n_init="auto"              # or n_init=10
)

# Hierarchical
cluster = fmda.cluster(
    method="hierarchical",
    n_clusters=6,
    linkage="ward",            # alias for linkage_method
    atoms="protein and name CA"
)

# DBSCAN
cluster = fmda.cluster(
    method="dbscan",
    eps=0.4,                   # Distance threshold in nm
    min_samples=5,
    atoms="protein and name CA"
)
```

## Strict Mode

### When to Use Strict Mode

Use strict mode when you want to:
- Validate all options are recognized
- Catch typos in parameter names
- Ensure reproducibility with exact parameters
- Debug option passing issues

### Example with Strict Mode

```python
# This will raise an error if any unknown options are provided
try:
    results = fmda.analyze(
        include=["rmsd", "rmsf"],
        options={
            "rmsd": {"ref": 0, "atoms": "protein", "unknown_param": "value"},
            "rmsf": {"atoms": "protein"}
        },
        strict=True
    )
except ValueError as e:
    print(f"Invalid options: {e}")
```

### Non-Strict Mode (Default)

In non-strict mode (default), unknown options are logged as warnings but do not stop execution:

```python
# Unknown options logged but not raised
results = fmda.analyze(
    include=["rmsd"],
    options={
        "rmsd": {"ref": 0, "atoms": "protein", "typo_param": "value"}
    },
    strict=False  # default
)
# Warning logged: "Unknown options (not accepted by callable): ['typo_param']"
```

## Complete YAML Example

See `examples/options_example.yaml` for a comprehensive example with all analyses:

```bash
# Run with all analyses configured via YAML
fastmda analyze -traj trajectory.dcd -top topology.pdb --options examples/options_example.yaml

# Run with strict validation
fastmda analyze -traj trajectory.dcd -top topology.pdb --options examples/options_example.yaml --strict
```

## Advanced: Pre/Post Hooks

The options forwarding system includes a framework for pre and post processing hooks. Currently implemented hooks include:

### Per-Residue Aggregation (RMSF)

The `per_residue` option in RMSF uses a post-hook to aggregate per-atom RMSF values to per-residue:

```python
rmsf = fmda.rmsf(
    atoms="protein and name CA",
    per_residue=True
)

# Results include both per-atom and per-residue RMSF
print(rmsf.results["rmsf"])           # Per-atom RMSF
print(rmsf.results["rmsf_per_residue"])  # Per-residue RMSF
```

## Backward Compatibility

All existing code continues to work unchanged. The new features are additive:

```python
# Old style (still works)
rmsd = fmda.rmsd(reference_frame=0, atoms="protein and name CA", align=True)

# New style with aliases (also works)
rmsd = fmda.rmsd(ref=0, selection="protein and name CA", align=True)

# Mixed style (works too)
rmsd = fmda.rmsd(ref=0, atoms="protein and name CA", align=True)
```

## Benefits

1. **Familiar API**: Use MDTraj and scikit-learn parameter names directly
2. **Reduced Learning Curve**: No need to learn FastMDAnalysis-specific names
3. **Improved Reproducibility**: Copy parameters from MDTraj scripts directly
4. **Flexible Configuration**: YAML files with intuitive parameter names
5. **Better Error Messages**: Strict mode catches parameter typos early

## Implementation Notes

- Aliases are resolved before forwarding to analysis constructors
- Only parameters accepted by the target function are forwarded
- Unknown parameters are logged (non-strict) or raise errors (strict)
- The `strict` flag propagates from `analyze()` to all individual analyses
- All alias mappings are documented in module docstrings
