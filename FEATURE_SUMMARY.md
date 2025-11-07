# Feature Implementation Summary: Permissive Options Passthrough

## Overview

Successfully implemented a comprehensive permissive options passthrough system for FastMDAnalysis that allows users to supply MDTraj and scikit-learn compatible parameters across all analyses with automatic alias mapping and pre/post hook support.

## Implementation Details

### Core Components

#### 1. Options Forwarding Utility (`src/fastmdanalysis/utils/options.py`)
- **OptionsForwarder class**: Main utility handling alias resolution, signature introspection, and hook processing
- **Signature introspection**: Automatically forwards only accepted parameters to target functions
- **Alias mapping**: Translates user-friendly names to canonical parameters (e.g., `ref` → `reference_frame`)
- **Pre/Post hooks**: Framework for custom parameter processing
- **Strict mode**: Toggle between permissive (log warnings) and strict (raise errors) validation

**Key Methods:**
- `apply_aliases()`: Resolves alias names to canonical parameters
- `forward_to_callable()`: Filters options based on target function signature
- `process_options()`: Full pipeline (aliases + hooks + forwarding)

#### 2. Analysis Module Updates

All 8 analysis modules updated with:
- Alias mappings defined as class attributes (`_ALIASES`)
- New parameters for enhanced functionality
- Support for `strict` mode parameter
- Updated docstrings documenting aliases

**Per-Module Changes:**

**RMSD:**
- Aliases: `ref`, `reference` → `reference_frame`; `atoms`, `selection`, `atom_indices` → `atoms`
- No new parameters (existing functionality exposed through aliases)

**RMSF:**
- Aliases: `atoms`, `selection`, `atom_indices` → `atoms`
- New: `per_residue` parameter for residue-level aggregation (post-hook)

**RG:**
- Aliases: `atoms`, `selection`, `atom_indices` → `atoms`
- New: `mass_weighted` parameter (MDTraj default is already mass-weighted)

**HBonds:**
- Aliases: `distance_cutoff_nm` → `distance`; `angle_cutoff_deg` → `angle`
- New: `distance`, `angle`, `periodic`, `sidechain_only`, `exclude_water` parameters
- Enhanced: Automatic parameter passing to MDTraj's baker_hubbard function

**SS:**
- Aliases: `atoms`, `selection`, `atom_indices` → `atoms`
- New: `algorithm`, `mkdssp_path` parameters

**SASA:**
- Aliases: `probe_radius_nm` → `probe_radius`; `atoms`, `selection`, `atom_indices` → `atoms`
- New: `n_sphere_points` parameter for calculation precision

**DimRed:**
- Aliases: `method` → `methods`; `perplexity` → `tsne_perplexity`; `n_iter` → `max_iter`; `dissimilarity` → `metric`
- New: `n_components`, `metric` (MDS), explicit parameter passing
- Enhanced: All sklearn parameters properly forwarded

**Cluster:**
- Aliases: `method` → `methods`; `linkage` → `linkage_method`
- New: `random_state`, `n_init`, `linkage_method` parameters
- Enhanced: Full sklearn parameter support for all clustering methods

#### 3. Orchestrator Integration

**analyze() function** (`src/fastmdanalysis/analysis/analyze.py`):
- Added `strict` parameter
- Automatic propagation of strict flag to all analyses
- Updated to inject strict mode into options dictionary

**CLI** (`src/fastmdanalysis/cli/analyze.py`):
- Added `--strict` flag
- Passes strict mode through to analyze() function

### Testing

#### Test Coverage

**test_options_forwarding.py** (399 lines, 18 tests):
- Alias mapping (simple, mixed, edge cases)
- Callable forwarding with signature introspection
- Pre-hook processing (with and without extra data)
- Strict vs non-strict error handling
- Full pipeline integration
- Realistic usage scenarios (RMSD, Cluster)

**test_analysis_options.py** (464 lines, 25+ tests):
- Each analysis module with aliases
- New parameter functionality
- Per-residue aggregation (RMSF)
- Distance/angle cutoffs (HBonds)
- Method selection (DimRed, Cluster)
- Strict mode behavior
- YAML-style option handling

**test_analyze_strict_mode.py** (211 lines, 12 tests):
- Strict mode propagation through orchestrator
- Multiple analyses with different option styles
- Empty options handling
- All analysis types with specific options

**Demo Scripts:**
- `test_options_standalone.py`: Standalone verification (no dependencies)
- `test_options_demo.py`: Full integration demo (requires mdtraj)

All tests pass ✅

### Documentation

#### Created Documentation

**OPTIONS_PASSTHROUGH.md** (360 lines):
- Complete feature overview
- Quick start guide (API and CLI)
- Alias tables for all 8 analyses
- Detailed examples for each analysis
- Strict mode usage guide
- Advanced features (hooks)
- Backward compatibility notes

**options_example.yaml** (91 lines):
- Comprehensive example configuration
- All 8 analyses configured
- Comments explaining each parameter
- Real-world usage patterns
- Both alias and canonical names shown

**Updated README.md:**
- Added feature to highlights
- Updated options file examples
- Added reference to detailed documentation

## Benefits

### For Users

1. **Familiar API**: Use MDTraj and scikit-learn parameter names directly
2. **Reduced Learning Curve**: No need to learn FastMDAnalysis-specific names
3. **Improved Reproducibility**: Copy parameters from MDTraj/sklearn scripts directly
4. **Flexible Configuration**: YAML files with intuitive parameter names
5. **Better Error Messages**: Strict mode catches parameter typos early

### For Developers

1. **Extensible Framework**: Easy to add new aliases and hooks
2. **Maintainable**: Centralized options handling logic
3. **Testable**: Comprehensive test coverage
4. **Documented**: Clear documentation of all mappings

## Usage Examples

### Quick Examples

```python
# RMSD with aliases
rmsd = fmda.rmsd(ref=0, selection="protein and name CA", align=True)

# RMSF with per-residue aggregation
rmsf = fmda.rmsf(atoms="protein", per_residue=True)

# Cluster with sklearn parameters
cluster = fmda.cluster(method="kmeans", n_clusters=5, random_state=42, n_init="auto")

# DimRed with familiar names
dimred = fmda.dimred(method="tsne", n_components=2, perplexity=30, max_iter=1000)
```

### YAML Configuration

```yaml
rmsd:
  ref: 0
  atoms: "protein and name CA"
  align: true

rmsf:
  selection: "protein"
  per_residue: true

cluster:
  method: kmeans
  n_clusters: 6
  random_state: 42
  n_init: "auto"
```

### CLI Usage

```bash
# Use YAML options file
fastmda analyze -traj traj.dcd -top top.pdb --options options.yaml

# Enable strict mode
fastmda analyze -traj traj.dcd -top top.pdb --options options.yaml --strict
```

## Technical Details

### Architecture

```
User Input (aliases, new params)
        ↓
OptionsForwarder.apply_aliases()
        ↓
OptionsForwarder.pre_hooks (optional)
        ↓
OptionsForwarder.forward_to_callable()
        ↓
Analysis Module Constructor
        ↓
Analysis.run() [post-hooks applied here]
```

### Alias Resolution Logic

1. User provides options (mix of aliases and canonical names)
2. OptionsForwarder resolves aliases to canonical names
3. If both alias and canonical provided:
   - Non-strict mode: log warning, use canonical
   - Strict mode: raise ValueError
4. Forward only accepted parameters to target function
5. Unknown parameters:
   - Non-strict mode: log info message
   - Strict mode: raise ValueError

### Hook System

**Pre-hooks:**
- Execute before forwarding to analysis
- Can transform parameter values
- Can return extra data for later use
- Example: Convert `align: true` to superposition call

**Post-hooks:**
- Execute after analysis completes
- Can aggregate or transform results
- Example: Per-residue aggregation in RMSF

## Statistics

- **Files Changed**: 18 files
- **Lines Added**: 2,322
- **Lines Removed**: 34
- **Test Files**: 3 (1,074 lines total)
- **Documentation**: 3 files (545 lines total)
- **Core Implementation**: 257 lines (options.py)
- **Test Coverage**: 30+ tests, all passing ✅

## Backward Compatibility

All existing code continues to work unchanged. The new features are purely additive:

```python
# Old style (still works)
rmsd = fmda.rmsd(reference_frame=0, atoms="protein")

# New style with aliases (also works)
rmsd = fmda.rmsd(ref=0, selection="protein")

# Mixed style (works too)
rmsd = fmda.rmsd(ref=0, atoms="protein")
```

## Future Enhancements

Potential future improvements:
1. Additional pre/post hooks for other analyses
2. More comprehensive parameter validation
3. Auto-completion support for aliases in IDEs
4. Integration with configuration management tools
5. Extended documentation with more examples

## Conclusion

The permissive options passthrough feature has been successfully implemented with:
- ✅ Full functionality as specified
- ✅ Comprehensive test coverage
- ✅ Complete documentation
- ✅ Backward compatibility
- ✅ Production-ready code

The feature enhances FastMDAnalysis usability while maintaining code quality and test coverage standards.
