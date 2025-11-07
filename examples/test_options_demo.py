#!/usr/bin/env python
"""
Demo script showing the new permissive options passthrough feature.
This demonstrates alias mapping and the new parameters.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "../src")

# Test the options forwarding utility
from fastmdanalysis.utils.options import OptionsForwarder, forward_options, apply_alias_mapping

print("=" * 70)
print("Testing Options Forwarding Utility")
print("=" * 70)

# Test 1: Simple alias mapping
print("\n1. Simple Alias Mapping:")
aliases = {"ref": "reference_frame", "atoms": "atom_indices"}
options = {"ref": 0, "atoms": "protein"}
result = apply_alias_mapping(options, aliases)
print(f"   Input:  {options}")
print(f"   Output: {result}")
assert result == {"reference_frame": 0, "atom_indices": "protein"}
print("   ✓ PASS")

# Test 2: OptionsForwarder with callable
print("\n2. Forwarding to Callable:")
def my_function(reference_frame: int = 0, atoms: str = None, align: bool = True):
    return reference_frame, atoms, align

forwarder = OptionsForwarder(
    aliases={"ref": "reference_frame", "selection": "atoms"}
)
options = {"ref": 5, "selection": "protein", "align": False, "unknown": "value"}
forwarded, hook_data = forwarder.process_options(options, callable_obj=my_function)
print(f"   Input:     {options}")
print(f"   Forwarded: {forwarded}")
assert forwarded == {"reference_frame": 5, "atoms": "protein", "align": False}
print("   ✓ PASS")

# Test 3: Pre-hook processing
print("\n3. Pre-hook Processing:")
def double_hook(value, context):
    return value * 2

forwarder_with_hook = OptionsForwarder(
    pre_hooks={"multiplier": double_hook}
)
options = {"multiplier": 5, "other": "value"}
processed, hook_data = forwarder_with_hook.process_options(options)
print(f"   Input:     {options}")
print(f"   Processed: {processed}")
assert processed["multiplier"] == 10
print("   ✓ PASS")

# Test 4: Strict mode
print("\n4. Strict Mode:")
try:
    forwarder_strict = OptionsForwarder(
        aliases={"ref": "reference_frame"},
        strict=True
    )
    options_dup = {"ref": 0, "reference_frame": 1}
    forwarder_strict.apply_aliases(options_dup)
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   Caught expected error: {e}")
    print("   ✓ PASS")

print("\n" + "=" * 70)
print("Testing Analysis Module Initialization")
print("=" * 70)

# Test analysis modules (without running, just init)
from fastmdanalysis.analysis import (
    RMSDAnalysis,
    RMSFAnalysis,
    RGAnalysis,
)

# Mock trajectory for testing
class MockTraj:
    n_frames = 10
    n_atoms = 100
    class Topology:
        def select(self, selection):
            return [0, 1, 2, 3, 4]
    topology = Topology()

traj = MockTraj()

# Test 5: RMSD with aliases
print("\n5. RMSD Analysis with Aliases:")
rmsd = RMSDAnalysis(
    traj,
    ref=5,  # alias for reference_frame
    selection="protein",  # alias for atoms
    align=True,
    output="/tmp/rmsd_test"
)
print(f"   reference_frame: {rmsd.reference_frame}")
print(f"   atoms: {rmsd.atoms}")
print(f"   align: {rmsd.align}")
assert rmsd.reference_frame == 5
assert rmsd.atoms == "protein"
assert rmsd.align is True
print("   ✓ PASS")

# Test 6: RMSF with per_residue
print("\n6. RMSF Analysis with per_residue:")
rmsf = RMSFAnalysis(
    traj,
    atom_indices="protein and name CA",  # alias for atoms
    per_residue=True,
    output="/tmp/rmsf_test"
)
print(f"   atoms: {rmsf.atoms}")
print(f"   per_residue: {rmsf.per_residue}")
assert rmsf.atoms == "protein and name CA"
assert rmsf.per_residue is True
print("   ✓ PASS")

# Test 7: RG with mass_weighted
print("\n7. RG Analysis with mass_weighted:")
rg = RGAnalysis(
    traj,
    selection="protein",  # alias for atoms
    mass_weighted=False,
    output="/tmp/rg_test"
)
print(f"   atoms: {rg.atoms}")
print(f"   mass_weighted: {rg.mass_weighted}")
assert rg.atoms == "protein"
assert rg.mass_weighted is False
print("   ✓ PASS")

print("\n" + "=" * 70)
print("All Tests Passed! ✓")
print("=" * 70)
print("\nThe permissive options passthrough feature is working correctly.")
print("Aliases are properly mapped and new parameters are recognized.")
