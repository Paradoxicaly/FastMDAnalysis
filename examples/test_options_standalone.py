#!/usr/bin/env python
"""
Standalone test of options forwarding utility (no dependencies).
This demonstrates the core functionality without requiring mdtraj.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import only the options module (no mdtraj needed)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "options",
    src_path / "fastmdanalysis" / "utils" / "options.py"
)
options_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(options_module)

OptionsForwarder = options_module.OptionsForwarder
forward_options = options_module.forward_options
apply_alias_mapping = options_module.apply_alias_mapping

print("=" * 70)
print("Testing Options Forwarding Utility (Standalone)")
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
    # Order matters: canonical first, then alias
    options_dup = {"reference_frame": 1, "ref": 0}
    forwarder_strict.apply_aliases(options_dup)
    print("   ✗ FAIL: Should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    print(f"   Caught expected error: {str(e)[:60]}...")
    print("   ✓ PASS")

# Test 5: Forwarding with unknown options (non-strict)
print("\n5. Unknown Options (Non-Strict):")
forwarder = OptionsForwarder(strict=False)
options = {"known": "value", "unknown1": "val1", "unknown2": "val2"}

def target_func(known: str = None):
    return known

forwarded, dropped = forwarder.forward_to_callable(target_func, options)
print(f"   Input:     {options}")
print(f"   Forwarded: {forwarded}")
print(f"   Dropped:   {dropped}")
assert forwarded == {"known": "value"}
assert set(dropped) == {"unknown1", "unknown2"}
print("   ✓ PASS")

# Test 6: Complex alias resolution
print("\n6. Complex Alias Resolution:")
aliases = {
    "ref": "reference_frame",
    "reference": "reference_frame",
    "atoms": "atom_indices",
    "selection": "atom_indices",
}
forwarder = OptionsForwarder(aliases=aliases)

# Test with multiple aliases for same param
options1 = {"ref": 0, "selection": "protein"}
resolved1 = forwarder.apply_aliases(options1)
print(f"   Input:    {options1}")
print(f"   Resolved: {resolved1}")
assert resolved1 == {"reference_frame": 0, "atom_indices": "protein"}
print("   ✓ PASS")

# Test 7: Pre-hook with data return
print("\n7. Pre-hook with Extra Data:")
def hook_with_data(value, context):
    # Return (processed_value, extra_data)
    return value.upper(), {"original": value}

forwarder_data = OptionsForwarder(
    pre_hooks={"name": hook_with_data}
)
options = {"name": "test"}
processed, hook_data = forwarder_data.process_options(options)
print(f"   Input:       {options}")
print(f"   Processed:   {processed}")
print(f"   Hook Data:   {hook_data}")
assert processed["name"] == "TEST"
assert hook_data["name"]["original"] == "test"
print("   ✓ PASS")

# Test 8: Full pipeline
print("\n8. Full Pipeline (Aliases + Hooks + Forwarding):")
def align_hook(value, context):
    if value is True:
        return "superpose"
    return value

def rmsd_func(reference_frame: int = 0, atoms: str = None, align: str = "align"):
    return reference_frame, atoms, align

forwarder_full = OptionsForwarder(
    aliases={"ref": "reference_frame", "selection": "atoms"},
    pre_hooks={"align": align_hook},
)
options = {
    "ref": 10,
    "selection": "protein and name CA",
    "align": True,
    "unknown": "ignored"
}
forwarded, hook_data = forwarder_full.process_options(options, callable_obj=rmsd_func)
print(f"   Input:     {options}")
print(f"   Forwarded: {forwarded}")
assert forwarded["reference_frame"] == 10
assert forwarded["atoms"] == "protein and name CA"
assert forwarded["align"] == "superpose"
assert "unknown" not in forwarded
print("   ✓ PASS")

print("\n" + "=" * 70)
print("All Tests Passed! ✓")
print("=" * 70)
print("\nThe options forwarding utility is working correctly:")
print("  • Alias mapping works")
print("  • Signature introspection works")
print("  • Pre-hooks work (with and without extra data)")
print("  • Strict mode works")
print("  • Unknown option handling works")
print("  • Full pipeline integration works")
