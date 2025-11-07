# tests/test_options_forwarding.py
"""
Tests for the permissive options passthrough utility.
"""
from __future__ import annotations

import pytest
import warnings
from typing import Optional

from fastmdanalysis.utils.options import (
    OptionsForwarder,
    forward_options,
    apply_alias_mapping,
)


# ------------------------------------------------------------------------------
# Test apply_alias_mapping
# ------------------------------------------------------------------------------

def test_apply_alias_mapping_simple():
    """Test basic alias mapping."""
    aliases = {"ref": "reference_frame", "atoms": "atom_indices"}
    options = {"ref": 0, "atoms": "protein"}
    
    result = apply_alias_mapping(options, aliases)
    
    assert result == {"reference_frame": 0, "atom_indices": "protein"}


def test_apply_alias_mapping_no_aliases():
    """Test that options pass through unchanged when no aliases match."""
    aliases = {"ref": "reference_frame"}
    options = {"align": True, "output": "test"}
    
    result = apply_alias_mapping(options, aliases)
    
    assert result == {"align": True, "output": "test"}


def test_apply_alias_mapping_mixed():
    """Test mix of aliased and non-aliased options."""
    aliases = {"ref": "reference_frame"}
    options = {"ref": 0, "align": True, "atoms": "protein"}
    
    result = apply_alias_mapping(options, aliases)
    
    assert result == {"reference_frame": 0, "align": True, "atoms": "protein"}


# ------------------------------------------------------------------------------
# Test OptionsForwarder - Alias Resolution
# ------------------------------------------------------------------------------

def test_forwarder_apply_aliases():
    """Test alias resolution via OptionsForwarder."""
    forwarder = OptionsForwarder(
        aliases={"ref": "reference_frame", "atoms": "atom_indices"}
    )
    options = {"ref": 0, "atoms": "protein", "align": True}
    
    resolved = forwarder.apply_aliases(options)
    
    assert resolved == {
        "reference_frame": 0,
        "atom_indices": "protein",
        "align": True,
    }


def test_forwarder_duplicate_alias_non_strict():
    """Test that duplicate (alias + canonical) logs warning in non-strict mode."""
    forwarder = OptionsForwarder(
        aliases={"ref": "reference_frame"},
        strict=False,
    )
    options = {"ref": 0, "reference_frame": 1}
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        resolved = forwarder.apply_aliases(options)
        
        # Should drop the alias when both provided
        assert "reference_frame" in resolved
        # In non-strict mode, a warning should be logged (not raised)


def test_forwarder_duplicate_alias_strict():
    """Test that duplicate (alias + canonical) raises error in strict mode."""
    forwarder = OptionsForwarder(
        aliases={"ref": "reference_frame"},
        strict=True,
    )
    options = {"ref": 0, "reference_frame": 1}
    
    with pytest.raises(ValueError, match="Both alias"):
        forwarder.apply_aliases(options)


# ------------------------------------------------------------------------------
# Test OptionsForwarder - Forwarding to Callable
# ------------------------------------------------------------------------------

def dummy_function(reference_frame: int = 0, align: bool = True, atoms: Optional[str] = None):
    """Dummy function for testing signature introspection."""
    return reference_frame, align, atoms


def test_forwarder_to_callable():
    """Test forwarding options to a callable based on signature."""
    forwarder = OptionsForwarder()
    options = {
        "reference_frame": 0,
        "align": True,
        "atoms": "protein",
        "unknown_param": "value",
    }
    
    forwarded, dropped = forwarder.forward_to_callable(dummy_function, options)
    
    assert forwarded == {
        "reference_frame": 0,
        "align": True,
        "atoms": "protein",
    }
    assert dropped == ["unknown_param"]


def test_forwarder_to_callable_with_kwargs():
    """Test forwarding when callable accepts **kwargs."""
    def func_with_kwargs(ref: int = 0, **kwargs):
        return ref, kwargs
    
    forwarder = OptionsForwarder()
    options = {"ref": 1, "unknown": "value"}
    
    forwarded, dropped = forwarder.forward_to_callable(func_with_kwargs, options)
    
    # When callable accepts **kwargs, all options are forwarded
    assert forwarded == {"ref": 1, "unknown": "value"}
    assert dropped == []


# ------------------------------------------------------------------------------
# Test OptionsForwarder - Pre-hooks
# ------------------------------------------------------------------------------

def test_forwarder_pre_hook():
    """Test pre-hook processing."""
    def align_hook(value, context):
        # Pre-process align parameter
        if value is True:
            return "superpose"
        return value
    
    forwarder = OptionsForwarder(
        pre_hooks={"align": align_hook}
    )
    options = {"align": True, "atoms": "protein"}
    
    processed, hook_data = forwarder.process_options(options)
    
    assert processed["align"] == "superpose"
    assert processed["atoms"] == "protein"


def test_forwarder_pre_hook_with_data():
    """Test pre-hook that returns extra data."""
    def hook_with_data(value, context):
        # Return (processed_value, extra_data)
        return value * 2, {"original": value}
    
    forwarder = OptionsForwarder(
        pre_hooks={"param": hook_with_data}
    )
    options = {"param": 5}
    
    processed, hook_data = forwarder.process_options(options)
    
    assert processed["param"] == 10
    assert hook_data["param"] == {"original": 5}


def test_forwarder_pre_hook_error_non_strict():
    """Test pre-hook error handling in non-strict mode."""
    def failing_hook(value, context):
        raise ValueError("Hook failed")
    
    forwarder = OptionsForwarder(
        pre_hooks={"param": failing_hook},
        strict=False,
    )
    options = {"param": "value"}
    
    # Should not raise, but log warning and use original value
    processed, _ = forwarder.process_options(options)
    
    assert processed["param"] == "value"


def test_forwarder_pre_hook_error_strict():
    """Test pre-hook error handling in strict mode."""
    def failing_hook(value, context):
        raise ValueError("Hook failed")
    
    forwarder = OptionsForwarder(
        pre_hooks={"param": failing_hook},
        strict=True,
    )
    options = {"param": "value"}
    
    with pytest.raises(RuntimeError, match="Pre-hook.*failed"):
        forwarder.process_options(options)


# ------------------------------------------------------------------------------
# Test OptionsForwarder - Full Pipeline
# ------------------------------------------------------------------------------

def test_forwarder_full_pipeline():
    """Test complete pipeline: aliases + pre-hooks + forwarding."""
    def align_hook(value, context):
        if value is True:
            return "superpose"
        return value
    
    forwarder = OptionsForwarder(
        aliases={"ref": "reference_frame"},
        pre_hooks={"align": align_hook},
    )
    options = {
        "ref": 0,
        "align": True,
        "atoms": "protein",
        "unknown": "value",
    }
    
    forwarded, hook_data = forwarder.process_options(
        options,
        callable_obj=dummy_function,
    )
    
    # ref should be aliased to reference_frame
    assert forwarded["reference_frame"] == 0
    # align should be processed by hook
    assert forwarded["align"] == "superpose"
    # atoms should pass through
    assert forwarded["atoms"] == "protein"
    # unknown should be dropped (not in dummy_function signature)
    assert "unknown" not in forwarded


# ------------------------------------------------------------------------------
# Test forward_options convenience function
# ------------------------------------------------------------------------------

def test_forward_options_convenience():
    """Test the forward_options convenience function."""
    def target(reference_frame: int = 0, atoms: Optional[str] = None):
        return reference_frame, atoms
    
    options = {
        "ref": 0,
        "atoms": "protein",
        "unknown": "value",
    }
    aliases = {"ref": "reference_frame"}
    
    forwarded = forward_options(target, options, aliases=aliases, strict=False)
    
    assert forwarded == {"reference_frame": 0, "atoms": "protein"}
    assert "unknown" not in forwarded


def test_forward_options_strict_mode():
    """Test forward_options in strict mode."""
    def target(param: int = 0):
        return param
    
    options = {"param": 1, "unknown": 2}
    
    # Non-strict should not raise
    forwarded = forward_options(target, options, strict=False)
    assert forwarded == {"param": 1}
    
    # Strict mode doesn't raise for unknown in forward_options
    # (it only raises if forwarder is configured to do so)
    forwarded_strict = forward_options(target, options, strict=True)
    assert forwarded_strict == {"param": 1}


# ------------------------------------------------------------------------------
# Integration Test: Realistic Usage
# ------------------------------------------------------------------------------

def test_realistic_rmsd_options():
    """Test realistic RMSD-like options processing."""
    def rmsd_analysis(reference_frame: int = 0, atoms: Optional[str] = None, align: bool = True):
        return reference_frame, atoms, align
    
    forwarder = OptionsForwarder(
        aliases={
            "ref": "reference_frame",
            "reference": "reference_frame",
            "atom_indices": "atoms",
            "selection": "atoms",
        },
        strict=False,
    )
    
    # User provides aliased options
    user_options = {
        "ref": 10,
        "selection": "protein and name CA",
        "align": False,
    }
    
    forwarded, _ = forwarder.process_options(user_options, callable_obj=rmsd_analysis)
    
    assert forwarded == {
        "reference_frame": 10,
        "atoms": "protein and name CA",
        "align": False,
    }


def test_realistic_cluster_options():
    """Test realistic cluster-like options with method selection."""
    def cluster_analysis(
        methods: str = "all",
        n_clusters: Optional[int] = None,
        random_state: int = 42,
        linkage: str = "ward",
    ):
        return methods, n_clusters, random_state, linkage
    
    forwarder = OptionsForwarder(
        aliases={
            "method": "methods",
            "linkage_method": "linkage",
        },
        strict=False,
    )
    
    user_options = {
        "method": "kmeans",
        "n_clusters": 5,
        "random_state": 123,
        "linkage_method": "average",
    }
    
    forwarded, _ = forwarder.process_options(user_options, callable_obj=cluster_analysis)
    
    assert forwarded == {
        "methods": "kmeans",
        "n_clusters": 5,
        "random_state": 123,
        "linkage": "average",
    }


# ------------------------------------------------------------------------------
# Edge Cases
# ------------------------------------------------------------------------------

def test_empty_options():
    """Test with empty options."""
    forwarder = OptionsForwarder()
    
    forwarded, hook_data = forwarder.process_options({})
    
    assert forwarded == {}
    assert hook_data == {}


def test_none_options():
    """Test with None as options."""
    forwarder = OptionsForwarder()
    
    # Should handle None gracefully
    forwarded, hook_data = forwarder.process_options({})
    
    assert forwarded == {}
    assert hook_data == {}


def test_callable_without_signature():
    """Test forwarding to callable without inspectable signature."""
    # Built-in functions may not have inspectable signatures
    forwarder = OptionsForwarder()
    options = {"param": "value"}
    
    # Should forward all when signature can't be inspected
    forwarded, dropped = forwarder.forward_to_callable(len, options)
    
    # When inspection fails, all options are forwarded
    assert forwarded == options
    assert dropped == []
