# tests/test_analyze_strict_mode.py
"""
Tests for the analyze orchestrator with strict mode.
"""
from __future__ import annotations

import pytest


def test_analyze_with_strict_mode_false(fastmda, tmp_path):
    """Test analyze orchestrator with strict=False (default)."""
    options = {
        "rmsd": {
            "ref": 0,
            "atoms": "protein and name CA",
        },
        "rmsf": {
            "atoms": "protein and name CA",
        },
    }
    
    results = fastmda.analyze(
        include=["rmsd", "rmsf"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,  # non-strict mode
    )
    
    assert "rmsd" in results
    assert "rmsf" in results
    assert results["rmsd"].ok
    assert results["rmsf"].ok


def test_analyze_with_strict_mode_true(fastmda, tmp_path):
    """Test analyze orchestrator with strict=True."""
    options = {
        "rmsd": {
            "reference_frame": 0,
            "atoms": "protein and name CA",
        },
        "rmsf": {
            "atoms": "protein and name CA",
        },
    }
    
    results = fastmda.analyze(
        include=["rmsd", "rmsf"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=True,  # strict mode
    )
    
    assert "rmsd" in results
    assert "rmsf" in results
    assert results["rmsd"].ok
    assert results["rmsf"].ok


def test_analyze_strict_mode_propagates_to_analyses(fastmda, tmp_path):
    """Test that strict mode flag propagates to individual analyses."""
    options = {
        "rmsd": {
            "ref": 0,  # using alias
            "atoms": "protein",
        },
    }
    
    # With strict=True, aliases should still work
    # (strict mode is about unknown options, not aliases)
    results = fastmda.analyze(
        include=["rmsd"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=True,
    )
    
    assert results["rmsd"].ok


def test_analyze_with_aliased_options(fastmda, tmp_path):
    """Test analyze with various aliased options."""
    options = {
        "rmsd": {
            "ref": 0,
            "selection": "protein and name CA",
            "align": True,
        },
        "rmsf": {
            "selection": "protein and name CA",
            "per_residue": True,
        },
        "rg": {
            "selection": "protein",
            "mass_weighted": False,
        },
    }
    
    results = fastmda.analyze(
        include=["rmsd", "rmsf", "rg"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,
    )
    
    assert all(results[name].ok for name in ["rmsd", "rmsf", "rg"])


def test_analyze_with_cluster_options(fastmda, tmp_path):
    """Test analyze with cluster-specific options."""
    options = {
        "cluster": {
            "method": "kmeans",
            "n_clusters": 3,
            "random_state": 42,
            "n_init": "auto",
        },
    }
    
    results = fastmda.analyze(
        include=["cluster"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,
    )
    
    assert results["cluster"].ok


def test_analyze_with_dimred_options(fastmda, tmp_path):
    """Test analyze with dimred-specific options."""
    options = {
        "dimred": {
            "method": "pca",
            "n_components": 2,
            "random_state": 42,
        },
    }
    
    results = fastmda.analyze(
        include=["dimred"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,
    )
    
    assert results["dimred"].ok


def test_analyze_with_hbonds_options(fastmda, tmp_path):
    """Test analyze with hbonds-specific options."""
    options = {
        "hbonds": {
            "distance_cutoff_nm": 0.25,
            "angle_cutoff_deg": 120,
            "periodic": False,
            "exclude_water": False,
        },
    }
    
    results = fastmda.analyze(
        include=["hbonds"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,
    )
    
    assert results["hbonds"].ok


def test_analyze_with_sasa_options(fastmda, tmp_path):
    """Test analyze with sasa-specific options."""
    options = {
        "sasa": {
            "probe_radius_nm": 0.14,
            "n_sphere_points": 960,
            "selection": "protein",
        },
    }
    
    results = fastmda.analyze(
        include=["sasa"],
        options=options,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=False,
    )
    
    assert results["sasa"].ok


def test_analyze_empty_options_strict_mode(fastmda, tmp_path):
    """Test analyze with no options in strict mode."""
    # Should work fine - no unknown options
    results = fastmda.analyze(
        include=["rmsd", "rmsf"],
        options=None,
        verbose=False,
        output=str(tmp_path / "analyze"),
        strict=True,
    )
    
    assert results["rmsd"].ok
    assert results["rmsf"].ok
