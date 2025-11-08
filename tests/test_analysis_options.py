# tests/test_analysis_options.py
"""
Tests for analysis modules using the new permissive options passthrough.
"""
from __future__ import annotations

import pytest
import numpy as np

from fastmdanalysis.analysis import (
    RMSDAnalysis,
    RMSFAnalysis,
    RGAnalysis,
    HBondsAnalysis,
    SASAAnalysis,
    SSAnalysis,
    DimRedAnalysis,
    ClusterAnalysis,
)


# ------------------------------------------------------------------------------
# RMSD Analysis Tests
# ------------------------------------------------------------------------------

def test_rmsd_with_aliases(traj, outdir):
    """Test RMSD with aliased options."""
    # Using 'ref' instead of 'reference_frame'
    analysis = RMSDAnalysis(
        traj,
        ref=0,  # alias for reference_frame
        selection="protein and name CA",  # alias for atoms
        align=True,
        output=str(outdir / "rmsd"),
    )
    
    results = analysis.run()
    
    assert "rmsd" in results
    assert results["rmsd"].shape[0] == traj.n_frames
    assert analysis.reference_frame == 0
    assert analysis.atoms == "protein and name CA"


def test_rmsd_with_reference_alias(traj, outdir):
    """Test RMSD with 'reference' alias."""
    analysis = RMSDAnalysis(
        traj,
        reference=5,  # another alias for reference_frame
        atoms="protein",
        output=str(outdir / "rmsd"),
    )
    
    assert analysis.reference_frame == 5


def test_rmsd_strict_mode_unknown_option(traj, outdir):
    """Test RMSD strict mode with unknown option."""
    # In strict mode, this should not raise during init
    # because we filter unknown options in the forwarder
    analysis = RMSDAnalysis(
        traj,
        reference_frame=0,
        atoms="protein",
        unknown_param="value",  # unknown option
        strict=False,  # non-strict should log warning
        output=str(outdir / "rmsd"),
    )
    
    # Should still work
    results = analysis.run()
    assert "rmsd" in results


# ------------------------------------------------------------------------------
# RMSF Analysis Tests
# ------------------------------------------------------------------------------

def test_rmsf_per_residue_aggregation(traj, outdir):
    """Test RMSF with per_residue aggregation."""
    analysis = RMSFAnalysis(
        traj,
        atoms="protein and name CA",
        per_residue=True,
        output=str(outdir / "rmsf"),
    )
    
    results = analysis.run()
    
    assert "rmsf" in results
    assert "rmsf_per_residue" in results
    
    # Per-residue should have fewer entries than per-atom
    n_residues = results["rmsf_per_residue"].shape[0]
    n_atoms = results["rmsf"].shape[0]
    assert n_residues <= n_atoms


def test_rmsf_with_aliases(traj, outdir):
    """Test RMSF with aliased options."""
    analysis = RMSFAnalysis(
        traj,
        selection="protein",  # alias for atoms
        output=str(outdir / "rmsf"),
    )
    
    results = analysis.run()
    assert "rmsf" in results
    assert analysis.atoms == "protein"


# ------------------------------------------------------------------------------
# RG Analysis Tests
# ------------------------------------------------------------------------------

def test_rg_mass_weighted_option(traj, outdir):
    """Test RG with mass_weighted option."""
    analysis = RGAnalysis(
        traj,
        atoms="protein",
        mass_weighted=True,  # Note: MDTraj's compute_rg is already mass-weighted
        output=str(outdir / "rg"),
    )
    
    results = analysis.run()
    
    assert "rg" in results
    assert analysis.mass_weighted is True


def test_rg_with_aliases(traj, outdir):
    """Test RG with aliased options."""
    analysis = RGAnalysis(
        traj,
        selection="protein",  # alias for atoms
        output=str(outdir / "rg"),
    )
    
    assert analysis.atoms == "protein"


# ------------------------------------------------------------------------------
# HBonds Analysis Tests
# ------------------------------------------------------------------------------

def test_hbonds_with_cutoff_options(traj, outdir):
    """Test HBonds with distance and angle cutoffs."""
    analysis = HBondsAnalysis(
        traj,
        atoms="protein",
        distance_cutoff_nm=0.30,  # alias for distance
        angle_cutoff_deg=110,     # alias for angle
        periodic=False,
        output=str(outdir / "hbonds"),
    )
    
    assert analysis.distance == 0.30
    assert analysis.angle == 110.0
    assert analysis.periodic is False


def test_hbonds_exclude_water(traj, outdir):
    """Test HBonds with exclude_water option."""
    analysis = HBondsAnalysis(
        traj,
        exclude_water=True,
        output=str(outdir / "hbonds"),
    )
    
    assert analysis.exclude_water is True


def test_hbonds_sidechain_only(traj, outdir):
    """Test HBonds with sidechain_only option."""
    analysis = HBondsAnalysis(
        traj,
        sidechain_only=True,
        output=str(outdir / "hbonds"),
    )
    
    assert analysis.sidechain_only is True


# ------------------------------------------------------------------------------
# SASA Analysis Tests
# ------------------------------------------------------------------------------

def test_sasa_with_probe_radius_alias(traj, outdir):
    """Test SASA with probe_radius_nm alias."""
    analysis = SASAAnalysis(
        traj,
        probe_radius_nm=0.16,  # alias for probe_radius
        atoms="protein",
        output=str(outdir / "sasa"),
    )
    
    assert analysis.probe_radius == 0.16


def test_sasa_with_n_sphere_points(traj, outdir):
    """Test SASA with n_sphere_points option."""
    analysis = SASAAnalysis(
        traj,
        probe_radius=0.14,
        n_sphere_points=480,
        atoms="protein",
        output=str(outdir / "sasa"),
    )
    
    assert analysis.n_sphere_points == 480


# ------------------------------------------------------------------------------
# SS Analysis Tests
# ------------------------------------------------------------------------------

def test_ss_with_algorithm(traj, outdir):
    """Test SS with algorithm option."""
    analysis = SSAnalysis(
        traj,
        atoms="protein",
        algorithm="dssp",
        output=str(outdir / "ss"),
    )
    
    assert analysis.algorithm == "dssp"


def test_ss_with_mkdssp_path(traj, outdir):
    """Test SS with mkdssp_path option."""
    analysis = SSAnalysis(
        traj,
        atoms="protein",
        mkdssp_path="/usr/bin/mkdssp",
        output=str(outdir / "ss"),
    )
    
    assert analysis.mkdssp_path == "/usr/bin/mkdssp"


# ------------------------------------------------------------------------------
# DimRed Analysis Tests
# ------------------------------------------------------------------------------

def test_dimred_method_alias(traj, outdir):
    """Test DimRed with 'method' (singular) alias."""
    analysis = DimRedAnalysis(
        traj,
        method="pca",  # alias for methods
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert "pca" in analysis.methods


def test_dimred_n_components(traj, outdir):
    """Test DimRed with n_components option."""
    analysis = DimRedAnalysis(
        traj,
        methods="pca",
        n_components=3,
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert analysis.n_components == 3


def test_dimred_perplexity_alias(traj, outdir):
    """Test DimRed with perplexity option (alias for tsne_perplexity)."""
    analysis = DimRedAnalysis(
        traj,
        methods="tsne",
        perplexity=20,
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert analysis.tsne_perplexity == 20


def test_dimred_max_iter(traj, outdir):
    """Test DimRed with max_iter option."""
    analysis = DimRedAnalysis(
        traj,
        methods="tsne",
        max_iter=1000,
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert analysis.tsne_max_iter == 1000


def test_dimred_n_iter_deprecated_alias(traj, outdir):
    """Test DimRed with deprecated n_iter alias."""
    analysis = DimRedAnalysis(
        traj,
        methods="tsne",
        n_iter=800,  # deprecated, should map to max_iter
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert analysis.tsne_max_iter == 800


def test_dimred_mds_metric(traj, outdir):
    """Test DimRed with MDS metric option."""
    analysis = DimRedAnalysis(
        traj,
        methods="mds",
        metric="euclidean",
        atoms="protein and name CA",
        outdir=str(outdir / "dimred"),
    )
    
    assert analysis.mds_metric == "euclidean"


# ------------------------------------------------------------------------------
# Cluster Analysis Tests
# ------------------------------------------------------------------------------

def test_cluster_method_alias(traj, outdir):
    """Test Cluster with 'method' (singular) alias."""
    analysis = ClusterAnalysis(
        traj,
        method="kmeans",  # alias for methods
        n_clusters=3,
        atoms="protein and name CA",
        output=str(outdir / "cluster"),
    )
    
    assert "kmeans" in analysis.methods


def test_cluster_random_state(traj, outdir):
    """Test Cluster with random_state option."""
    analysis = ClusterAnalysis(
        traj,
        methods="kmeans",
        n_clusters=3,
        random_state=123,
        atoms="protein and name CA",
        output=str(outdir / "cluster"),
    )
    
    assert analysis.random_state == 123


def test_cluster_n_init(traj, outdir):
    """Test Cluster with n_init option."""
    analysis = ClusterAnalysis(
        traj,
        methods="kmeans",
        n_clusters=3,
        n_init="auto",
        atoms="protein and name CA",
        output=str(outdir / "cluster"),
    )
    
    assert analysis.n_init == "auto"


def test_cluster_linkage_alias(traj, outdir):
    """Test Cluster with linkage alias for linkage_method."""
    analysis = ClusterAnalysis(
        traj,
        methods="hierarchical",
        n_clusters=3,
        linkage="average",  # alias for linkage_method
        atoms="protein and name CA",
        output=str(outdir / "cluster"),
    )
    
    assert analysis.linkage_method == "average"


def test_cluster_dbscan_params(traj, outdir):
    """Test Cluster DBSCAN with eps and min_samples."""
    analysis = ClusterAnalysis(
        traj,
        methods="dbscan",
        eps=0.3,
        min_samples=10,
        atoms="protein and name CA",
        output=str(outdir / "cluster"),
    )
    
    assert analysis.eps == 0.3
    assert analysis.min_samples == 10


# ------------------------------------------------------------------------------
# Strict Mode Tests
# ------------------------------------------------------------------------------

def test_strict_mode_enabled(traj, outdir):
    """Test that strict mode can be enabled."""
    # This should work - strict mode is just a flag
    analysis = RMSDAnalysis(
        traj,
        reference_frame=0,
        atoms="protein",
        strict=True,
        output=str(outdir / "rmsd"),
    )
    
    assert analysis.strict is True


def test_strict_mode_duplicate_alias():
    """Test strict mode with duplicate alias and canonical."""
    from fastmdanalysis.analysis.rmsd import RMSDAnalysis
    
    # This should raise an error due to both alias and canonical provided
    with pytest.raises(ValueError, match="Both alias"):
        analysis = RMSDAnalysis(
            None,  # trajectory not needed for init test
            ref=0,  # alias
            reference_frame=1,  # canonical
            strict=True,
        )


# ------------------------------------------------------------------------------
# Integration Test: Options from YAML
# ------------------------------------------------------------------------------

def test_yaml_style_options(traj, outdir):
    """Test that YAML-style options work as expected."""
    # Simulate options that would come from a YAML file
    rmsd_options = {
        "ref": 0,
        "align": True,
        "atoms": "protein and name CA",
    }
    
    analysis = RMSDAnalysis(traj, output=str(outdir / "rmsd"), **rmsd_options)
    
    assert analysis.reference_frame == 0
    assert analysis.align is True
    assert analysis.atoms == "protein and name CA"
    
    results = analysis.run()
    assert "rmsd" in results


def test_mixed_aliases_and_canonical(traj, outdir):
    """Test using a mix of aliases and canonical names."""
    # Using both alias and canonical for different params
    analysis = RMSDAnalysis(
        traj,
        ref=0,  # alias
        atoms="protein",  # canonical
        align=True,  # canonical
        output=str(outdir / "rmsd"),
    )
    
    assert analysis.reference_frame == 0
    assert analysis.atoms == "protein"
    assert analysis.align is True

def test_fastmda_analyze_forwards_options(fastmda, tmp_path):
    """Ensure FastMDAnalysis.analyze forwards permissive options to analyses."""
    opts = {"rmsd": {"align": False}}
    results = fastmda.analyze(include=["rmsd"], options=opts, output=tmp_path / "forward_out")

    rmsd_result = results["rmsd"]
    assert rmsd_result.ok
    assert rmsd_result.value.align is False


def test_fastmda_analyze_strict_mode_raises_on_unknown(fastmda):
    """Strict mode should raise when unknown options are supplied."""
    opts = {"rmsd": {"bogus_option": 123}}
    with pytest.raises(ValueError, match="Unknown options"):
        fastmda.analyze(include=["rmsd"], options=opts, strict=True)
