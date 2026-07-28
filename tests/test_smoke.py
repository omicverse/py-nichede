"""Smoke tests — no R, no reference dump required."""

from __future__ import annotations

import numpy as np
import pytest

import pynichede as nde


def test_import_surface():
    for name in nde.__all__:
        assert hasattr(nde, name), f"missing export {name}"
    assert nde.__version__ == "0.1.0"


def test_pipeline_runs_end_to_end(toy):
    counts, coord, lib, deconv = toy
    obj = nde.create_nichede_object(counts, coord, lib, deconv, sigma=[100, 200])
    assert obj.counts.shape == counts.shape
    assert obj.num_cells.shape == deconv.shape
    assert abs(obj.spot_distance - 100.0) < 1e-12

    obj = nde.calculate_effective_niche(obj, cutoff=0.05)
    assert len(obj.effective_niche) == 2
    for en in obj.effective_niche:
        assert en.shape == deconv.shape
        assert np.isfinite(en).all()

    obj = nde.niche_DE(obj, num_cores=1, C=50, M=5, verbose=False)
    assert len(obj.niche_DE) == 2
    assert "gene_level" in obj.niche_DE_pval_pos
    assert obj.niche_DE_pval_pos["cell_type_level"].shape == (counts.shape[1], 3)
    assert obj.niche_DE_pval_pos["interaction_level"].shape == (3, 3, counts.shape[1])


def test_class_api_chains(toy):
    counts, coord, lib, deconv = toy
    m = (nde.NicheDE.from_matrices(counts, coord, lib, deconv, sigma=[100])
         .effective_niche()
         .run(num_cores=1, C=50, M=5, verbose=False))
    assert "Niche-DE object with" in repr(m)
    df = m.genes("G", positive=True, alpha=1.0)
    assert list(df.columns) == ["Genes", "Pvalues.Gene"]
    ad = m.to_anndata()
    assert "nichede_effective_niche_100" in ad.obsm
    assert "nichede" in ad.uns


def test_large_scale_matches_exact(toy):
    """The tiled effective niche is an exact reformulation, not an approximation.

    The bounding box is padded by ``sigma * sqrt(-log(cutoff))``, the radius past
    which the Gaussian kernel is provably below ``cutoff`` and therefore
    truncated to zero, so no contribution can be missed.
    """
    counts, coord, lib, deconv = toy
    a = nde.calculate_effective_niche(
        nde.create_nichede_object(counts, coord, lib, deconv, sigma=[100, 200]))
    b = nde.calculate_effective_niche_large_scale(
        nde.create_nichede_object(counts, coord, lib, deconv, sigma=[100, 200]),
        batch_size=10)
    for x, y in zip(a.effective_niche, b.effective_niche):
        assert np.max(np.abs(x - y)) < 1e-10


def test_merge_and_filter(toy):
    counts, coord, lib, deconv = toy
    o1 = nde.create_nichede_object(counts, coord, lib, deconv, sigma=[100])
    o2 = nde.create_nichede_object(counts, coord * 3, lib, deconv, sigma=[100])
    m = nde.merge_objects([o1, o2])
    assert m.counts.shape[0] == 2 * counts.shape[0]
    assert set(np.unique(m.batch_ID)) == {1, 2}

    o1 = nde.calculate_effective_niche(o1)
    keep = list(np.asarray(o1.cell_names)[::2])
    f = nde.filter_nde(o1, keep)
    assert f.counts.shape[0] == len(keep)
    assert f.effective_niche[0].shape[0] == len(keep)


def test_int_false_path(toy):
    counts, coord, lib, deconv = toy
    obj = nde.create_nichede_object(np.log1p(counts), coord, np.log1p(lib), deconv,
                                    sigma=[100], Int=False)
    obj = nde.calculate_effective_niche(obj)
    obj = nde.niche_DE(obj, num_cores=1, C=10, M=5, Int=False, verbose=False)
    assert len(obj.niche_DE[0]) == counts.shape[1]


def test_integer_check_rejects_floats(toy):
    counts, coord, lib, deconv = toy
    with pytest.raises(ValueError, match="integers"):
        nde.create_nichede_object(counts + 0.5, coord, lib, deconv, sigma=[100])
