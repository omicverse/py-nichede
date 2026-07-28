"""Multi-batch parity gate.

The canonical fixture is a single tissue section, so nothing in
``test_exact_match.py`` exercises the multi-batch code path: ``MergeObjects``'
coordinate renormalisation, the factor ``batchvar`` that ``model.matrix``
expands into ``nlevels - 1`` treatment-contrast dummies, and the extra dummy
block that ``niche_DE_core`` appends to ``X'WX`` before the Cholesky.

``tests/r_reference_supplement.R`` therefore also merges the fixture with a
1.5x-rescaled copy of itself, runs the R ``niche_DE`` on the two-batch object,
and dumps the result as ``ref_mb_*``.  This module gates the Python port
against it at the same pre-registered thresholds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

import pynichede as nde


@pytest.fixture(scope="module")
def merged(ref):
    if not ref.has("ref_mb_T_stat_1"):
        pytest.skip("reference dump predates the multi-batch anchor "
                    "(re-run tests/r_reference_supplement.R)")
    cts = list(ref.meta["cell_types"])
    cells = list(ref.meta["cell_names"])
    all_genes = list(ref.meta["gene_names"])
    genes = list(ref.meta["mb_genes"])
    sigma = np.atleast_1d(np.asarray(ref.meta["sigma"], dtype=float))

    counts = pd.DataFrame(ref["ref_counts"], index=cells, columns=all_genes)[genes]
    coord = pd.DataFrame(ref["in_coord"], index=cells, columns=["x", "y"])
    lib = pd.DataFrame(ref["ref_ref_expr"], index=cts, columns=all_genes)[genes]
    dec = pd.DataFrame(ref["in_deconv"], index=cells, columns=cts)

    o1 = nde.create_nichede_object(counts, coord, lib, dec, sigma=sigma)
    o2 = nde.create_nichede_object(counts, coord * 1.5, lib, dec, sigma=sigma)
    om = nde.merge_objects([o1, o2])
    om = nde.calculate_effective_niche(om, cutoff=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        om = nde.niche_DE(om, num_cores=4, C=150, M=10, gamma=0.8,
                          batch=True, verbose=False)
    return om, len(cts), len(genes)


def test_merged_object_construction(ref, merged, gate):
    om, _, _ = merged
    thr = gate["num_cells"]["threshold"]
    assert set(np.unique(om.batch_ID)) == {1, 2}
    assert np.max(np.abs(ref["ref_mb_coord"] - np.asarray(om.coord))) < thr
    assert np.max(np.abs(ref["ref_mb_num_cells"] - np.asarray(om.num_cells))) < thr
    assert np.max(np.abs(ref["ref_mb_batch"] - np.asarray(om.batch_ID))) == 0


def test_merged_effective_niche(ref, merged, gate):
    om, _, _ = merged
    thr = gate["effective_niche"]["threshold"]
    for k in range(1, len(om.sigma) + 1):
        err = np.max(np.abs(ref[f"ref_mb_en_{k}"] - om.effective_niche[k - 1]))
        assert err < thr, f"multi-batch effective_niche[{k}] max|err|={err:.3e}"


def test_multibatch_valid_flags_exact(ref, merged):
    om, _, _ = merged
    for k in range(1, len(om.sigma) + 1):
        rv = ref[f"ref_mb_valid_{k}"]
        cv = np.array([r["valid"] for r in om.niche_DE[k - 1]])
        assert float((rv == cv).mean()) == 1.0, (
            f"multi-batch valid flag[{k}]: R={int(rv.sum())} py={int(cv.sum())}")


def test_multibatch_T_stat_parity(ref, merged, gate):
    om, n_ct, n_gene = merged
    thr = gate["T_stat"]["threshold"]
    sthr = gate["T_stat"]["secondary"]["threshold"]
    for k in range(1, len(om.sigma) + 1):
        rT = ref[f"ref_mb_T_stat_{k}"].reshape(n_ct, n_ct, n_gene, order="F")
        cT = np.full((n_ct, n_ct, n_gene), np.nan)
        for g, r in enumerate(om.niche_DE[k - 1]):
            if r["valid"] == 1:
                cT[:, :, g] = r["T_stat"]
        m = np.isfinite(rT) & np.isfinite(cT)
        assert (np.isfinite(rT) == np.isfinite(cT)).all(), "NaN pattern differs"
        assert pearsonr(rT[m], cT[m])[0] >= thr - 1e-12
        assert spearmanr(rT[m], cT[m])[0] >= sthr - 1e-12


def test_multibatch_pvalue_parity(ref, merged, gate):
    om, _, _ = merged
    g = gate["pval_pos_gene_level"]
    rp = np.asarray(ref["ref_mb_pval_pos_gene"], dtype=float)
    cp = np.asarray(om.niche_DE_pval_pos["gene_level"], dtype=float)
    m = np.isfinite(rp) & np.isfinite(cp)
    rho = spearmanr(-np.log10(np.clip(rp[m], 1e-300, 1)),
                    -np.log10(np.clip(cp[m], 1e-300, 1)))[0]
    assert rho >= g["threshold"], f"multi-batch gene-level p Spearman={rho:.6f}"
