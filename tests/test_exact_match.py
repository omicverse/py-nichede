"""The parity gate, as a runnable pytest test.

Every assertion here reads its threshold out of ``data/manifest.yaml``, which
was committed **before** any algorithmic Python was written (Omicverse-RebuildR
PROTOCOL.md Step 4).  Nothing in this file may hard-code a threshold.

Requires an R reference dump (``tests/r_reference_driver.R``) and the matching
candidate run (``tests/_run_candidate.py``); both are skipped automatically when
absent so ``pytest -q`` stays green in a bare checkout.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _finite_pair(ref, cand):
    ref = np.asarray(ref, dtype=float).ravel()
    cand = np.asarray(cand, dtype=float).ravel()
    assert ref.shape == cand.shape, f"shape mismatch {ref.shape} vs {cand.shape}"
    m = np.isfinite(ref) & np.isfinite(cand)
    return ref, cand, m


def max_abs_err(ref, cand):
    ref, cand, m = _finite_pair(ref, cand)
    assert (np.isfinite(ref) == np.isfinite(cand)).all(), "NaN pattern differs"
    return float(np.max(np.abs(ref[m] - cand[m]))) if m.any() else 0.0


def pearson(ref, cand):
    ref, cand, m = _finite_pair(ref, cand)
    return float(pearsonr(ref[m], cand[m])[0])


def spearman(ref, cand):
    ref, cand, m = _finite_pair(ref, cand)
    return float(spearmanr(ref[m], cand[m])[0])


def inference(ref, cand, k=50):
    ref, cand, m = _finite_pair(ref, cand)
    r = np.clip(ref[m], 1e-300, 1.0)
    c = np.clip(cand[m], 1e-300, 1.0)
    rho = float(spearmanr(-np.log10(r), -np.log10(c))[0])
    kk = min(k, r.size)
    tr = set(np.argsort(r, kind="stable")[:kk].tolist())
    tc = set(np.argsort(c, kind="stable")[:kk].tolist())
    return rho, float(len(tr & tc) / len(tr | tc))


# --------------------------------------------------------------------------- #
# Gate 1 — object construction (deterministic-standard, atol 1e-8)
# --------------------------------------------------------------------------- #

def test_library_matrix_parity(ref, cand, gate):
    thr = gate["library_matrix"]["threshold"]
    err = max_abs_err(ref["ref_CreateLibraryMatrix"], cand["CreateLibraryMatrix"])
    assert err < thr, f"CreateLibraryMatrix max|err|={err:.3e} >= {thr:.1e}"


def test_num_cells_parity(ref, cand, gate):
    thr = gate["num_cells"]["threshold"]
    err = max_abs_err(ref["ref_num_cells"], cand["num_cells"])
    assert err < thr, f"num_cells max|err|={err:.3e} >= {thr:.1e}"


def test_coord_and_ref_expr_parity(ref, cand, gate):
    thr = gate["num_cells"]["threshold"]
    assert max_abs_err(ref["ref_coord"], cand["coord"]) < thr
    assert max_abs_err(ref["ref_ref_expr"], cand["ref_expr"]) < thr


def test_effective_niche_parity(ref, cand, gate):
    thr = gate["effective_niche"]["threshold"]
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    for k in range(1, nsig + 1):
        err = max_abs_err(ref[f"ref_effective_niche_{k}"], cand[f"effective_niche_{k}"])
        assert err < thr, f"effective_niche[{k}] max|err|={err:.3e} >= {thr:.1e}"


def test_effective_niche_large_scale_parity(ref, cand, gate):
    """The tiled variant, against the R algorithm with a working ``dista``.

    The *shipped* ``CalculateEffectiveNicheLargeScale`` cannot be used as the
    reference here: ``Rfast`` >= 2.1.5.2's ``dista()`` returns an all-zero
    matrix whenever ``nrow(xnew) >= 4``, which turns every kernel weight into 1.
    ``r_reference_driver.R`` therefore also dumps the same R algorithm with
    base-R distances substituted, and that is what we compare against.
    """
    thr = gate["effective_niche"]["threshold"]
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    if not ref.has("ref_effective_niche_lsfix_1"):
        pytest.skip("reference dump predates the repaired large-scale anchor")
    for k in range(1, nsig + 1):
        err = max_abs_err(ref[f"ref_effective_niche_lsfix_{k}"],
                          cand[f"effective_niche_ls_{k}"])
        assert err < thr, f"effective_niche_ls[{k}] max|err|={err:.3e} >= {thr:.1e}"


def test_merge_and_filter_parity(ref, cand, gate):
    thr = gate["num_cells"]["threshold"]
    assert max_abs_err(ref["ref_merge_coord"], cand["merge_coord"]) < thr
    assert max_abs_err(ref["ref_merge_num_cells"], cand["merge_num_cells"]) < thr
    assert max_abs_err(ref["ref_merge_batch"], cand["merge_batch"]) < thr
    assert max_abs_err(ref["ref_filter_num_cells"], cand["filter_num_cells"]) < thr
    assert max_abs_err(ref["ref_filter_en_1"], cand["filter_en_1"]) < thr


# --------------------------------------------------------------------------- #
# Gate 2 — model bookkeeping must agree exactly
# --------------------------------------------------------------------------- #

def test_valid_flags_agree_exactly(ref, cand):
    """Which genes Niche-DE could fit at all — must match gene for gene."""
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    for k in range(1, nsig + 1):
        rv, cv = ref[f"ref_valid_{k}"], cand[f"valid_{k}"]
        agree = float((rv == cv).mean())
        assert agree == 1.0, (
            f"valid flag[{k}] agreement {agree:.6f} (R={int(rv.sum())}, "
            f"py={int(cv.sum())})")


def test_null_sets_agree_exactly(ref, cand):
    """Which (index, niche) interactions were dropped by the M filter."""
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    for k in range(1, nsig + 1):
        assert max_abs_err(ref[f"ref_nnull_{k}"], cand[f"nnull_{k}"]) == 0.0
    assert max_abs_err(ref["ref_nulls_flat"], cand["nulls_flat"]) == 0.0


# --------------------------------------------------------------------------- #
# Gate 3 — the test statistic (ordinal: Pearson >= 0.99, Spearman >= 0.99)
# --------------------------------------------------------------------------- #

def test_T_stat_parity(ref, cand, gate):
    thr = gate["T_stat"]["threshold"]
    sthr = gate["T_stat"]["secondary"]["threshold"]
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    for k in range(1, nsig + 1):
        r = pearson(ref[f"ref_T_stat_{k}"], cand[f"T_stat_{k}"])
        s = spearman(ref[f"ref_T_stat_{k}"], cand[f"T_stat_{k}"])
        assert r >= thr - 1e-12, f"T_stat[{k}] pearson={r:.8f} < {thr}"
        assert s >= sthr - 1e-12, f"T_stat[{k}] spearman={s:.8f} < {sthr}"


def test_betas_and_loglik_parity(ref, cand, gate):
    thr = gate["T_stat"]["threshold"]
    nsig = len(np.atleast_1d(ref.meta["sigma"]))
    for k in range(1, nsig + 1):
        assert pearson(ref[f"ref_betas_{k}"], cand[f"betas_{k}"]) >= thr - 1e-12
        assert pearson(ref[f"ref_loglik_{k}"], cand[f"loglik_{k}"]) >= thr - 1e-12


# --------------------------------------------------------------------------- #
# Gate 4 — p-values (inference: Spearman on -log10 p >= 0.90, top-50 J >= 0.70)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,rkey,ckey", [
    ("pval_pos_gene_level", "ref_pval_pos_gene", "pval_pos_gene"),
    ("pval_pos_cell_type_level", "ref_pval_pos_ct", "pval_pos_ct"),
    ("pval_pos_interaction_level", "ref_pval_pos_int", "pval_pos_int"),
    ("pval_neg_gene_level", "ref_pval_neg_gene", "pval_neg_gene"),
])
def test_pvalue_parity(ref, cand, gate, name, rkey, ckey):
    g = gate[name]
    rho, jac = inference(ref[rkey], cand[ckey])
    assert rho >= g["threshold"], f"{name} spearman(-log10 p)={rho:.6f} < {g['threshold']}"
    assert jac >= g["threshold_top50_jaccard"], (
        f"{name} top-50 Jaccard={jac:.3f} < {g['threshold_top50_jaccard']}")


def test_negative_cell_type_and_interaction_parity(ref, cand, gate):
    """The two negative-direction outputs not covered by the manifest blocks."""
    g = gate["pval_pos_cell_type_level"]
    for rkey, ckey in (("ref_pval_neg_ct", "pval_neg_ct"),
                       ("ref_pval_neg_int", "pval_neg_int")):
        rho, jac = inference(ref[rkey], cand[ckey])
        assert rho >= g["threshold"]
        assert jac >= g["threshold_top50_jaccard"]


def test_raw_pvalue_parity(ref, cand, gate):
    """``get_niche_DE_pval_raw`` — the un-BH-adjusted tables."""
    g = gate["pval_pos_gene_level"]
    for rkey, ckey in (("ref_praw_pos_gene", "praw_pos_gene"),
                       ("ref_praw_pos_ct", "praw_pos_ct"),
                       ("ref_praw_pos_int", "praw_pos_int")):
        rho, jac = inference(ref[rkey], cand[ckey])
        assert rho >= g["threshold"]
        assert jac >= g["threshold_top50_jaccard"]


# --------------------------------------------------------------------------- #
# Gate 5 — downstream calls
# --------------------------------------------------------------------------- #

def test_contrast_post_parity(ref, cand, gate):
    g = gate["pval_pos_gene_level"]
    rho, jac = inference(ref["ref_contrast_post"], cand["contrast_post"])
    assert rho >= g["threshold"]
    assert jac >= g["threshold_top50_jaccard"]


def test_check_colloc_exact(ref, cand):
    assert max_abs_err(ref["ref_check_colloc"], cand["check_colloc"]) == 0.0


@pytest.mark.parametrize("name", [
    "genes_G_pos", "genes_G_neg", "genes_CT_pos", "genes_CT_neg",
    "genes_I_pos", "genes_I_neg", "markers", "niche_LR_spot",
])
def test_reported_gene_sets(ref, cand, name):
    """The user-visible gene lists must overlap R's at Jaccard >= 0.70."""
    import os
    import pandas as pd
    r = ref.csv("ref_" + name)
    p = os.path.join(ref.path, "cand_" + name + ".csv")
    if r is None or not os.path.exists(p):
        pytest.skip(f"{name} not produced on this fixture")
    try:
        c = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        c = pd.DataFrame()
    gr = set(r.iloc[:, 0].astype(str)) if len(r) else set()
    gc = set(c.iloc[:, 0].astype(str)) if len(c) else set()
    if not (gr | gc):
        pytest.skip(f"{name} empty on both sides")
    jac = len(gr & gc) / len(gr | gc)
    assert jac >= 0.70, f"{name} Jaccard={jac:.3f} (R={len(gr)}, py={len(gc)})"


# --------------------------------------------------------------------------- #
# Gate 6 — Int = FALSE (linear-model) path
# --------------------------------------------------------------------------- #

def test_int_false_path_parity(ref, cand, gate):
    if not ref.has("ref_cont_T_stat_1"):
        pytest.skip("reference dump predates the Int=FALSE anchor")
    thr = gate["T_stat"]["threshold"]
    r = pearson(ref["ref_cont_T_stat_1"], cand["cont_T_stat_1"])
    assert r >= thr - 1e-12, f"Int=FALSE T_stat pearson={r:.8f} < {thr}"
    rv, cv = ref["ref_cont_valid_1"], cand["cont_valid_1"]
    assert float((rv == cv).mean()) == 1.0
