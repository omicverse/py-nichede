"""Niche-DE p-value machinery — port of ``R/niche_DE_helper_functions.R``.

Three nested levels of pooling:

* **interaction level** — one p-value per (index, niche, gene); the raw Wald
  p-values, Cauchy-combined across kernel bandwidths and BH-adjusted across
  niche cell types.
* **cell-type level** — Brown-combine the ``n_celltype`` interaction p-values
  of one index cell type (using the β̂ covariance to account for their
  dependence), Cauchy-combine across bandwidths, BH-adjust across cell types.
* **gene level** — Brown-combine all ``n_celltype^2`` interaction p-values of a
  gene, Cauchy-combine across bandwidths, BH-adjust across genes.

The Cauchy weights are ``exp(loglik - max loglik)`` truncated below 0.1, so the
bandwidth that fits a gene best dominates that gene's combined p-value.

======================================  ============================================
R                                       Python
======================================  ============================================
``nb_lik``                              :func:`~pynichede.rstats.nb_lik`
``ultosymmetric``                       :func:`ultosymmetric`
``T_to_p``                              :func:`T_to_p`
``gene_level``                          :func:`gene_level`
``celltype_level``                      :func:`celltype_level`
``gene_level_fisher``                   :func:`gene_level_fisher`
``celltype_level_fisher``               :func:`celltype_level_fisher`
``contrast_post``                       :func:`contrast_post`
``check_colloc``                        :func:`check_colloc`
``get_niche_DE_pval_fisher``            :func:`get_niche_DE_pval_fisher`
``get_niche_DE_pval_raw``               :func:`get_niche_DE_pval_raw`
======================================  ============================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import cauchy, norm

from .poolr import fisher_generalized, mvnconv
from .rstats import p_adjust, weighted_mean

__all__ = [
    "ultosymmetric", "T_to_p", "gene_level", "celltype_level",
    "gene_level_fisher", "celltype_level_fisher", "contrast_post",
    "check_colloc", "get_niche_DE_pval_fisher", "get_niche_DE_pval_raw",
]


def ultosymmetric(m):
    """``nicheDE::ultosymmetric`` — mirror an upper-triangular matrix."""
    m = np.asarray(m, dtype=np.float64)
    return m + m.T - np.diag(np.diag(m))


def T_to_p(T_stat, alternative: str = "two.sided"):
    """``nicheDE::T_to_p`` — normal tail probability of a Wald statistic."""
    if T_stat is None:
        return None
    T = np.asarray(T_stat, dtype=np.float64)
    if alternative == "positive":
        return 1.0 - norm.cdf(T)
    if alternative == "negative":
        return 1.0 - norm.cdf(-T)
    if alternative == "two.sided":
        return 1.0 - norm.cdf(np.abs(T))
    raise ValueError(f"unknown alternative {alternative!r}")     # pragma: no cover


def _cauchy_upper(x):
    """``1 - pcauchy(x)``."""
    return 1.0 - cauchy.cdf(x)


def gene_level(p, w=None) -> float:
    """``nicheDE::gene_level`` — the (weighted) Cauchy combination test."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if w is None:
        w = np.ones_like(p)
    t = np.tan((0.5 - p) * np.pi)
    return float(_cauchy_upper(weighted_mean(t, w, na_rm=True)))


def celltype_level(p, w=None):
    """``nicheDE::celltype_level`` — row-wise Cauchy combination."""
    p = np.asarray(p, dtype=np.float64)
    if w is None:
        w = np.ones(p.shape[1])
    t = np.tan((0.5 - p) * np.pi)
    out = np.array([weighted_mean(t[i], w, na_rm=True) for i in range(t.shape[0])])
    return _cauchy_upper(out)


def gene_level_fisher(p, varcov, beta_cov: bool = True) -> float:
    """``nicheDE::gene_level_fisher`` — Brown's method over one gene's p-values.

    ``p`` is the ``n_celltype x n_celltype`` matrix indexed ``[index, niche]``.
    R flattens ``t(p)`` column-wise, which is exactly the ordering of the
    design-matrix columns that ``varcov`` is expressed in, so the two line up
    without any reindexing.

    With ``beta_cov=True`` the β̂ covariance is first rescaled by
    ``diag(varcov)^-1`` (turning it into the covariance of the Wald statistics),
    converted to a correlation matrix, and then mapped through
    :func:`~pynichede.poolr.mvnconv` to the covariance of the ``-2 log p``
    statistics.
    """
    if p is None or varcov is None:
        return np.nan
    p = np.asarray(p, dtype=np.float64)
    pv = p.T.ravel(order="F")                # == as.vector(t(p)) in R
    pv = pv[~np.isnan(pv)]
    varcov = np.asarray(varcov, dtype=np.float64)
    if pv.size == 0 or varcov.ndim != 2 or pv.size != varcov.shape[0]:
        return np.nan

    if beta_cov:
        V = ultosymmetric(varcov)
        A = np.diag(1.0 / np.diag(V))
        V = A @ V @ A.T
        d = np.sqrt(np.diag(V))
        V = V / np.outer(d, d)               # cov2cor
        np.fill_diagonal(V, 1.0)
        varcov = mvnconv(V, side=1, target="m2lp", cov2cor=False)
    try:
        return float(fisher_generalized(pv, varcov))
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        return np.nan


def celltype_level_fisher(p, varcov):
    """``nicheDE::celltype_level_fisher`` — Brown's method per index cell type.

    The per-cell-type blocks are read off ``varcov`` **sequentially**: because
    the design columns are ordered ``(index, niche)`` with ``niche`` varying
    fastest, the non-null entries belonging to one index cell type form a
    contiguous run.
    """
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    pt = p.T                                  # [niche, index]
    out = np.full(pt.shape[1], np.nan)
    ind_counter = 0
    if varcov is not None and np.asarray(varcov).ndim == 2:
        V = ultosymmetric(np.asarray(varcov, dtype=np.float64))
        A = np.diag(1.0 / np.diag(V))
        V = A @ V @ A.T
        d = np.sqrt(np.diag(V))
        V = V / np.outer(d, d)
        np.fill_diagonal(V, 1.0)
        varcov = mvnconv(V, side=1, target="m2lp", cov2cor=False)
    else:
        varcov = None

    for it in range(pt.shape[1]):
        p_CT = pt[:, it]
        p_CT = p_CT[~np.isnan(p_CT)]
        add = p_CT.size
        if add > 1:
            if varcov is None:
                out[it] = np.nan
                continue
            sl = slice(ind_counter, ind_counter + add)
            out[it] = gene_level_fisher(p_CT.reshape(1, -1), varcov[sl, sl],
                                        beta_cov=False)
            ind_counter += add
        elif add == 1:
            out[it] = p_CT[0]
            ind_counter += add
    return out


def contrast_post(betas_all, V_cov_all, nulls_all, index: int, niche, n_type=None):
    """``nicheDE::contrast_post``.

    One-sided test of ``beta[index, niche1] - beta[index, niche2] > 0`` using
    the joint covariance of the two coefficients.  ``index`` and ``niche`` are
    **0-based** here (R's are 1-based).
    """
    ngene = len(betas_all)
    if n_type is None:
        for V, nl in zip(V_cov_all, nulls_all):
            if V is not None and np.asarray(V).ndim == 2:
                n_type = int(np.sqrt(len(nl) + np.asarray(V).shape[0]))
                break
        else:                                                    # pragma: no cover
            n_type = int(np.sqrt(len(nulls_all[0])))

    out = np.full(ngene, np.nan)
    for j in range(ngene):
        null = np.asarray(nulls_all[j], dtype=int)
        V = V_cov_all[j]
        if null.size == n_type ** 2 or betas_all[j] is None:
            continue
        V_cov = np.full((n_type ** 2, n_type ** 2), np.nan)
        keep = np.setdiff1d(np.arange(n_type ** 2), null)
        if V is None:
            continue
        V = np.asarray(V, dtype=np.float64)
        if V.shape[0] != keep.size:
            continue
        V_cov[np.ix_(keep, keep)] = V
        betas = np.asarray(betas_all[j], dtype=np.float64)
        try:
            i1 = index * n_type + niche[0]
            i2 = index * n_type + niche[1]
            id1, id2 = min(i1, i2), max(i1, i2)
            var_contrast = V_cov[i1, i1] + V_cov[i2, i2] - 2.0 * V_cov[id1, id2]
            T = (betas[index, niche[0]] - betas[index, niche[1]]) / np.sqrt(var_contrast)
            out[j] = 1.0 - norm.cdf(T)
        except (IndexError, FloatingPointError, ValueError):     # pragma: no cover
            continue
    return out


def check_colloc(obj, index: int, niche: int):
    """``nicheDE::check_colloc`` — number of spots where both types co-occur.

    ``index`` / ``niche`` are **0-based** column positions of ``num_cells``.
    """
    num_cells = np.asarray(obj.num_cells, dtype=np.float64)
    index_ind = np.flatnonzero(num_cells[:, index] > 0)
    out = np.empty(len(obj.sigma))
    for k in range(len(obj.sigma)):
        en = obj.effective_niche[k][:, niche]
        niche_ind = np.flatnonzero(en > en.min())
        out[k] = np.isin(niche_ind, index_ind).sum()
    return out


# --------------------------------------------------------------------------- #
# The orchestrator
# --------------------------------------------------------------------------- #

def _loglik_matrix(obj):
    ll = np.array([[r["log_likelihood"] for r in obj.niche_DE[k]]
                   for k in range(len(obj.niche_DE))], dtype=np.float64).T
    return ll


def _pval_core(obj, pos: bool, adjust: bool, verbose: bool = True):
    n_sig = len(obj.sigma)
    ngene = len(obj.gene_names)
    n_ct = len(obj.cell_types)

    log_liks = _loglik_matrix(obj)
    alt = "positive" if pos else "negative"
    T_stat_list = [[T_to_p(r["T_stat"], alt) for r in obj.niche_DE[k]]
                   for k in range(n_sig)]
    if n_sig == 1:
        T_stat_list = T_stat_list + [T_stat_list[0]]
        log_liks = np.column_stack([log_liks[:, 0], log_liks[:, 0]])
    n_slots = len(T_stat_list)

    # ---- gene level -------------------------------------------------------
    if verbose:
        print("Computing Gene Level Pvalues")
    gene_p = np.full((ngene, n_slots), np.nan)
    for k in range(n_slots):
        if k >= n_sig:
            gene_p[:, k] = gene_p[:, 0]
            continue
        varcovs = [r["Varcov"] for r in obj.niche_DE[k]]
        gene_p[:, k] = [gene_level_fisher(T_stat_list[k][g], varcovs[g])
                        for g in range(ngene)]

    if verbose:
        print("Combining Gene Level Pvalues Across Kernel Bandwidths")
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(np.where(np.isnan(log_liks), -np.inf, log_liks), axis=1)
        W = np.exp(log_liks - mx[:, None])
    W[W < 0.1] = 0.0
    gene_pv = np.array([gene_level(gene_p[i], W[i]) for i in range(ngene)])
    gene_out = p_adjust(gene_pv, "BH") if adjust else gene_pv

    # ---- cell type level --------------------------------------------------
    if verbose:
        print("Computing Cell Type Level Pvalues")
    CT_p = []
    for k in range(n_slots):
        if k >= n_sig:
            CT_p.append(CT_p[0])
            continue
        varcovs = [r["Varcov"] for r in obj.niche_DE[k]]
        M = np.full((ngene, n_ct), np.nan)
        for g in range(ngene):
            v = celltype_level_fisher(T_stat_list[k][g], varcovs[g])
            if v is not None:
                M[g, :len(v)] = v
        CT_p.append(M)
    CT_merge = np.stack(CT_p, axis=2)

    if verbose:
        print("Combining Cell Type Level Pvalues Across Kernel Bandwidths")
    CT_cauchy = np.empty((ngene, n_ct))
    for i in range(ngene):
        for j in range(n_ct):
            CT_cauchy[i, j] = gene_level(CT_merge[i, j, :], W[i])
    CT_out = np.vstack([p_adjust(CT_cauchy[j], "BH") for j in range(ngene)]) \
        if adjust else CT_cauchy

    # ---- interaction level ------------------------------------------------
    if verbose:
        print("Computing and Combining interaction Level Pvalues Across Kernel bandwidths")
    pgt_merge = np.zeros((n_ct, n_ct, ngene))
    valid = np.zeros((n_ct, n_ct, ngene))
    for s in range(n_slots):
        pgt_temp = np.zeros((n_ct, n_ct, ngene))
        for g in range(ngene):
            x = T_stat_list[s][g]
            if x is None:
                pgt = np.full((n_ct, n_ct), np.nan)
            else:
                pgt = np.tan((0.5 - np.asarray(x)) * np.pi)
            pgt_temp[:, :, g] = pgt * W[g, s]
            valid[:, :, g] = valid[:, :, g] + W[g, s] * (1 - np.isnan(pgt))
        pgt_temp[np.isnan(pgt_temp)] = 0.0
        pgt_merge = pgt_merge + pgt_temp
    with np.errstate(invalid="ignore", divide="ignore"):
        pgt_merge[valid == 0] = np.nan
        pgt_merge = pgt_merge / valid
        pgt_cauchy = _cauchy_upper(pgt_merge)

    if adjust:
        pgt_out = pgt_cauchy.copy()
        for g in range(ngene):
            for k in range(n_ct):
                pgt_out[k, :, g] = p_adjust(pgt_cauchy[k, :, g], "BH")
    else:
        pgt_out = pgt_cauchy

    return {
        "gene_level": gene_out,
        "cell_type_level": pd.DataFrame(CT_out, index=obj.gene_names,
                                        columns=obj.cell_types),
        "interaction_level": pgt_out,
    }


def get_niche_DE_pval_fisher(obj, pos: bool = True, verbose: bool = True):
    """``nicheDE::get_niche_DE_pval_fisher`` — BH-adjusted p-values."""
    res = _pval_core(obj, pos=pos, adjust=True, verbose=verbose)
    if pos:
        obj.niche_DE_pval_pos = res
    else:
        obj.niche_DE_pval_neg = res
    return obj


def get_niche_DE_pval_raw(obj, pos: bool = True, verbose: bool = True):
    """``nicheDE::get_niche_DE_pval_raw`` — the same p-values without BH."""
    res = _pval_core(obj, pos=pos, adjust=False, verbose=verbose)
    if pos:
        obj.niche_DE_pval_pos = res
    else:
        obj.niche_DE_pval_neg = res
    return obj
