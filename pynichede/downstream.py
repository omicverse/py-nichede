"""Downstream Niche-DE analyses — port of the rest of ``R/niche_DE_main_functions.R``.

============================  ==========================================
R                             Python
============================  ==========================================
``get_niche_DE_genes``        :func:`get_niche_DE_genes`
``niche_DE_markers``          :func:`niche_DE_markers`
``niche_LR_spot``             :func:`niche_LR_spot`
``niche_LR_cell``             :func:`niche_LR_cell`
============================  ==========================================
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .pvalues import check_colloc, contrast_post, gene_level
from .rstats import p_adjust, r_glm_fit, r_quantile

__all__ = ["get_niche_DE_genes", "niche_DE_markers", "niche_LR_spot", "niche_LR_cell"]


def _dense(x):
    return x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) \
        else np.asarray(x, dtype=np.float64)


def _order_decreasing(x):
    """R's ``order(x, decreasing = TRUE)`` — ties broken by ascending index."""
    return np.argsort(-np.asarray(x, dtype=np.float64), kind="stable")


def _warn_colloc(obj, index, niche, name_i, name_n):
    colloc = check_colloc(obj, index, niche)
    for k, v in enumerate(colloc):
        if v < 30:
            warnings.warn(
                f"Less than 30 observations containing collocalization of {name_i} "
                f"and {name_n} at kernel bandwidth {obj.sigma[k]}. Results may be unreliable.")


# --------------------------------------------------------------------------- #
# get_niche_DE_genes
# --------------------------------------------------------------------------- #

def get_niche_DE_genes(obj, test_level: str, index=None, niche=None,
                       positive: bool = True, alpha: float = 0.05) -> pd.DataFrame:
    """``nicheDE::get_niche_DE_genes``.

    ``test_level`` is ``'G'`` (gene), ``'CT'`` (cell type) or ``'I'``
    (interaction).  Higher levels are *nested*: a gene must clear the gene-level
    gate before its cell-type p-value is considered, and both before its
    interaction p-value is.
    """
    if test_level not in ("G", "CT", "I"):
        raise ValueError("test.level must be one of G (gene), CT (cell type), or I (interaction)")
    if not isinstance(positive, (bool, np.bool_)):
        raise ValueError("positive argument must be TRUE or FALSE")

    pv = obj.niche_DE_pval_pos if positive else obj.niche_DE_pval_neg
    genes = np.asarray(obj.gene_names)
    gene_p = np.asarray(pv["gene_level"], dtype=np.float64)

    if test_level == "G":
        sel = np.flatnonzero(gene_p < alpha)
        out = pd.DataFrame({"Genes": genes[sel], "Pvalues.Gene": gene_p[sel]})
        return out.sort_values(out.columns[1]).reset_index(drop=True)

    cts = list(obj.cell_types)
    if index not in cts:
        raise ValueError("Index cell type not found")
    ct_index = cts.index(index)
    ct_p = np.asarray(pv["cell_type_level"], dtype=np.float64)

    if test_level == "CT":
        _warn_colloc(obj, ct_index, cts.index(niche) if niche in cts else ct_index,
                     index, niche)
        sel = np.flatnonzero((gene_p < alpha) & (ct_p[:, ct_index] < alpha))
        out = pd.DataFrame({"Genes": genes[sel],
                            "Pvalues.Cell.Type": ct_p[sel, ct_index]})
        return out.sort_values(out.columns[1]).reset_index(drop=True)

    if niche not in cts:
        raise ValueError("Niche cell type not found")
    niche_index = cts.index(niche)
    _warn_colloc(obj, ct_index, niche_index, index, niche)
    int_p = np.asarray(pv["interaction_level"], dtype=np.float64)
    sel = np.flatnonzero((gene_p < alpha) & (ct_p[:, ct_index] < alpha) &
                         (int_p[ct_index, niche_index, :] < alpha))
    out = pd.DataFrame({"Genes": genes[sel],
                        "Pvalues.Interaction": int_p[ct_index, niche_index, sel]})
    return out.sort_values(out.columns[1]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# niche_DE_markers
# --------------------------------------------------------------------------- #

def niche_DE_markers(obj, index: str, niche1: str, niche2: str,
                     alpha: float = 0.05) -> pd.DataFrame:
    """``nicheDE::niche_DE_markers``.

    Genes that the index cell type up-regulates next to ``niche1`` *relative to*
    ``niche2``: a contrast test on the two β̂ coefficients per kernel bandwidth,
    Cauchy-combined across bandwidths and BH-adjusted.
    """
    cts = list(obj.cell_types)
    for nm, lbl in ((index, "Index"), (niche1, "Niche1"), (niche2, "Niche2")):
        if nm not in cts:
            raise ValueError(f"{lbl} cell type not found")
    ii, n1, n2 = cts.index(index), cts.index(niche1), cts.index(niche2)
    n_type = len(cts)

    _warn_colloc(obj, ii, n1, index, niche1)
    _warn_colloc(obj, ii, n2, index, niche2)

    pvals = []
    for k in range(len(obj.sigma)):
        res = obj.niche_DE[k]
        pvals.append(contrast_post([r["betas"] for r in res],
                                   [r["Varcov"] for r in res],
                                   [r["nulls"] for r in res],
                                   ii, (n1, n2), n_type=n_type))
    pval = np.column_stack(pvals)

    log_liks = np.array([[r["log_likelihood"] for r in obj.niche_DE[k]]
                         for k in range(len(obj.niche_DE))], dtype=np.float64).T
    if log_liks.shape[1] >= 2:
        with np.errstate(invalid="ignore"):
            mx = np.nanmax(np.where(np.isnan(log_liks), -np.inf, log_liks), axis=1)
            W = np.exp(log_liks - mx[:, None])
        W[W < 0.1] = 0.0
    else:
        W = np.ones_like(pval)

    contrast = np.array([gene_level(pval[i], W[i]) for i in range(pval.shape[0])])
    contrast_bh = p_adjust(contrast, "BH")
    out = pd.DataFrame({"Genes": np.asarray(obj.gene_names), "Adj.Pvalues": contrast_bh})
    out = out[out["Adj.Pvalues"] < alpha]
    return out.sort_values("Adj.Pvalues").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# niche-LR
# --------------------------------------------------------------------------- #

def _top_kernel(obj):
    log_liks = np.array([[r["log_likelihood"] for r in obj.niche_DE[k]]
                         for k in range(len(obj.niche_DE))], dtype=np.float64).T
    if log_liks.shape[1] == 1:
        return np.zeros(log_liks.shape[0], dtype=int), log_liks
    return np.array([_order_decreasing(row)[0] for row in log_liks]), log_liks


def _ligand_scores(obj, ligand_target_matrix, index, niche, K, M, truncation_value):
    """Shared first half of ``niche_LR_spot`` / ``niche_LR_cell``."""
    top_kernel, _ = _top_kernel(obj)

    T_vector = []
    for k in range(len(obj.sigma)):
        v = np.array([r["T_stat"][index, niche] if r["T_stat"] is not None else np.nan
                      for r in obj.niche_DE[k]])
        v = np.minimum(v, abs(truncation_value))
        v = np.maximum(v, -abs(truncation_value))
        T_vector.append(v)

    L = obj.ref_expr
    L_genes = list(L.columns)
    Lv = _dense(L)
    CT_filter = np.array([r_quantile(Lv[i], 0.25) for i in range(Lv.shape[0])])

    cand_lig = set(np.asarray(L_genes)[Lv[niche] > CT_filter[niche]])
    ltm = ligand_target_matrix.loc[:, [c for c in ligand_target_matrix.columns
                                       if c in cand_lig]]

    genes_all = np.asarray(obj.gene_names)
    pear_cor = np.full(ltm.shape[1], np.nan)
    score_norm = np.full(ltm.shape[1], np.nan)
    top_DE: dict[str, str] = {}

    ltm_index = pd.Index(ltm.index)
    for j, ligand in enumerate(ltm.columns):
        if ligand not in L_genes:
            continue
        ind = L_genes.index(ligand)
        sig = T_vector[top_kernel[ind]]
        keep = ~np.isnan(sig)
        genes = genes_all[keep]
        sig = sig[keep]

        common = ltm_index.isin(set(genes))
        lv = ltm.loc[common]
        genes_keep = np.isin(genes, lv.index.to_numpy())
        sig = sig[genes_keep]
        genes = genes[genes_keep]
        lv = lv.loc[genes]                       # reorder to the data's gene order

        col = lv.iloc[:, j].to_numpy(dtype=np.float64)
        col = (col - col.mean()) / col.std(ddof=1)          # R's scale()
        top_cors = _order_decreasing(col)[:K]
        weight = col[top_cors] / col[top_cors].mean()
        pear_cor[j] = float(np.sum(sig[top_cors] * weight))
        best5 = _order_decreasing(sig[top_cors] * weight)[:5]
        top_DE[ligand] = ",".join(np.asarray(lv.index)[top_cors][best5])
        score_norm[j] = float(np.sum(weight ** 2))

    pear_cor = pear_cor / np.sqrt(score_norm)
    with np.errstate(invalid="ignore"):
        srt = np.sort(pear_cor)[::-1]
        cut = max(1.64, srt[M - 1]) if M <= srt.size and not np.isnan(srt[M - 1]) else 1.64
        top_index = np.flatnonzero(pear_cor >= cut)
    top_genes = list(np.asarray(ltm.columns)[top_index])
    return top_genes, top_DE, Lv, CT_filter, L_genes, top_kernel


def _glm_celltype_test(Y, nst, pos1: int):
    """R's ``glm(Y ~ nst, family = 'poisson')`` coefficient test for column ``pos1``.

    ``pos1`` is the 0-based column of ``nst``; R reads ``coef(check)[pos1 + 2]``
    (intercept first) and pulls the matching row out of ``summary()``'s
    coefficient table, which has already dropped the aliased rows.
    """
    design = np.column_stack([np.ones(Y.size), nst])
    fit = r_glm_fit(design, Y, family="poisson")
    beta = fit.coefficients[pos1 + 1]
    if np.isnan(beta):
        return np.nan, np.nan
    keep = np.sort(fit.pivot[:fit.rank])
    Xk = design[:, keep]
    W = fit.fitted_values
    try:
        cov = np.linalg.inv((Xk * W[:, None]).T @ Xk)
    except np.linalg.LinAlgError:                             # pragma: no cover
        return np.nan, np.nan
    pos_in_keep = int(np.flatnonzero(keep == pos1 + 1)[0])
    se = np.sqrt(cov[pos_in_keep, pos_in_keep])
    z = beta / se
    return float(beta), float(2.0 * norm.sf(abs(z)))


def _finish_lr(lr_mat, ligands, L_genes, receptor_beta, receptor_p, top_DE, alpha):
    df = lr_mat.copy()
    df.columns = ["ligand", "receptor"] + list(df.columns[2:])
    df = df.iloc[:, :2]
    df = df[df["ligand"].isin(ligands) & df["receptor"].isin(set(L_genes))]
    df = df.reset_index(drop=True)
    df["receptor_beta"] = receptor_beta
    df["receptor_pval"] = p_adjust(np.asarray(receptor_p, dtype=np.float64), "BH")
    if receptor_beta is not None and not all(b is None for b in receptor_beta):
        df = df[df["receptor_beta"] > 0]
    df = df[df["receptor_pval"] < alpha]
    if df.empty:
        raise ValueError("no ligand-receptor pairs to report")
    out = df[["ligand", "receptor"]].copy()
    out["top_downstream_niche_DE_genes"] = [top_DE.get(lig, "") for lig in out["ligand"]]
    return out.reset_index(drop=True)


def niche_LR_spot(obj, ligand_cell: str, receptor_cell: str, ligand_target_matrix,
                  lr_mat, K: int = 25, M: int = 50, alpha: float = 0.05,
                  truncation_value: float = 3) -> pd.DataFrame:
    """``nicheDE::niche_LR_spot`` — ligand–receptor inference for spot-level data.

    A ligand scores highly when the index cell type's niche-DE T-statistics for
    that ligand's top-``K`` NicheNet target genes are large and positive; the
    candidate is then confirmed by a Poisson regression of its own expression
    on the per-spot cell-type composition.
    """
    cts = list(obj.cell_types)
    niche, index = cts.index(ligand_cell), cts.index(receptor_cell)
    _warn_colloc(obj, index, niche, receptor_cell, ligand_cell)

    top_genes, top_DE, Lv, CT_filter, L_genes, top_kernel = _ligand_scores(
        obj, ligand_target_matrix, index, niche, K, M, truncation_value)
    if not top_genes:
        raise ValueError("No candidate ligands")

    counts = _dense(obj.counts)
    num_cells = _dense(obj.num_cells)

    lig_beta, lig_p = [], []
    for gene in top_genes:
        ig = L_genes.index(gene)
        tk = top_kernel[ig]
        en = obj.effective_niche[tk][:, index]
        sel = np.flatnonzero(en > en.min())
        Y = counts[sel, ig]
        nst = num_cells[sel].copy()
        nst[:, Lv[:, ig] < CT_filter] = 0.0
        if Lv[niche, ig] < CT_filter[niche]:
            lig_beta.append(np.nan)
            lig_p.append(np.nan)
            continue
        b, p = _glm_celltype_test(Y, nst, niche)
        lig_beta.append(b)
        lig_p.append(p)

    lig_p = p_adjust(np.asarray(lig_p, dtype=np.float64), "BH")
    lig_beta = np.asarray(lig_beta, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ligands = [g for g, p, b in zip(top_genes, lig_p, lig_beta)
                   if p < alpha and b > 0]

    lm = lr_mat.copy()
    lm.columns = ["ligand", "receptor"] + list(lm.columns[2:])
    sub = lm[lm["ligand"].isin(ligands) & lm["receptor"].isin(set(L_genes))].reset_index(drop=True)

    rec_beta, rec_p = [], []
    for _, row in sub.iterrows():
        ig = L_genes.index(row["receptor"])
        il = L_genes.index(row["ligand"])
        tk = top_kernel[il]
        en = obj.effective_niche[tk][:, niche]
        sel = np.flatnonzero(en > en.min())
        Y = counts[sel, ig]
        nst = num_cells[sel].copy()
        nst[:, Lv[:, il] < CT_filter] = 0.0
        if Lv[index, il] < CT_filter[index]:
            rec_beta.append(np.nan)
            rec_p.append(np.nan)
            continue
        b, p = _glm_celltype_test(Y, nst, index)
        rec_beta.append(b)
        rec_p.append(p)

    return _finish_lr(sub, ligands, L_genes, rec_beta, rec_p, top_DE, alpha)


def niche_LR_cell(obj, ligand_cell: str, receptor_cell: str, ligand_target_matrix,
                  lr_mat, K: int = 25, M: int = 50, alpha: float = 0.05,
                  alpha_2: float = 0.5, truncation_value: float = 3) -> pd.DataFrame:
    """``nicheDE::niche_LR_cell`` — the single-cell-resolution variant.

    Identical ligand scoring; the confirmation step replaces the Poisson
    regression with a normal-approximation test of the ligand's mean expression
    in pure cells of the niche type against the ``alpha_2`` quantile of the
    reference profile.
    """
    cts = list(obj.cell_types)
    niche, index = cts.index(ligand_cell), cts.index(receptor_cell)
    _warn_colloc(obj, index, niche, receptor_cell, ligand_cell)

    top_genes, top_DE, Lv, CT_filter, L_genes, top_kernel = _ligand_scores(
        obj, ligand_target_matrix, index, niche, K, M, truncation_value)

    counts = _dense(obj.counts)
    num_cells = _dense(obj.num_cells)
    ref = _dense(obj.ref_expr)

    def _norm_test(Y, nst, col, ref_row):
        m = nst[:, col] == 1
        if m.sum() == 0:
            return np.nan
        lambda_niche = float(Y[m].mean())
        lambda_rest = float(r_quantile(ref_row, alpha_2))
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (lambda_niche - lambda_rest) / np.sqrt(lambda_niche / m.sum())
        return float(1.0 - norm.cdf(t))

    lig_p = []
    for gene in top_genes:
        ig = L_genes.index(gene)
        en = obj.effective_niche[top_kernel[ig]][:, index]
        sel = np.flatnonzero(en > en.min())
        nst = num_cells[sel].copy()
        nst[:, Lv[:, ig] < CT_filter] = 0.0
        lig_p.append(_norm_test(counts[sel, ig], nst, niche, ref[niche]))

    lig_p = p_adjust(np.asarray(lig_p, dtype=np.float64), "BH")
    with np.errstate(invalid="ignore"):
        ligands = [g for g, p in zip(top_genes, lig_p) if p < alpha]

    lm = lr_mat.copy()
    lm.columns = ["ligand", "receptor"] + list(lm.columns[2:])
    sub = lm[lm["ligand"].isin(ligands) & lm["receptor"].isin(set(L_genes))].reset_index(drop=True)

    rec_p = []
    for _, row in sub.iterrows():
        ig = L_genes.index(row["receptor"])
        il = L_genes.index(row["ligand"])
        en = obj.effective_niche[top_kernel[il]][:, niche]
        sel = np.flatnonzero(en > en.min())
        nst = num_cells[sel].copy()
        nst[:, Lv[:, il] < CT_filter] = 0.0
        rec_p.append(_norm_test(counts[sel, ig], nst, index, ref[index]))

    return _finish_lr(sub, ligands, L_genes, None, rec_p, top_DE, alpha)
