"""Compute every parity number in ``data/manifest.yaml`` and print a table.

    python tests/parity_report.py <ref_dir> [--json out.json]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refload import RefDump                                   # noqa: E402


def det(ref, cand):
    ref = np.asarray(ref, dtype=float).ravel()
    cand = np.asarray(cand, dtype=float).ravel()
    if ref.shape != cand.shape:
        return dict(max_abs_err=np.inf, note=f"shape {ref.shape} vs {cand.shape}")
    m = np.isfinite(ref) & np.isfinite(cand)
    both_nan = (~np.isfinite(ref)) & (~np.isfinite(cand))
    return dict(max_abs_err=float(np.max(np.abs(ref[m] - cand[m]))) if m.any() else 0.0,
                n=int(m.sum()),
                nan_agree=float((m | both_nan).mean()))


def corr(ref, cand):
    ref = np.asarray(ref, dtype=float).ravel()
    cand = np.asarray(cand, dtype=float).ravel()
    m = np.isfinite(ref) & np.isfinite(cand)
    if m.sum() < 2:
        return dict(pearson=np.nan, spearman=np.nan, n=int(m.sum()))
    return dict(pearson=float(pearsonr(ref[m], cand[m])[0]),
                spearman=float(spearmanr(ref[m], cand[m])[0]),
                n=int(m.sum()),
                max_abs_err=float(np.max(np.abs(ref[m] - cand[m]))),
                defined_agree=float((np.isfinite(ref) == np.isfinite(cand)).mean()))


def infer(ref, cand, k=50):
    ref = np.asarray(ref, dtype=float).ravel()
    cand = np.asarray(cand, dtype=float).ravel()
    m = np.isfinite(ref) & np.isfinite(cand)
    r = np.clip(ref[m], 1e-300, 1.0)
    c = np.clip(cand[m], 1e-300, 1.0)
    if r.size < 2:
        return dict(spearman_neglog10p=np.nan, top50_jaccard=np.nan, n=int(r.size))
    rho = float(spearmanr(-np.log10(r), -np.log10(c))[0])
    kk = min(k, r.size)
    tr = set(np.argsort(r, kind="stable")[:kk].tolist())
    tc = set(np.argsort(c, kind="stable")[:kk].tolist())
    return dict(spearman_neglog10p=rho,
                top50_jaccard=float(len(tr & tc) / len(tr | tc)),
                pearson_neglog10p=float(pearsonr(-np.log10(r), -np.log10(c))[0]),
                n=int(r.size),
                defined_agree=float((np.isfinite(ref) == np.isfinite(cand)).mean()))


def main():
    ref_dir = sys.argv[1]
    d = RefDump(ref_dir)
    C = np.load(os.path.join(ref_dir, "candidate.npz"), allow_pickle=True)
    nsig = len(np.atleast_1d(d.meta["sigma"]))
    rows = []

    def add(name, kind, res, thr=None):
        rows.append((name, kind, res, thr))

    # --- helper probes (deterministic) ------------------------------------
    for nm, rn, cn in [("T_to_p(two.sided)", "ref_T_to_p_two", "T_to_p_two"),
                       ("T_to_p(positive)", "ref_T_to_p_pos", "T_to_p_pos"),
                       ("T_to_p(negative)", "ref_T_to_p_neg", "T_to_p_neg"),
                       ("ultosymmetric", "ref_ultosymmetric", "ultosymmetric"),
                       ("gene_level", "ref_gene_level", "gene_level"),
                       ("celltype_level", "ref_celltype_level", "celltype_level"),
                       ("nb_lik", "ref_nb_lik", "nb_lik"),
                       ("mvnconv(m2lp,side=1)", "ref_mvnconv_m2lp_s1", "mvnconv_m2lp_s1"),
                       ("mvnconv(m2lp,side=2)", "ref_mvnconv_m2lp_s2", "mvnconv_m2lp_s2"),
                       ("fisher(generalized)", "ref_fisher_generalized", "fisher_generalized")]:
        add(nm, "det", det(d[rn], C[cn]))

    # --- object construction ----------------------------------------------
    add("CreateLibraryMatrix", "det", det(d["ref_CreateLibraryMatrix"], C["CreateLibraryMatrix"]), 1e-8)
    add("num_cells", "det", det(d["ref_num_cells"], C["num_cells"]), 1e-8)
    add("coord (scaled)", "det", det(d["ref_coord"], C["coord"]), 1e-8)
    add("ref_expr", "det", det(d["ref_ref_expr"], C["ref_expr"]), 1e-8)
    for k in range(1, nsig + 1):
        add(f"effective_niche[{k}]", "det",
            det(d[f"ref_effective_niche_{k}"], C[f"effective_niche_{k}"]), 1e-8)
        if d.has(f"ref_effective_niche_lsfix_{k}"):
            add(f"effective_niche_largescale[{k}]", "det",
                det(d[f"ref_effective_niche_lsfix_{k}"], C[f"effective_niche_ls_{k}"]), 1e-8)
        add(f"effective_niche_largescale_shippedR[{k}]", "det",
            det(d[f"ref_effective_niche_ls_{k}"], C[f"effective_niche_ls_{k}"]), None)
    add("MergeObjects.coord", "det", det(d["ref_merge_coord"], C["merge_coord"]), 1e-8)
    add("MergeObjects.num_cells", "det", det(d["ref_merge_num_cells"], C["merge_num_cells"]), 1e-8)
    add("MergeObjects.batch_ID", "det", det(d["ref_merge_batch"], C["merge_batch"]), 1e-8)
    add("Filter_NDE.num_cells", "det", det(d["ref_filter_num_cells"], C["filter_num_cells"]), 1e-8)
    add("Filter_NDE.effective_niche", "det", det(d["ref_filter_en_1"], C["filter_en_1"]), 1e-8)

    # --- validity / nulls --------------------------------------------------
    for k in range(1, nsig + 1):
        rv, cv = d[f"ref_valid_{k}"], C[f"valid_{k}"]
        rows.append((f"valid flag[{k}]", "agree",
                     dict(agreement=float((rv == cv).mean()),
                          n_ref=int(rv.sum()), n_cand=int(cv.sum())), 1.0))
        add(f"n_nulls[{k}]", "det", det(d[f"ref_nnull_{k}"], C[f"nnull_{k}"]), 1e-8)
    add("nulls (first 200 genes)", "det", det(d["ref_nulls_flat"], C["nulls_flat"]), 1e-8)
    add("Varcov (first 50 valid genes)", "det", det(d["ref_varcov_flat"], C["varcov_flat"]))

    # --- test statistics ---------------------------------------------------
    for k in range(1, nsig + 1):
        add(f"T_stat[{k}]", "corr", corr(d[f"ref_T_stat_{k}"], C[f"T_stat_{k}"]), 0.99)
        add(f"betas[{k}]", "corr", corr(d[f"ref_betas_{k}"], C[f"betas_{k}"]), 0.99)
        add(f"log_likelihood[{k}]", "corr", corr(d[f"ref_loglik_{k}"], C[f"loglik_{k}"]), 0.99)

    # --- Int = FALSE path --------------------------------------------------
    if d.has("ref_cont_T_stat_1"):
        add("Int=FALSE T_stat[1]", "corr",
            corr(d["ref_cont_T_stat_1"], C["cont_T_stat_1"]), 0.99)
        rv, cv = d["ref_cont_valid_1"], C["cont_valid_1"]
        rows.append(("Int=FALSE valid flag[1]", "agree",
                     dict(agreement=float((rv == cv).mean()),
                          n_ref=int(rv.sum()), n_cand=int(cv.sum())), 1.0))
        add("Int=FALSE pval_pos_gene_level", "inf",
            infer(d["ref_cont_pval_pos_gene"], C["cont_pval_pos_gene"]), 0.90)

    # --- p-values ----------------------------------------------------------
    for tag in ("pos", "neg"):
        add(f"pval_{tag}_gene_level", "inf", infer(d[f"ref_pval_{tag}_gene"], C[f"pval_{tag}_gene"]), 0.90)
        add(f"pval_{tag}_cell_type_level", "inf", infer(d[f"ref_pval_{tag}_ct"], C[f"pval_{tag}_ct"]), 0.90)
        add(f"pval_{tag}_interaction_level", "inf", infer(d[f"ref_pval_{tag}_int"], C[f"pval_{tag}_int"]), 0.90)
    add("praw_pos_gene_level", "inf", infer(d["ref_praw_pos_gene"], C["praw_pos_gene"]), 0.90)
    add("praw_pos_cell_type_level", "inf", infer(d["ref_praw_pos_ct"], C["praw_pos_ct"]), 0.90)
    add("praw_pos_interaction_level", "inf", infer(d["ref_praw_pos_int"], C["praw_pos_int"]), 0.90)

    # --- downstream --------------------------------------------------------
    add("contrast_post", "inf", infer(d["ref_contrast_post"], C["contrast_post"]), 0.90)
    add("check_colloc", "det", det(d["ref_check_colloc"], C["check_colloc"]), 1e-8)

    for nm in ["genes_G_pos", "genes_G_neg", "genes_CT_pos", "genes_CT_neg",
               "genes_I_pos", "genes_I_neg", "markers",
               "niche_LR_spot", "niche_LR_cell"]:
        r = d.csv("ref_" + nm)
        import pandas as pd
        cp = os.path.join(ref_dir, "cand_" + nm + ".csv")
        try:
            c = pd.read_csv(cp) if os.path.exists(cp) else None
        except pd.errors.EmptyDataError:
            c = pd.DataFrame()
        if r is None or c is None:
            continue
        if nm.startswith("niche_LR"):
            gr = set(map(tuple, r.iloc[:, :2].astype(str).to_numpy())) if len(r) else set()
            gc = set(map(tuple, c.iloc[:, :2].astype(str).to_numpy())) if len(c) else set()
        else:
            gr = set(r.iloc[:, 0].astype(str)) if len(r) else set()
            gc = set(c.iloc[:, 0].astype(str)) if len(c) else set()
        jac = len(gr & gc) / len(gr | gc) if (gr | gc) else 1.0
        rows.append((nm, "set", dict(jaccard=float(jac), n_ref=len(gr), n_cand=len(gc)), 0.7))

    # --- print -------------------------------------------------------------
    print(f"\n{'output':38s} {'kind':6s} {'metric':>44s}  {'thr':>8s}")
    print("-" * 104)
    out_json = {}
    for name, kind, res, thr in rows:
        out_json[name] = {"kind": kind, **{k: (None if isinstance(v, float) and np.isnan(v) else v)
                                           for k, v in res.items()}, "threshold": thr}
        if kind == "det":
            s = f"max|err|={res.get('max_abs_err', float('nan')):.3e} n={res.get('n', 0)}"
        elif kind == "corr":
            s = (f"pearson={res['pearson']:.6f} spearman={res['spearman']:.6f} "
                 f"n={res['n']}")
        elif kind == "inf":
            s = (f"spearman(-log10p)={res['spearman_neglog10p']:.6f} "
                 f"top50J={res['top50_jaccard']:.3f}")
        elif kind == "agree":
            s = f"agreement={res['agreement']:.6f} R={res['n_ref']} py={res['n_cand']}"
        else:
            s = f"jaccard={res['jaccard']:.3f} R={res['n_ref']} py={res['n_cand']}"
        print(f"{name:38s} {kind:6s} {s:>44s}  {str(thr):>8s}")

    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as fh:
            json.dump(out_json, fh, indent=2)


if __name__ == "__main__":
    main()
