"""Run the Python port on the same inputs the R reference driver used.

Reads the ``in_*`` binaries the R driver dumped (so both sides start from
byte-identical data), runs the full pipeline, and writes the candidate outputs
next to the reference as ``cand_*.npz``.

    python tests/_run_candidate.py <ref_dir> [n_jobs]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refload import RefDump                                   # noqa: E402
import pynichede as nde                                       # noqa: E402


def build(ref_dir):
    d = RefDump(ref_dir)
    genes = list(d.meta["gene_names"])
    cells = list(d.meta["cell_names"])
    cts = list(d.meta["cell_types"])
    sigma = np.atleast_1d(np.asarray(d.meta["sigma"], dtype=float))

    counts_in = d["in_counts"]
    coord_in = d["in_coord"]
    lib_in = d["in_libmat"]
    dec_in = d["in_deconv"]

    # the R driver dumped inputs *before* gene filtering, so recover the names
    all_genes = None
    lib_genes = None
    return d, genes, cells, cts, sigma, counts_in, coord_in, lib_in, dec_in


def main():
    ref_dir = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    d = RefDump(ref_dir)

    cts = list(d.meta["cell_types"])
    cells = list(d.meta["cell_names"])
    genes = list(d.meta["gene_names"])           # post-filter, library order
    sigma = np.atleast_1d(np.asarray(d.meta["sigma"], dtype=float))

    # Reconstruct the *pre-filter* frames.  ref_counts / ref_ref_expr are the
    # post-filter matrices in library order, which is what CreateNicheDEObject
    # produces; feeding them back in is a fixed point of the gene-intersection
    # logic, so the object construction still exercises the real code path.
    counts = pd.DataFrame(d["ref_counts"], index=cells, columns=genes)
    coord = pd.DataFrame(d["in_coord"], index=cells, columns=["imagerow", "imagecol"])
    libmat = pd.DataFrame(d["ref_ref_expr"], index=cts, columns=genes)
    deconv = pd.DataFrame(d["in_deconv"], index=cells, columns=cts)

    out = {}
    t0 = time.time()
    obj = nde.create_nichede_object(counts, coord, libmat, deconv,
                                    sigma=sigma, Int=True)
    out["t_create"] = time.time() - t0
    out["num_cells"] = np.asarray(obj.num_cells, dtype=float)
    out["coord"] = np.asarray(obj.coord, dtype=float)
    out["ref_expr"] = np.asarray(obj.ref_expr, dtype=float)
    out["scale"] = np.atleast_1d(obj.scale)

    # CreateLibraryMatrix probe
    probe_lab = np.asarray(d.meta["probe_ct_labels"])
    ct_df = pd.DataFrame({"cell": cells, "type": probe_lab})
    out["CreateLibraryMatrix"] = np.asarray(
        nde.create_library_matrix(counts, ct_df), dtype=float)

    t0 = time.time()
    obj = nde.calculate_effective_niche(obj, cutoff=0.05)
    out["t_en"] = time.time() - t0
    for k in range(len(sigma)):
        out[f"effective_niche_{k + 1}"] = obj.effective_niche[k]

    # large-scale variant on a fresh object
    obj_ls = nde.create_nichede_object(counts, coord, libmat, deconv,
                                       sigma=sigma, Int=True)
    obj_ls = nde.calculate_effective_niche_large_scale(obj_ls, batch_size=200,
                                                       cutoff=0.05, standardize=True)
    for k in range(len(sigma)):
        out[f"effective_niche_ls_{k + 1}"] = obj_ls.effective_niche[k]
    del obj_ls

    # Int = FALSE (linear-model) path
    cont_genes = list(d.meta["cont_genes"])
    cont_counts = pd.DataFrame(d["in_cont_counts"], index=cells, columns=cont_genes)
    cont_lib = pd.DataFrame(d["in_cont_libmat"], index=cts, columns=cont_genes)
    obj_c = nde.create_nichede_object(cont_counts, coord, cont_lib, deconv,
                                      sigma=sigma, Int=False)
    obj_c = nde.calculate_effective_niche(obj_c, cutoff=0.05)
    obj_c = nde.niche_DE(obj_c, num_cores=n_jobs, C=150, M=10, gamma=0.8,
                         Int=False, batch=True, self_EN=False, verbose=False)
    ngc = len(cont_genes)
    Tc = np.full((len(cts), len(cts), ngc), np.nan)
    vdc = np.zeros(ngc)
    for g, r in enumerate(obj_c.niche_DE[0]):
        if r["valid"] == 1:
            Tc[:, :, g] = r["T_stat"]
        vdc[g] = r["valid"]
    out["cont_T_stat_1"] = Tc
    out["cont_valid_1"] = vdc
    out["cont_pval_pos_gene"] = np.asarray(obj_c.niche_DE_pval_pos["gene_level"], dtype=float)
    del obj_c

    t0 = time.time()
    obj = nde.niche_DE(obj, num_cores=n_jobs, C=150, M=10, gamma=0.8,
                       Int=True, batch=True, self_EN=False, verbose=True)
    out["t_nde"] = time.time() - t0
    print(f"[cand] niche_DE {out['t_nde']:.1f}s")

    n_ct, ngene = len(cts), len(genes)
    for k in range(len(sigma)):
        T = np.full((n_ct, n_ct, ngene), np.nan)
        B = np.full((n_ct, n_ct, ngene), np.nan)
        ll = np.full(ngene, np.nan)
        vd = np.zeros(ngene)
        nn = np.zeros(ngene)
        for g, r in enumerate(obj.niche_DE[k]):
            if r["valid"] == 1:
                T[:, :, g] = r["T_stat"]
                B[:, :, g] = r["betas"]
            ll[g] = r["log_likelihood"]
            vd[g] = r["valid"]
            nn[g] = len(r["nulls"])
        out[f"T_stat_{k + 1}"] = T
        out[f"betas_{k + 1}"] = B
        out[f"loglik_{k + 1}"] = ll
        out[f"valid_{k + 1}"] = vd
        out[f"nnull_{k + 1}"] = nn

    # varcov of the first 50 valid genes at sigma[1]
    vc, vdim, vgene = [], [], []
    for g, r in enumerate(obj.niche_DE[0]):
        if r["valid"] == 1 and len(vgene) < 50:
            V = np.asarray(r["Varcov"], dtype=float)
            vgene.append(g + 1)
            vdim.append(V.shape[0])
            vc.append(V.ravel(order="F"))
    out["varcov_flat"] = np.concatenate(vc) if vc else np.zeros(0)
    out["varcov_dim"] = np.asarray(vdim)
    out["varcov_genes"] = np.asarray(vgene)

    nl = []
    for g in range(min(200, ngene)):
        nl.append(np.asarray(obj.niche_DE[0][g]["nulls"], dtype=float) + 1)  # -> 1-based
    out["nulls_flat"] = np.concatenate(nl) if nl else np.zeros(0)
    out["nulls_len"] = np.asarray([len(a) for a in nl])

    for tag, pv in (("pos", obj.niche_DE_pval_pos), ("neg", obj.niche_DE_pval_neg)):
        out[f"pval_{tag}_gene"] = np.asarray(pv["gene_level"], dtype=float)
        out[f"pval_{tag}_ct"] = np.asarray(pv["cell_type_level"], dtype=float)
        out[f"pval_{tag}_int"] = np.asarray(pv["interaction_level"], dtype=float)

    raw = nde.get_niche_DE_pval_raw.__wrapped__ if hasattr(
        nde.get_niche_DE_pval_raw, "__wrapped__") else None
    from pynichede.pvalues import _pval_core
    rawres = _pval_core(obj, pos=True, adjust=False, verbose=False)
    out["praw_pos_gene"] = np.asarray(rawres["gene_level"], dtype=float)
    out["praw_pos_ct"] = np.asarray(rawres["cell_type_level"], dtype=float)
    out["praw_pos_int"] = np.asarray(rawres["interaction_level"], dtype=float)

    # ---- downstream -------------------------------------------------------
    index_ct, niche_ct, niche_ct2 = "tumor_epithelial", "myeloid", "stromal"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lv in ("G", "CT", "I"):
            for pos in (True, False):
                try:
                    r = nde.get_niche_DE_genes(obj, lv, index=index_ct, niche=niche_ct,
                                               positive=pos, alpha=0.05)
                except Exception as e:                       # pragma: no cover
                    print("genes", lv, pos, "err", e)
                    r = pd.DataFrame()
                r.to_csv(os.path.join(ref_dir, f"cand_genes_{lv}_{'pos' if pos else 'neg'}.csv"),
                         index=False)
        try:
            mk = nde.niche_DE_markers(obj, index_ct, niche_ct, niche_ct2, alpha=0.05)
        except Exception as e:                               # pragma: no cover
            print("markers err", e)
            mk = pd.DataFrame()
        mk.to_csv(os.path.join(ref_dir, "cand_markers.csv"), index=False)

        ii = cts.index(index_ct); n1 = cts.index(niche_ct); n2 = cts.index(niche_ct2)
        res0 = obj.niche_DE[0]
        out["contrast_post"] = nde.contrast_post(
            [r["betas"] for r in res0], [r["Varcov"] for r in res0],
            [r["nulls"] for r in res0], ii, (n1, n2), n_type=n_ct)
        out["check_colloc"] = nde.check_colloc(obj, ii, n1)

    # ---- helper probes ----------------------------------------------------
    pT = d["probe_T"]
    out["T_to_p_two"] = nde.T_to_p(pT, "two.sided")
    out["T_to_p_pos"] = nde.T_to_p(pT, "positive")
    out["T_to_p_neg"] = nde.T_to_p(pT, "negative")
    out["ultosymmetric"] = nde.ultosymmetric(d["probe_M"])
    out["gene_level"] = np.atleast_1d(nde.gene_level(d["probe_p"], d["probe_w"]))
    out["celltype_level"] = nde.celltype_level(d["probe_pm"], d["probe_wc"])
    out["nb_lik"] = np.atleast_1d(nde.nb_lik([1, 5, 3, 0, 7], [2, 4, 3, 1, 6], 1.7))
    Rp = d["probe_poolr_R"]
    out["mvnconv_m2lp_s1"] = nde.mvnconv(Rp, 1, "m2lp")
    out["mvnconv_m2lp_s2"] = nde.mvnconv(Rp, 2, "m2lp")
    out["fisher_generalized"] = np.atleast_1d(
        nde.fisher_generalized(d["probe_poolr_p"], nde.mvnconv(Rp, 1, "m2lp")))

    # ---- merge / filter probes -------------------------------------------
    o1 = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=sigma, Int=True)
    coord2 = coord * 2
    o2 = nde.create_nichede_object(counts, coord2, libmat, deconv, sigma=sigma, Int=True)
    om = nde.merge_objects([o1, o2])
    out["merge_coord"] = np.asarray(om.coord, dtype=float)
    out["merge_num_cells"] = np.asarray(om.num_cells, dtype=float)
    out["merge_batch"] = np.asarray(om.batch_ID, dtype=float)

    keep = list(np.asarray(obj.cell_names)[::3])
    of = nde.filter_nde(obj, keep)
    out["filter_num_cells"] = np.asarray(of.num_cells, dtype=float)
    out["filter_en_1"] = of.effective_niche[0]

    # ---- niche-LR ---------------------------------------------------------
    if d.has("in_ligand_target_matrix"):
        ltm = pd.DataFrame(d["in_ligand_target_matrix"],
                           index=list(d.meta["ltm_rownames"]),
                           columns=list(d.meta["ltm_colnames"]))
        lr = pd.DataFrame({"ligand": list(d.meta["lr_ligand"]),
                           "receptor": list(d.meta["lr_receptor"])})
        for tag, res in (("spot", "spot"), ("cell", "cell")):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = nde.niche_LR_spot(obj, niche_ct, index_ct, ltm, lr) \
                        if res == "spot" else \
                        nde.niche_LR_cell(obj, niche_ct, index_ct, ltm, lr)
            except Exception as e:
                print(f"niche_LR_{tag} error: {e}")
                df = pd.DataFrame()
            df.to_csv(os.path.join(ref_dir, f"cand_niche_LR_{tag}.csv"), index=False)

    np.savez(os.path.join(ref_dir, "candidate.npz"), **out)
    print("wrote", os.path.join(ref_dir, "candidate.npz"))


if __name__ == "__main__":
    main()
