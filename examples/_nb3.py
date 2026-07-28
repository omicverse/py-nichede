"""Builder for ``examples/function_by_function_R_parity.ipynb`` (Notebook 3).

One subsection per **exported R symbol** in ``nichede-ref/NAMESPACE`` (25 ``export()``
entries plus the ``S3method(print, Niche_DE)``, 26 in total).  Each carries a full
parameter table, the R one-liner, the Python equivalent, and a numerical comparison
against the R dump.
"""

from __future__ import annotations

from _nb_common import FIXTURE_LOAD, PREAMBLE, code, md

# --------------------------------------------------------------------------- #
# Per-function specifications
#   name        : R symbol
#   py          : Python counterpart
#   blurb       : one paragraph
#   params      : list of (R name, Py name, type, default, range, description)
#   r_line      : the R one-liner (markdown only)
#   py_code     : the executed Python cell
# --------------------------------------------------------------------------- #

FUNCS = []


def F(name, py, blurb, params, r_line, py_code):
    FUNCS.append(dict(name=name, py=py, blurb=blurb, params=params,
                      r_line=r_line, py_code=py_code))


# ---- 1. print.Niche_DE ----------------------------------------------------- #
F("print.Niche_DE", "NicheDEObject.__repr__",
  "The S3 print method for the `Niche_DE` S4 class. Registered with "
  "`S3method(print, Niche_DE)` rather than `export()`, so it is reached through "
  "`print(obj)` (or `getS3method`). It returns — rather than cats — a one-line summary "
  "string. In Python the same string is produced by `NicheDEObject.__repr__`, so `print(obj)` "
  "and `repr(obj)` both work and `NicheDE.__repr__` forwards to it.",
  [("object", "self", "Niche_DE / NicheDEObject", "—", "a constructed object",
    "the object to summarise")],
  "print(obj)   # dispatches to nicheDE:::print.Niche_DE",
  r'''
r_str = pf.meta["ref_print_Niche_DE"]
py_str = repr(obj)
print("R      :", r_str)
print("Python :", py_str)
ok = (r_str == py_str)
print("identical string:", ok)
verdict("print.Niche_DE", "summary string", "deterministic", "exact string match",
        1.0 if ok else 0.0, ok)
''')

# ---- 2. CreateLibraryMatrix ------------------------------------------------ #
F("CreateLibraryMatrix", "pynichede.create_library_matrix",
  "Average expression profile per cell type: for each level of `cell_type[, 2]` it takes the "
  "column means of the corresponding rows of `data`, downsampling to 1000 cells first if the "
  "type is larger. Row order follows R's `unique()`, i.e. **first appearance**, not sorted.",
  [("data", "data", "matrix / dgCMatrix / DataFrame", "—", "cells × genes",
    "single-cell (or spot) counts. Python also accepts a `scipy.sparse` matrix or a "
    "`pandas.DataFrame`; row names must be present because both languages check them."),
   ("cell_type", "cell_type", "data.frame / DataFrame or 2-col array", "—",
    "column 1 = cell name, column 2 = cell type",
    "assignment table; column 1 must line up **row for row** with `data`."),
   ("—", "random_state", "int or None", "`None`", "any seed",
    "**new in Python.** R calls bare `sample()` for the >1000-cell downsampling, so its result "
    "depends on the ambient `.Random.seed`; the Python argument makes that reproducible. "
    "Not reached on this fixture (no cell type exceeds 1000).")],
  "L <- nicheDE::CreateLibraryMatrix(data, cell_type)",
  r'''
ct_df = pd.DataFrame({"cell": cells, "type": list(d.meta["probe_ct_labels"])})
L_py = nde.create_library_matrix(counts, ct_df)
L_R  = d["ref_CreateLibraryMatrix"]
print("R rownames :", list(d.meta["probe_ct_types"]))
print("Py index   :", list(L_py.index))
e = det(L_R, L_py.to_numpy())["max_abs_err"]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(L_R, L_py.to_numpy(), "CreateLibraryMatrix", "mean count", ax); plt.show()
verdict("CreateLibraryMatrix", "library matrix", "deterministic", "max abs err", e, e <= 1e-8)
''')

# ---- 3. CreateLibraryMatrixFromSeurat -------------------------------------- #
F("CreateLibraryMatrixFromSeurat", "pynichede.create_library_matrix_from_anndata",
  "The Seurat front end of `CreateLibraryMatrix`: it pulls `GetAssay(obj, assay)@counts`, "
  "transposes it to cells × genes, and takes the cell-type labels from `Idents(obj)`. The "
  "Python port takes an **AnnData** instead, because that is the equivalent Python object "
  "model — `Idents(obj)` maps to `adata.obs[celltype_key]` and the counts slot to "
  "`adata.layers[layer]` or `adata.X`. Note the R function reads the v3-only slot "
  "`sobj_assay@counts` and therefore **fails on Seurat ≥ 5 `Assay5` objects**; the reference "
  "below was produced with `options(Seurat.object.assay.version = 'v3')`.",
  [("seurat_object", "adata", "Seurat / AnnData", "—", "—",
    "**type changed.** Seurat object → `anndata.AnnData`."),
   ("assay", "layer", "character / str or None", "—", "an assay / layer name",
    "**renamed + semantics.** R selects a Seurat assay; Python selects "
    "`adata.layers[layer]`, or `adata.X` when `layer=None`."),
   ("—", "celltype_key", "str", "—", "a column of `adata.obs`",
    "**new in Python.** Replaces `Seurat::Idents(obj)`, which has no AnnData analogue — "
    "AnnData has no privileged identity column, so it must be named."),
   ("—", "random_state", "int or None", "`None`", "any seed",
    "**new in Python.** Same downsampling seed as `create_library_matrix`.")],
  'L <- nicheDE::CreateLibraryMatrixFromSeurat(seurat_object, assay = "RNA")',
  r'''
import anndata as ad
sub_genes = list(pf.meta["seurat_sub_genes"])
labels    = list(pf.meta["seurat_ct_labels"])
counts_pre = pd.DataFrame(d["in_counts"], index=cells, columns=list(pf.meta["counts_genes"]))
A = ad.AnnData(counts_pre[sub_genes].to_numpy())
A.obs_names = cells; A.var_names = sub_genes
A.obs["cell_type"] = pd.Categorical(labels, categories=pd.unique(labels))
L_seu_py = nde.create_library_matrix_from_anndata(A, "cell_type")
L_seu_R  = pf["ref_CreateLibraryMatrixFromSeurat"]
print("R (Seurat) :", L_seu_R.shape, list(pf.meta["seurat_lib_types"]))
print("Py(AnnData):", tuple(L_seu_py.shape), list(L_seu_py.index))
print("Seurat build note:", pf.meta["seurat_note"])
e = det(L_seu_R, L_seu_py.to_numpy())["max_abs_err"]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(L_seu_R, L_seu_py.to_numpy(), "CreateLibraryMatrixFromSeurat", "mean count", ax)
plt.show()
verdict("CreateLibraryMatrixFromSeurat", "library matrix", "deterministic",
        "max abs err", e, e <= 1e-8)
''')

# ---- 4. CreateNicheDEObject ------------------------------------------------ #
F("CreateNicheDEObject", "pynichede.create_nichede_object",
  "Builds the `Niche_DE` object: intersects the genes of the counts and library matrices "
  "(keeping the **library matrix's** ordering), converts deconvolution weights into expected "
  "cell counts per spot, and rescales the coordinates so the mean second-nearest-neighbour "
  "spot distance is 100 — which is why `sigma` is in those rescaled units.",
  [("counts_mat", "counts_mat", "matrix / DataFrame", "—", "spots × genes",
    "raw spatial counts; row and column names required and must be unique."),
   ("coordinate_mat", "coordinate_mat", "matrix / DataFrame", "—", "spots × 2",
    "spot coordinates; rescaled internally, `obj@scale` records the factor."),
   ("library_mat", "library_mat", "matrix / DataFrame", "—", "cell types × genes",
    "reference average expression profile per cell type."),
   ("deconv_mat", "deconv_mat", "matrix / DataFrame", "—", "spots × cell types",
    "deconvolution weights; column names must equal `rownames(library_mat)`."),
   ("sigma", "sigma", "numeric vector / sequence", "— (required in R)", "positive",
    "**default changed.** R has no default and errors if missing; Python defaults to `()`, "
    "which triggers the same fallback R uses internally, "
    "`[0.1*d, d, 2*d, 3*d]` with `d` = the rescaled spot distance."),
   ("Int", "Int", "logical / bool", "`TRUE` / `True`", "—",
    "`True` → counts must be integers, negative-binomial branch downstream; "
    "`False` → linear model with a gene-specific variance."),
   ("—", "random_state", "int or None", "`None`", "any seed",
    "**new in Python.** Only reached for > 10 000 spots, where R subsamples 1000 points to "
    "estimate the spot distance.")],
  "obj <- nicheDE::CreateNicheDEObject(counts_mat, coordinate_mat, library_mat,\n"
  "                                    deconv_mat, sigma = c(1, 100, 250), Int = TRUE)",
  r'''
obj = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=sigma, Int=True)
rows = {
    "num_cells": det(d["ref_num_cells"], np.asarray(obj.num_cells))["max_abs_err"],
    "coord (rescaled)": det(d["ref_coord"], np.asarray(obj.coord))["max_abs_err"],
    "ref_expr": det(d["ref_ref_expr"], np.asarray(obj.ref_expr))["max_abs_err"],
    "counts (reordered)": det(d["ref_counts"], np.asarray(obj.counts))["max_abs_err"],
    "scale": abs(float(np.atleast_1d(d.meta["ref_scale"])[0]) - float(obj.scale[0])),
}
for k, v in rows.items():
    print(f"  {k:22s} max abs err = {v:.3e}")
print("gene order identical to R:", list(obj.gene_names) == list(d.meta["gene_names"]))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(d["ref_num_cells"], np.asarray(obj.num_cells),
            "CreateNicheDEObject@num_cells", "expected cells", ax)
plt.show()
e = max(rows.values())
verdict("CreateNicheDEObject", "num_cells / coord / ref_expr / counts", "deterministic",
        "max abs err", e, e <= 1e-8)
''')

# ---- 5. CreateNicheDEObjectFromSeurat -------------------------------------- #
F("CreateNicheDEObjectFromSeurat", "pynichede.create_nichede_object_from_anndata",
  "Same constructor, Seurat front end: counts from `GetAssay(obj, assay)@counts` and "
  "coordinates from `seurat_object@images[[1]]@coordinates$imagerow / $imagecol`. The Python "
  "port reads `adata.X` (or a layer) and `adata.obsm[spatial_key]`, then delegates to "
  "`create_nichede_object`, so everything after the extraction step is shared code.",
  [("seurat_object", "adata", "Seurat / AnnData", "—", "—",
    "**type changed**, as for `CreateLibraryMatrixFromSeurat`."),
   ("assay", "layer", "character / str or None", "—", "assay / layer name",
    "**renamed + semantics**; `None` means `adata.X`."),
   ("library_mat", "library_mat", "matrix / DataFrame", "—", "cell types × genes", "as above."),
   ("deconv_mat", "deconv_mat", "matrix / DataFrame", "—", "spots × cell types", "as above."),
   ("sigma", "sigma", "numeric vector / sequence", "— / `()`", "positive",
    "**default changed**, as for `CreateNicheDEObject`."),
   ("Int", "Int", "logical / bool", "`TRUE` / `True`", "—", "as above."),
   ("—", "spatial_key", "str", "`'spatial'`", "a key of `adata.obsm`",
    "**new in Python.** Replaces R's hard-coded `@images[[1]]@coordinates`; AnnData stores "
    "coordinates under a user-chosen `obsm` key."),
   ("—", "random_state", "int or None", "`None`", "any seed", "as above.")],
  'obj <- nicheDE::CreateNicheDEObjectFromSeurat(seurat_object, "Spatial",\n'
  "                                              library_mat, deconv_mat, sigma, Int = TRUE)",
  r'''
A2 = ad.AnnData(np.asarray(counts, dtype=np.float64))
A2.obs_names = cells; A2.var_names = genes
A2.obsm["spatial"] = np.asarray(coord, dtype=np.float64)
obj_seu = nde.create_nichede_object_from_anndata(A2, libmat, deconv, sigma=sigma,
                                                 spatial_key="spatial", Int=True)
e_nc = det(pf["ref_seurat_num_cells"], np.asarray(obj_seu.num_cells))["max_abs_err"]
e_co = det(pf["ref_seurat_coord"], np.asarray(obj_seu.coord))["max_abs_err"]
print(f"num_cells  max abs err vs R(Seurat) = {e_nc:.3e}")
print(f"coord      max abs err vs R(Seurat) = {e_co:.3e}")
print("scale  R =", float(np.atleast_1d(pf.meta["ref_seurat_scale"])[0]),
      " Python =", float(obj_seu.scale[0]))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(pf["ref_seurat_num_cells"], np.asarray(obj_seu.num_cells),
            "CreateNicheDEObjectFromSeurat@num_cells", "expected cells", ax)
plt.show()
e = max(e_nc, e_co)
verdict("CreateNicheDEObjectFromSeurat", "num_cells / coord", "deterministic",
        "max abs err", e, e <= 1e-8)
del A2, obj_seu
''')

# ---- 6. MergeObjects ------------------------------------------------------- #
F("MergeObjects", "pynichede.merge_objects",
  "Concatenates several objects into one, dividing each subsequent object's coordinates by "
  "`(100/scale_j)/(100/scale_1)` so one kernel bandwidth means the same physical distance in "
  "every batch, and assigning `batch_ID`. Validates that all objects share kernel bandwidths, "
  "`Int`, reference expression, cell types and gene order.",
  [("objects", "objects", "list of Niche_DE / list[NicheDEObject]", "—", "length ≥ 1",
    "the objects to merge; the first one defines the coordinate scale and gene order. Python "
    "also accepts a tuple, and `NicheDE.merge` accepts wrapped class instances.")],
  "om <- nicheDE::MergeObjects(list(o1, o2))",
  r'''
o1 = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=sigma, Int=True)
o2 = nde.create_nichede_object(counts, coord * 2, libmat, deconv, sigma=sigma, Int=True)
om = nde.merge_objects([o1, o2])
e_c = det(d["ref_merge_coord"], np.asarray(om.coord))["max_abs_err"]
e_n = det(d["ref_merge_num_cells"], np.asarray(om.num_cells))["max_abs_err"]
e_b = det(d["ref_merge_batch"], np.asarray(om.batch_ID))["max_abs_err"]
print(f"coord      max abs err = {e_c:.3e}")
print(f"num_cells  max abs err = {e_n:.3e}")
print(f"batch_ID   max abs err = {e_b:.3e}")
print("R merge scale:", d.meta["merge_scale"], " Python:", list(om.scale))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(d["ref_merge_coord"], np.asarray(om.coord), "MergeObjects@coord", "coordinate", ax)
plt.show()
e = max(e_c, e_n, e_b)
verdict("MergeObjects", "coord / num_cells / batch_ID", "deterministic",
        "max abs err", e, e <= 1e-8)
del o1, o2, om
''')

# ---- 7. Filter_NDE --------------------------------------------------------- #
F("Filter_NDE", "pynichede.filter_nde",
  "Subsets an object to the named observations. Because R implements it as a membership test "
  "(`obj@cell_names %in% cell_names`) the surviving rows keep their **original** order rather "
  "than the order of `cell_names`; the port reproduces that.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—", "the object to subset."),
   ("cell_names", "cell_names", "character vector / sequence[str]", "—",
    "subset of `obj@cell_names`", "names to keep; both languages error if any name is absent.")],
  "of <- nicheDE::Filter_NDE(obj, keep_cells)",
  r'''
obj = nde.calculate_effective_niche(obj, cutoff=0.05)
keep = list(np.asarray(obj.cell_names)[::3])
of = nde.filter_nde(obj, keep)
e_n = det(d["ref_filter_num_cells"], np.asarray(of.num_cells))["max_abs_err"]
e_e = det(d["ref_filter_en_1"], of.effective_niche[0])["max_abs_err"]
print("kept", len(of.cell_names), "of", len(obj.cell_names), "spots "
      f"(R kept {len(list(d.meta['filter_cells']))})")
print(f"num_cells        max abs err = {e_n:.3e}")
print(f"effective_niche  max abs err = {e_e:.3e}")
print("row order preserved:", list(of.cell_names) == list(d.meta["filter_cells"]))
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
overlay_det(d["ref_filter_en_1"], of.effective_niche[0],
            "Filter_NDE@effective_niche[1]", "z-score", ax)
plt.show()
e = max(e_n, e_e)
verdict("Filter_NDE", "num_cells / effective_niche", "deterministic", "max abs err", e, e <= 1e-8)
del of
''')

# ---- 8. CalculateEffectiveNiche -------------------------------------------- #
F("CalculateEffectiveNiche", "pynichede.calculate_effective_niche",
  "For each kernel bandwidth: `EN = K %*% num_cells` with `K = exp(-d^2 / sigma^2)` zeroed "
  "below `cutoff`, computed per batch, rescaled to batch 1's mean cell count, then column "
  "z-scored with R's `sd` (denominator `n-1`) and `NaN`s set to 0.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—",
    "must already carry `coord`, `num_cells`, `sigma` and `batch_ID`."),
   ("cutoff", "cutoff", "numeric / float", "`0.05`", "`(0, 1]`",
    "kernel weights below this are set to 0. `1` keeps only self-distance; small values "
    "approach an untruncated Gaussian.")],
  "obj <- nicheDE::CalculateEffectiveNiche(obj, cutoff = 0.05)",
  r'''
errs = [det(d[f"ref_effective_niche_{k+1}"], obj.effective_niche[k])["max_abs_err"]
        for k in range(len(sigma))]
for k, e_ in enumerate(errs):
    print(f"  sigma = {sigma[k]:6g}   max abs err = {e_:.3e}")
fig, ax = plt.subplots(len(sigma), 2, figsize=(9.5, 3.0 * len(sigma)))
for k in range(len(sigma)):
    overlay_det(d[f"ref_effective_niche_{k+1}"], obj.effective_niche[k],
                f"CalculateEffectiveNiche  sigma = {sigma[k]:g}", "z-score", ax[k])
plt.show()
e = max(errs)
verdict("CalculateEffectiveNiche", "effective_niche (all sigma)", "deterministic",
        "max abs err", e, e <= 1e-8)
''')

# ---- 9. CalculateEffectiveNicheLargeScale ---------------------------------- #
F("CalculateEffectiveNicheLargeScale", "pynichede.calculate_effective_niche_large_scale",
  "The tiled, bounded-memory version. It splits the tissue into a `ceil(sqrt(n_batches))` "
  "grid and, per tile, only considers candidate neighbours inside a bounding box padded by "
  "`sigma * sqrt(-log(cutoff))` — the exact radius past which the Gaussian falls below "
  "`cutoff`, so the tiling is exact, not approximate. **The shipped R implementation is "
  "broken** with `Rfast >= 2.1.5.2`: `Rfast::dista()` returns an all-zero matrix for ≥ 4 "
  "query rows, so every kernel weight collapses to 1. The R driver therefore also dumps a "
  "repaired run (`ref_effective_niche_lsfix_*`) with base-R `dist()` substituted, and that is "
  "the anchor the port is graded against.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—", "the object."),
   ("batch_size", "batch_size", "integer / int", "`1000`", "`≥ 1`",
    "target spots per tile; the grid side is `ceil(sqrt(ceil(n_spot / batch_size)))`. Affects "
    "memory and speed only — the result is invariant."),
   ("cutoff", "cutoff", "numeric / float", "`0.05`", "`(0, 1]`",
    "as in `CalculateEffectiveNiche`; also sets the bounding-box padding."),
   ("standardize", "standardize", "logical / bool", "`TRUE` / `True`", "—",
    "column z-score the result. `CalculateEffectiveNiche` always standardises, so leave it "
    "`True` for the two to agree.")],
  "obj <- nicheDE::CalculateEffectiveNicheLargeScale(obj, batch_size = 200,\n"
  "                                                  cutoff = 0.05, standardize = TRUE)",
  r'''
o_ls = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=sigma, Int=True)
o_ls = nde.calculate_effective_niche_large_scale(o_ls, batch_size=200, cutoff=0.05,
                                                 standardize=True)
tab = pd.DataFrame([dict(
    sigma=f"{sigma[k]:g}",
    vs_repaired_R=det(d[f"ref_effective_niche_lsfix_{k+1}"], o_ls.effective_niche[k])["max_abs_err"],
    vs_shipped_R=det(d[f"ref_effective_niche_ls_{k+1}"], o_ls.effective_niche[k])["max_abs_err"],
    vs_exact_routine=det(obj.effective_niche[k], o_ls.effective_niche[k])["max_abs_err"],
) for k in range(len(sigma))])
display(tab)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
x = np.arange(len(sigma))
ax[0].bar(x - 0.2, tab["vs_repaired_R"], 0.4, color=C_PY, label="vs REPAIRED R")
ax[0].bar(x + 0.2, tab["vs_shipped_R"], 0.4, color=C_BAD, label="vs SHIPPED R (Rfast bug)")
ax[0].axhline(1e-8, ls="--", color="k", lw=1, label="gate 1e-8")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(tab["sigma"])
ax[0].set_xlabel("sigma"); ax[0].set_ylabel("max abs err"); ax[0].legend(fontsize=7)
overlay_det(d["ref_effective_niche_lsfix_1"], o_ls.effective_niche[0],
            "vs repaired R, sigma = 1", "z-score", [ax[1], ax[1].twinx()])
plt.show()
e = float(tab["vs_repaired_R"].max())
verdict("CalculateEffectiveNicheLargeScale", "effective_niche vs REPAIRED R",
        "deterministic", "max abs err", e, e <= 1e-8)
verdict("CalculateEffectiveNicheLargeScale", "effective_niche vs SHIPPED R",
        "documented divergence", "max abs err", float(tab["vs_shipped_R"].max()), None)
del o_ls
''')

# ---- 10. niche_DE ---------------------------------------------------------- #
F("niche_DE", "pynichede.niche_DE",
  "The main regression. Per gene and per kernel bandwidth it fits "
  "`counts ~ X + batch + offset(log(EEJ))` as a Poisson GLM (R's `glm.fit`, LINPACK `dqrdc2` "
  "limited-pivot rank detection included), estimates over-dispersion by Brent minimisation of "
  "`nb_lik`, and forms Wald statistics from `(X'WX)^-1`. It then fills both p-value tables via "
  "`get_niche_DE_pval_fisher`.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—",
    "must already have an effective niche."),
   ("num_cores", "num_cores", "integer / int", "— / `1`", "`≥ 1`",
    "**default added in Python.** R requires it; Python defaults to serial. Maps `parallel`/"
    "`foreach` workers to `joblib` workers. Genes are dispatched in chunks (≈ 4 per worker) "
    "rather than one task each — see `ITERATION_LOG.md` iters 8–9."),
   ("outfile", "outfile", "character / any", "— / `None`", "path or `''`",
    "**no-op in Python.** R passes it to `parallel::makeCluster(outfile=)` to redirect worker "
    "output. Accepted for signature compatibility."),
   ("C", "C", "numeric / float", "`150`", "`≥ 0`",
    "minimum total count for a gene to be fitted at all."),
   ("M", "M", "integer / int", "`10`", "`≥ 1`",
    "minimum number of spots with a non-zero entry for an interaction column to be kept; "
    "columns below it join `nulls`."),
   ("gamma", "gamma", "numeric / float", "`0.8`", "`(0, 1)`",
    "quantile of each cell type's reference profile below which that cell type is considered "
    "not to express the gene (R's type-7 `quantile`, reimplemented exactly)."),
   ("print", "print_", "logical / bool", "`TRUE` / `True`", "—",
    "**renamed** (`print` is a Python builtin). Kept as a no-op alias; use `verbose`."),
   ("Int", "Int", "logical / bool", "`TRUE` / `True`", "—",
    "negative-binomial branch vs linear model with gene-specific variance."),
   ("batch", "batch", "logical / bool", "`TRUE` / `True`", "—",
    "include batch dummies. With a single batch the constant column is aliased against the "
    "intercept in both languages, so it contributes one `NA` coefficient and nothing else."),
   ("self_EN", "self_EN", "logical / bool", "`FALSE` / `False`", "—",
    "keep the `(a, a)` self-interaction columns. `False` zeroes the diagonal."),
   ("G", "G", "numeric / float", "`1`", "`≥ 1`",
    "**no-op in Python.** R splits the counts matrix into `G` chunks to bound worker memory; "
    "`joblib` shares the array instead, so no chunking is needed."),
   ("—", "verbose", "bool", "`True`", "—",
    "**new in Python.** Controls the progress messages that R emits unconditionally through "
    "`print`.")],
  "obj <- nicheDE::niche_DE(obj, num_cores = 16, outfile = '', C = 150, M = 10,\n"
  "                         gamma = 0.8, print = TRUE, Int = TRUE, batch = TRUE,\n"
  "                         self_EN = FALSE, G = 1)",
  r'''
t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    obj = nde.niche_DE(obj, num_cores=N_JOBS, C=150, M=10, gamma=0.8,
                       Int=True, batch=True, self_EN=False, verbose=False)
PY_T_NDE = time.perf_counter() - t0
R_T_NDE = float(np.atleast_1d(d.meta["time_niche_DE"])[0])
print(f"Python {PY_T_NDE:.1f} s  |  R {R_T_NDE:.1f} s  ->  {R_T_NDE / PY_T_NDE:.1f}x")

n_ct, ngene = len(cts), len(genes)
T_py, B_py, valid_py, nn_py, ll_py = {}, {}, {}, {}, {}
for k in range(len(sigma)):
    T = np.full((n_ct, n_ct, ngene), np.nan); B = np.full((n_ct, n_ct, ngene), np.nan)
    vd = np.zeros(ngene); nn = np.zeros(ngene); ll = np.full(ngene, np.nan)
    for g, r in enumerate(obj.niche_DE[k]):
        if r["valid"] == 1:
            T[:, :, g] = r["T_stat"]; B[:, :, g] = r["betas"]
        vd[g] = r["valid"]; nn[g] = len(r["nulls"]); ll[g] = r["log_likelihood"]
    T_py[k], B_py[k], valid_py[k], nn_py[k], ll_py[k] = T, B, vd, nn, ll

fig, ax = plt.subplots(1, len(sigma), figsize=(4.2 * len(sigma), 3.8))
sts = [scatter_corr(d[f"ref_T_stat_{k+1}"], T_py[k], f"T_stat sigma = {sigma[k]:g}", ax[k])
       for k in range(len(sigma))]
plt.show()
agr = [float((d[f"ref_valid_{k+1}"] == valid_py[k]).mean()) for k in range(len(sigma))]
nul = [float(np.max(np.abs(d[f"ref_nnull_{k+1}"] - nn_py[k]))) for k in range(len(sigma))]
bet = [corr(d[f"ref_betas_{k+1}"], B_py[k])["pearson"] for k in range(len(sigma))]
llc = [corr(d[f"ref_loglik_{k+1}"], ll_py[k])["pearson"] for k in range(len(sigma))]
print("valid-flag agreement :", agr)
print("nulls max abs err    :", nul)
p_min = min(s["pearson"] for s in sts)
verdict("niche_DE", "T_stat", "ordinal (pearson)", "Pearson", p_min, p_min >= 0.99)
verdict("niche_DE", "T_stat", "ordinal (spearman)", "Spearman",
        min(s["spearman"] for s in sts), min(s["spearman"] for s in sts) >= 0.99)
verdict("niche_DE", "betas", "ordinal (pearson)", "Pearson", min(bet), min(bet) >= 0.99)
verdict("niche_DE", "log_likelihood", "ordinal (pearson)", "Pearson", min(llc), min(llc) >= 0.99)
verdict("niche_DE", "valid flags", "classification", "agreement", min(agr), min(agr) >= 1.0)
verdict("niche_DE", "nulls sets", "deterministic", "max abs err", max(nul), max(nul) <= 1e-8)
''')

# ---- 11. niche_DE_no_parallel ---------------------------------------------- #
F("niche_DE_no_parallel", "pynichede.niche_DE_no_parallel",
  "The single-threaded entry point: identical mathematics, no cluster. In R it is a separate "
  "1571-line-file function with the `foreach` loop replaced by a plain `for`; in Python it "
  "delegates to `niche_DE(..., num_cores=1)` because the parallel path is already a pure "
  "partition of the same gene loop. The R reference for this function is the `Int = FALSE` "
  "run the driver performs with it, so the comparison below doubles as the linear-model "
  "branch's parity check.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—", "as `niche_DE`."),
   ("C", "C", "numeric / float", "`150`", "`≥ 0`", "as `niche_DE`."),
   ("M", "M", "integer / int", "`10`", "`≥ 1`", "as `niche_DE`."),
   ("gamma", "gamma", "numeric / float", "`0.8`", "`(0, 1)`", "as `niche_DE`."),
   ("print", "print_", "logical / bool", "`TRUE` / `True`", "—", "**renamed**, see `niche_DE`."),
   ("Int", "Int", "logical / bool", "`TRUE` / `True`", "—", "as `niche_DE`."),
   ("batch", "batch", "logical / bool", "`TRUE` / `True`", "—", "as `niche_DE`."),
   ("self_EN", "self_EN", "logical / bool", "`FALSE` / `False`", "—", "as `niche_DE`."),
   ("—", "verbose", "bool", "`True`", "—", "**new in Python**, as `niche_DE`.")],
  "obj_c <- nicheDE::niche_DE_no_parallel(obj_c, C = 150, M = 10, gamma = 0.8,\n"
  "                                       print = FALSE, Int = FALSE, batch = TRUE,\n"
  "                                       self_EN = FALSE)",
  r'''
cont_genes = list(d.meta["cont_genes"])
cont_counts = pd.DataFrame(d["in_cont_counts"], index=cells, columns=cont_genes)
cont_lib    = pd.DataFrame(d["in_cont_libmat"], index=cts,   columns=cont_genes)
o_c = nde.create_nichede_object(cont_counts, coord, cont_lib, deconv, sigma=sigma, Int=False)
o_c = nde.calculate_effective_niche(o_c, cutoff=0.05)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o_c = nde.niche_DE_no_parallel(o_c, C=150, M=10, gamma=0.8, Int=False,
                                   batch=True, self_EN=False, verbose=False)
Tc = np.full((len(cts), len(cts), len(cont_genes)), np.nan)
vc = np.zeros(len(cont_genes))
for g, r in enumerate(o_c.niche_DE[0]):
    if r["valid"] == 1:
        Tc[:, :, g] = r["T_stat"]
    vc[g] = r["valid"]
st = corr(d["ref_cont_T_stat_1"], Tc)
agree = float((d["ref_cont_valid_1"] == vc).mean())
print(f"Int=FALSE valid genes: R {int(d['ref_cont_valid_1'].sum())} / "
      f"Python {int(vc.sum())} of {len(cont_genes)}   agreement {agree:.6f}")
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
scatter_corr(d["ref_cont_T_stat_1"], Tc, "niche_DE_no_parallel, Int=FALSE T_stat", ax[0])
st_p = scatter_pval(d["ref_cont_pval_pos_gene"],
                    np.asarray(o_c.niche_DE_pval_pos["gene_level"], dtype=float),
                    "Int=FALSE gene-level p", [ax[1], ax[1].twinx()])
plt.show()
verdict("niche_DE_no_parallel", "Int=FALSE T_stat", "ordinal (pearson)", "Pearson",
        st["pearson"], st["pearson"] >= 0.99)
verdict("niche_DE_no_parallel", "Int=FALSE valid flags", "classification", "agreement",
        agree, agree >= 1.0)
verdict("niche_DE_no_parallel", "Int=FALSE gene-level p", "inference", "Spearman(-log10 p)",
        st_p["spearman_neglog10p"], st_p["spearman_neglog10p"] >= 0.90)
del o_c
''')

# ---- 12. get_niche_DE_pval_fisher ------------------------------------------ #
F("get_niche_DE_pval_fisher", "pynichede.get_niche_DE_pval_fisher",
  "Turns the Wald statistics into the three nested p-value tables **with** Benjamini–Hochberg "
  "adjustment: interaction level (raw Wald p, Cauchy-combined across bandwidths, BH across "
  "niche cell types), cell-type level (Brown within one index cell type) and gene level "
  "(Brown over all `n_celltype^2`). `niche_DE` calls it for you in both directions.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—", "must carry fitted `niche_DE` results."),
   ("pos", "pos", "logical / bool", "`TRUE` / `True`", "—",
    "`True` tests for up-regulation near the niche type and writes "
    "`obj@niche_DE_pval_pos`; `False` tests down-regulation and writes `..._neg`."),
   ("—", "verbose", "bool", "`True`", "—",
    "**new in Python.** R prints its four progress lines unconditionally.")],
  "obj <- nicheDE::get_niche_DE_pval_fisher(obj, pos = TRUE)",
  r'''
res = []
for tag, pv in (("pos", obj.niche_DE_pval_pos), ("neg", obj.niche_DE_pval_neg)):
    for lvl, key in (("gene", "gene_level"), ("ct", "cell_type_level"), ("int", "interaction_level")):
        st = infer(d[f"ref_pval_{tag}_{lvl}"], np.asarray(pv[key], dtype=float))
        res.append(dict(direction=tag, level=key, spearman=st["spearman_neglog10p"],
                        top50_jaccard=st["top50_jaccard"], n=st["n"]))
PV = pd.DataFrame(res); display(PV)
fig, ax = plt.subplots(2, 2, figsize=(9.0, 6.4))
scatter_pval(d["ref_pval_pos_gene"], np.asarray(obj.niche_DE_pval_pos["gene_level"], float),
             "pos, gene level", ax[0])
scatter_pval(d["ref_pval_neg_ct"], np.asarray(obj.niche_DE_pval_neg["cell_type_level"], float),
             "neg, cell-type level", ax[1])
plt.show()
verdict("get_niche_DE_pval_fisher", "all 6 tables", "inference", "min Spearman(-log10 p)",
        float(PV["spearman"].min()), float(PV["spearman"].min()) >= 0.90)
verdict("get_niche_DE_pval_fisher", "all 6 tables", "ranked", "min top-50 Jaccard",
        float(PV["top50_jaccard"].min()), float(PV["top50_jaccard"].min()) >= 0.70)
''')

# ---- 13. get_niche_DE_pval_raw --------------------------------------------- #
F("get_niche_DE_pval_raw", "pynichede.get_niche_DE_pval_raw",
  "The same three tables **without** the BH step, for users who want their own multiple-testing "
  "correction. Both languages overwrite the object's p-value slots, so call it on a copy; "
  "`pynichede` additionally offers `NicheDE.pval_raw()`, which returns the dict without "
  "mutating anything.",
  [("object", "obj", "Niche_DE / NicheDEObject", "—", "—", "as above."),
   ("pos", "pos", "logical / bool", "`TRUE` / `True`", "—", "as above."),
   ("—", "verbose", "bool", "`True`", "—", "**new in Python**, as above.")],
  "obj_raw <- nicheDE::get_niche_DE_pval_raw(obj, pos = TRUE)",
  r'''
from pynichede.pvalues import _pval_core
raw = _pval_core(obj, pos=True, adjust=False, verbose=False)
res = []
for lvl, key in (("gene", "gene_level"), ("ct", "cell_type_level"), ("int", "interaction_level")):
    st = infer(d[f"ref_praw_pos_{lvl}"], np.asarray(raw[key], dtype=float))
    res.append(dict(level=key, spearman=st["spearman_neglog10p"],
                    top50_jaccard=st["top50_jaccard"], n=st["n"]))
RAW = pd.DataFrame(res); display(RAW)
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.3))
scatter_pval(d["ref_praw_pos_gene"], np.asarray(raw["gene_level"], dtype=float),
             "raw (un-BH) gene level", ax)
plt.show()
# BH itself is bit-exact: feed R's own raw p-values through the Python p_adjust
from pynichede.rstats import p_adjust
bh_from_R = p_adjust(d["ref_praw_pos_gene"], "BH")
e_bh = det(d["ref_pval_pos_gene"], bh_from_R)["max_abs_err"]
print(f"p_adjust(R's own raw p, 'BH') vs R's adjusted p:  max abs err = {e_bh:.3e}")
verdict("get_niche_DE_pval_raw", "raw p-values", "inference", "min Spearman(-log10 p)",
        float(RAW["spearman"].min()), float(RAW["spearman"].min()) >= 0.90)
verdict("get_niche_DE_pval_raw", "BH step alone (R raw p in)", "deterministic",
        "max abs err", e_bh, e_bh <= 1e-8)
''')
