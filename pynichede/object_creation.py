"""Niche-DE object construction and effective-niche calculation.

Mirrors ``R/niche_DE_object_creation.R`` one function at a time:

===========================================  =========================================
R                                            Python
===========================================  =========================================
``CreateLibraryMatrix``                      :func:`create_library_matrix`
``CreateLibraryMatrixFromSeurat``            :func:`create_library_matrix_from_anndata`
``CreateNicheDEObject``                      :func:`create_nichede_object`
``CreateNicheDEObjectFromSeurat``            :func:`create_nichede_object_from_anndata`
``MergeObjects``                             :func:`merge_objects`
``Filter_NDE``                               :func:`filter_nde`
``CalculateEffectiveNiche``                  :func:`calculate_effective_niche`
``CalculateEffectiveNicheLargeScale``        :func:`calculate_effective_niche_large_scale`
``print.Niche_DE``                           ``NicheDEObject.__repr__``
===========================================  =========================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = [
    "NicheDEObject",
    "create_library_matrix",
    "create_library_matrix_from_anndata",
    "create_nichede_object",
    "create_nichede_object_from_anndata",
    "merge_objects",
    "filter_nde",
    "calculate_effective_niche",
    "calculate_effective_niche_large_scale",
]


def _as_dense(x) -> np.ndarray:
    if issparse(x):
        return np.asarray(x.todense(), dtype=np.float64)
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _names(x, axis: int):
    if isinstance(x, pd.DataFrame):
        return list(x.index if axis == 0 else x.columns)
    return None


class NicheDEObject:
    """Python mirror of the S4 ``Niche_DE`` class.

    Slot names follow the R object so that a reader of the R source can find
    them without translation: ``counts, coord, sigma, num_cells,
    effective_niche, ref_expr, cell_names, cell_types, gene_names, batch_ID,
    spot_distance, niche_DE, niche_DE_pval_pos, niche_DE_pval_neg, scale, Int``.
    """

    def __init__(self, counts, coord, sigma, num_cells, ref_expr, cell_names,
                 cell_types, gene_names, batch_ID, spot_distance, scale, Int=True):
        self.counts = counts
        self.coord = coord
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.num_cells = num_cells
        self.effective_niche: list[np.ndarray] = []
        self.ref_expr = ref_expr
        self.cell_names = list(cell_names)
        self.cell_types = list(cell_types)
        self.gene_names = list(gene_names)
        self.batch_ID = np.asarray(batch_ID)
        self.spot_distance = spot_distance
        self.niche_DE: list = []
        self.niche_DE_pval_pos: dict = {}
        self.niche_DE_pval_neg: dict = {}
        self.scale = np.atleast_1d(np.asarray(scale, dtype=np.float64))
        self.Int = bool(Int)

    def __repr__(self) -> str:
        return (f"Niche-DE object with {self.counts.shape[0]} observations, "
                f"{self.counts.shape[1]} genes, "
                f"{len(np.unique(self.batch_ID))} batch(es), and "
                f"{len(self.cell_types)} cell types.")


# --------------------------------------------------------------------------- #
# CreateLibraryMatrix
# --------------------------------------------------------------------------- #

def create_library_matrix(data, cell_type, random_state: int | None = None):
    """``nicheDE::CreateLibraryMatrix``.

    Average expression profile per cell type.  ``cell_type`` is a 2-column
    frame (cell name, cell type) whose first column must line up with the rows
    of ``data``, exactly as R requires.  Cell types with more than 1000 cells
    are downsampled to 1000 (the only RNG in the whole package).
    """
    if isinstance(cell_type, pd.DataFrame):
        ct_names = cell_type.iloc[:, 0].to_numpy()
        ct_lab = cell_type.iloc[:, 1].to_numpy()
    else:
        ct = np.asarray(cell_type, dtype=object)
        ct_names, ct_lab = ct[:, 0], ct[:, 1]

    rn = _names(data, 0)
    if rn is not None and np.mean(np.asarray(rn) == ct_names) != 1:
        raise ValueError("Data rownames and Cell type matrix names do not match")

    X = _as_dense(data)
    gene_names = _names(data, 1)

    # R's unique() keeps first-appearance order
    CT = list(pd.unique(ct_lab))
    rng = np.random.default_rng(random_state)
    L = np.full((len(CT), X.shape[1]), np.nan)
    for j, ct_j in enumerate(CT):
        cells = np.flatnonzero(ct_lab == ct_j)
        if cells.size > 1000:
            cells = rng.choice(cells, 1000, replace=False)
        L[j] = X[cells].mean(axis=0)
    return pd.DataFrame(L, index=CT, columns=gene_names)


def create_library_matrix_from_anndata(adata, celltype_key: str, layer=None,
                                       random_state: int | None = None):
    """``CreateLibraryMatrixFromSeurat``, with AnnData standing in for Seurat.

    ``Seurat::Idents(obj)`` maps to ``adata.obs[celltype_key]``; the counts slot
    of the requested assay maps to ``adata.layers[layer]`` (or ``adata.X``).
    """
    X = adata.layers[layer] if layer is not None else adata.X
    df = pd.DataFrame(_as_dense(X), index=list(adata.obs_names),
                      columns=list(adata.var_names))
    ct = pd.DataFrame({"cell": list(adata.obs_names),
                       "type": adata.obs[celltype_key].astype(str).to_numpy()})
    return create_library_matrix(df, ct, random_state=random_state)


# --------------------------------------------------------------------------- #
# CreateNicheDEObject
# --------------------------------------------------------------------------- #

def _min_spot_distance(coord: np.ndarray, rng) -> float:
    """R's spot-distance estimator.

    For < 10 000 spots R builds the full distance matrix and takes
    ``mean_j( sort(D[, j])[3] )`` — index 3 of the *sorted* column, whose first
    entry is the zero self-distance, i.e. the mean **second**-nearest-neighbour
    distance.  Above 10 000 spots it subsamples 1000 points and takes the
    median instead.
    """
    n = coord.shape[0]
    if n < 1e4:
        D = squareform(pdist(coord))
        Ds = np.sort(D, axis=0)
        return float(np.mean(Ds[2, :]))
    inds = rng.choice(n, 1000, replace=False)
    D = cdist(coord[inds], coord)          # Rfast::dista(xnew, x, trans = TRUE)
    out = np.empty(D.shape[0])
    for i in range(D.shape[0]):
        row = np.sort(D[i][D[i] > 0])
        out[i] = row[2]
    return float(np.median(out))


def create_nichede_object(counts_mat, coordinate_mat, library_mat, deconv_mat,
                          sigma=(), Int: bool = True, random_state: int | None = None):
    """``nicheDE::CreateNicheDEObject``.

    Parameters mirror R exactly.  ``counts_mat`` is spots x genes,
    ``coordinate_mat`` spots x 2, ``library_mat`` cell types x genes,
    ``deconv_mat`` spots x cell types.  Coordinates are rescaled so that the
    mean second-nearest-neighbour distance becomes 100.
    """
    rng = np.random.default_rng(random_state)

    counts_names = _names(counts_mat, 1)
    spot_names = _names(counts_mat, 0)
    lib_genes = _names(library_mat, 1)
    lib_types = _names(library_mat, 0)
    dec_types = _names(deconv_mat, 1)

    counts = _as_dense(counts_mat)
    coord = _as_dense(coordinate_mat)
    L = _as_dense(library_mat)
    P = _as_dense(deconv_mat)

    if Int and not np.all(counts % 1 == 0):
        raise ValueError("counts matrix must contain only integers")
    if spot_names is None or counts_names is None:
        raise ValueError("cell/spot names and gene names of counts matrix must be non-null")
    if len(set(spot_names)) != len(spot_names):
        raise ValueError("cell/spot names (rownames) of counts matrix must be unique")
    if len(set(counts_names)) != len(counts_names):
        raise ValueError("gene names (colnames) of counts matrix must be unique")
    if _names(coordinate_mat, 0) is not None and \
            np.mean(np.asarray(spot_names) == np.asarray(_names(coordinate_mat, 0))) != 1:
        raise ValueError("cell/spot names of counts matrix and coordinate matrix do not match")
    if _names(deconv_mat, 0) is not None and \
            np.mean(np.asarray(spot_names) == np.asarray(_names(deconv_mat, 0))) != 1:
        raise ValueError("cell/spot names of counts matrix and deconvolution matrix do not match")
    if dec_types is not None and lib_types is not None and \
            np.mean(np.asarray(dec_types) == np.asarray(lib_types)) != 1:
        raise ValueError("celltypes of deconvolution matrix and reference expression matrix do not match")

    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # gene intersection, keeping the library matrix's ordering (as R does)
    counts_set = set(counts_names)
    lib_keep = [j for j, g in enumerate(lib_genes) if g in counts_set]
    gene_list = [lib_genes[j] for j in lib_keep]
    L = L[:, lib_keep]

    LM = L.sum(axis=1)                                   # expected library size / cell type
    gl_set = set(gene_list)
    cnt_keep = [j for j, g in enumerate(counts_names) if g in gl_set]
    countsM = counts[:, cnt_keep]
    counts_kept_names = [counts_names[j] for j in cnt_keep]
    Lib_spot = countsM.sum(axis=1)

    EL = P @ LM                                          # expected library size per spot
    num_cell = Lib_spot / EL
    nst = P * num_cell[:, None]

    # reorder counts columns to the library matrix's gene order
    order = pd.Index(counts_kept_names).get_indexer(gene_list)
    countsM = countsM[:, order]

    min_dist = _min_spot_distance(coord, rng)
    scale = 100.0 / min_dist
    coord = coord * scale
    min_dist = 100.0

    if sigma.size == 0:
        sigma = np.array([min_dist * 0.001, min_dist, min_dist * 2, min_dist * 3])

    obj = NicheDEObject(
        counts=pd.DataFrame(countsM, index=spot_names, columns=gene_list),
        coord=pd.DataFrame(coord, index=spot_names,
                           columns=(_names(coordinate_mat, 1) or ["x", "y"])),
        sigma=sigma,
        num_cells=pd.DataFrame(nst, index=spot_names, columns=dec_types),
        ref_expr=pd.DataFrame(L, index=lib_types, columns=gene_list),
        cell_names=spot_names,
        cell_types=list(dec_types),
        gene_names=gene_list,
        batch_ID=np.ones(countsM.shape[0], dtype=int),
        spot_distance=min_dist,
        scale=scale,
        Int=Int,
    )
    return obj


def create_nichede_object_from_anndata(adata, library_mat, deconv_mat, sigma=(),
                                       layer=None, spatial_key: str = "spatial",
                                       Int: bool = True, random_state=None):
    """``CreateNicheDEObjectFromSeurat``, with AnnData standing in for Seurat.

    ``seurat_object@images[[1]]@coordinates$imagerow/imagecol`` maps to
    ``adata.obsm[spatial_key]``; the assay counts slot maps to
    ``adata.layers[layer]`` (or ``adata.X``).
    """
    X = adata.layers[layer] if layer is not None else adata.X
    counts = pd.DataFrame(_as_dense(X), index=list(adata.obs_names),
                          columns=list(adata.var_names))
    coord = pd.DataFrame(np.asarray(adata.obsm[spatial_key], dtype=np.float64),
                         index=list(adata.obs_names), columns=["x", "y"])
    return create_nichede_object(counts, coord, library_mat, deconv_mat,
                                 sigma=sigma, Int=Int, random_state=random_state)


# --------------------------------------------------------------------------- #
# MergeObjects / Filter_NDE
# --------------------------------------------------------------------------- #

def merge_objects(objects: list[NicheDEObject]) -> NicheDEObject:
    """``nicheDE::MergeObjects`` — concatenate objects, renormalising coordinates.

    Each subsequent object's coordinates are divided by
    ``(100/scale_j) / (100/scale_1)`` so that one kernel bandwidth means the
    same physical distance in every batch.
    """
    if not isinstance(objects, (list, tuple)):
        raise ValueError("Input must be a list of niche_DE objects")
    ref = objects[0]
    coord_merge = _as_dense(ref.coord)
    counts_merge = _as_dense(ref.counts)
    num_cells_merge = _as_dense(ref.num_cells)
    refr_spot_distance = 100.0 / ref.scale[0]
    batch_merge = np.asarray(ref.batch_ID).copy()
    scales = [1.0]

    for o in objects[1:]:
        if set(np.round(o.sigma, 12)) != set(np.round(ref.sigma, 12)):
            raise ValueError("all objects being merged must consider the same kernel bandwidths")
        if bool(o.Int) != bool(ref.Int):
            raise ValueError("all counts must all be integer valued or all be continuous")
        if not np.array_equal(np.asarray(o.ref_expr), np.asarray(ref.ref_expr)):
            raise ValueError("all objects being merged must use the same reference expression")
        if list(o.num_cells.columns) != list(ref.num_cells.columns):
            raise ValueError("cell types must be the same in merged objects")
        if list(o.counts.columns) != list(ref.counts.columns):
            raise ValueError("genes must be the same and in the same order in merged objects")

        sc = (100.0 / o.scale[0]) / refr_spot_distance
        scales.append(sc)
        coord_merge = np.vstack([coord_merge, _as_dense(o.coord) * (1.0 / sc)])
        num_cells_merge = np.vstack([num_cells_merge, _as_dense(o.num_cells)])
        counts_merge = np.vstack([counts_merge, _as_dense(o.counts)])
        batch_merge = np.concatenate([batch_merge,
                                      np.asarray(o.batch_ID) + batch_merge.max()])

    names = [str(i + 1) for i in range(counts_merge.shape[0])]
    return NicheDEObject(
        counts=pd.DataFrame(counts_merge, index=names, columns=list(ref.counts.columns)),
        coord=pd.DataFrame(coord_merge, index=names, columns=list(ref.coord.columns)),
        sigma=ref.sigma,
        num_cells=pd.DataFrame(num_cells_merge, index=names,
                               columns=list(ref.num_cells.columns)),
        ref_expr=ref.ref_expr,
        cell_names=names,
        cell_types=list(ref.num_cells.columns),
        gene_names=list(ref.counts.columns),
        batch_ID=batch_merge,
        spot_distance=100.0,
        scale=np.asarray(scales),
        Int=ref.Int,
    )


def filter_nde(obj: NicheDEObject, cell_names) -> NicheDEObject:
    """``nicheDE::Filter_NDE`` — keep only the named observations.

    Like R, the selection is a *membership* test, so the surviving rows keep
    their original order rather than the order of ``cell_names``.
    """
    cell_names = list(cell_names)
    if not set(cell_names).issubset(set(obj.cell_names)):
        raise ValueError("Some cell names not present in niche-DE object")
    keep = np.isin(np.asarray(obj.cell_names), np.asarray(cell_names))

    new = NicheDEObject(
        counts=obj.counts.loc[keep],
        coord=obj.coord.loc[keep],
        sigma=obj.sigma,
        num_cells=obj.num_cells.loc[keep],
        ref_expr=obj.ref_expr,
        cell_names=list(np.asarray(obj.cell_names)[keep]),
        cell_types=obj.cell_types,
        gene_names=obj.gene_names,
        batch_ID=np.asarray(obj.batch_ID)[keep],
        spot_distance=obj.spot_distance,
        scale=obj.scale,
        Int=obj.Int,
    )
    new.effective_niche = [en[keep] for en in obj.effective_niche]
    new.niche_DE = obj.niche_DE
    new.niche_DE_pval_pos = obj.niche_DE_pval_pos
    new.niche_DE_pval_neg = obj.niche_DE_pval_neg
    return new


# --------------------------------------------------------------------------- #
# Effective niche
# --------------------------------------------------------------------------- #

def _standardize(EN: np.ndarray) -> np.ndarray:
    """Column z-score with R's ``sd`` (denominator n-1), NaN-aware."""
    mu = np.nanmean(EN, axis=0)
    sd = np.nanstd(EN, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (EN - mu) / sd
    return out


def calculate_effective_niche(obj: NicheDEObject, cutoff: float = 0.05) -> NicheDEObject:
    """``nicheDE::CalculateEffectiveNiche``.

    ``EN = K @ num_cells`` with ``K = exp(-d^2 / sigma^2)`` truncated below
    ``cutoff``; one matrix per kernel bandwidth, column z-scored.
    """
    coord = _as_dense(obj.coord)
    ncell = _as_dense(obj.num_cells)
    batch = np.asarray(obj.batch_ID)
    obj.effective_niche = []

    for sig in obj.sigma:
        EN = None
        ref_size = None
        for counter, ID in enumerate(np.arange(1, len(np.unique(batch)) + 1)):
            sel = batch == ID
            K = squareform(pdist(coord[sel]))
            K = np.exp(-(K ** 2) / (sig ** 2))
            K[K < cutoff] = 0.0
            EN_ds = K @ ncell[sel]
            if counter == 0:
                EN = EN_ds
                ref_size = float(np.mean(ncell[sel].sum(axis=1)))
            else:
                sc = ref_size / float(np.mean(ncell[sel].sum(axis=1)))
                EN = np.vstack([EN, EN_ds * sc])
        EN = _standardize(EN)
        EN[np.isnan(EN)] = 0.0
        obj.effective_niche.append(EN)
    return obj


def calculate_effective_niche_large_scale(obj: NicheDEObject, batch_size: int = 1000,
                                          cutoff: float = 0.05,
                                          standardize: bool = True) -> NicheDEObject:
    """``nicheDE::CalculateEffectiveNicheLargeScale``.

    Tiles the tissue into a ``ceil(sqrt(n_batches))`` square grid and, for each
    tile, only considers candidate neighbours inside a bounding box padded by
    ``sigma * sqrt(-log(cutoff))`` — the radius past which the Gaussian kernel
    is guaranteed to be below ``cutoff``, so the truncation is exact rather
    than approximate.
    """
    coord = _as_dense(obj.coord)
    ncell = _as_dense(obj.num_cells)
    batch = np.asarray(obj.batch_ID)
    n_type = len(obj.cell_types)
    obj.effective_niche = []

    for sig in obj.sigma:
        EN = np.full((coord.shape[0], n_type), np.nan)
        ref_size = None
        for counter, ID in enumerate(np.arange(1, len(np.unique(batch)) + 1)):
            sel = np.flatnonzero(batch == ID)
            coord_ID = coord[sel]
            ncell_ID = ncell[sel]
            EN_ds = np.full((coord_ID.shape[0], n_type), np.nan)

            C = coord_ID.shape[0]
            num_iter = int(np.ceil(C / batch_size))
            square_side = int(np.ceil(np.sqrt(num_iter)))

            x_range = (coord_ID[:, 0].min(), coord_ID[:, 0].max())
            y_range = (coord_ID[:, 1].min(), coord_ID[:, 1].max())
            x_len = x_range[1] - x_range[0]
            y_len = y_range[1] - y_range[0]
            x_bins = np.linspace(x_range[0], x_range[1] + x_len / (2 * square_side), square_side)
            y_bins = np.linspace(y_range[0], y_range[1] + y_len / (2 * square_side), square_side)
            # R's findInterval: number of bin edges <= x  (1-based bin id)
            xb = np.searchsorted(x_bins, coord_ID[:, 0], side="right")
            yb = np.searchsorted(y_bins, coord_ID[:, 1], side="right")

            pad = sig * np.sqrt(-np.log(cutoff))
            for j in range(1, square_side + 1):
                for k in range(1, square_side + 1):
                    inds = np.flatnonzero((xb == j) & (yb == k))
                    if inds.size == 0:
                        continue
                    xr = (coord_ID[inds, 0].min() - pad, coord_ID[inds, 0].max() + pad)
                    yr = (coord_ID[inds, 1].min() - pad, coord_ID[inds, 1].max() + pad)
                    cand = np.flatnonzero(
                        (coord_ID[:, 0] >= xr[0]) & (coord_ID[:, 0] <= xr[1]) &
                        (coord_ID[:, 1] >= yr[0]) & (coord_ID[:, 1] <= yr[1]))
                    D = cdist(coord_ID[inds], coord_ID[cand])
                    D = np.exp(-(D ** 2) / (sig ** 2))
                    D[D < cutoff] = 0.0
                    EN_ds[inds] = D @ ncell_ID[cand]

            if counter == 0:
                EN[sel] = EN_ds
                ref_size = float(np.mean(ncell_ID.sum(axis=1)))
            else:
                sc = ref_size / float(np.mean(ncell_ID.sum(axis=1)))
                EN[sel] = EN_ds * sc

        if standardize:
            EN = _standardize(EN)
        EN[np.isnan(EN)] = 0.0
        obj.effective_niche.append(EN)
    return obj
