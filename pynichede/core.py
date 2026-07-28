"""Class API — the scanpy/omicverse-idiomatic front end.

The functional API in the other modules mirrors R one-to-one; this class wraps
it in a method-chaining object so a Python user can write::

    nde = (NicheDE.from_matrices(counts, coord, libmat, deconv, sigma=[1, 100, 250])
             .effective_niche()
             .run(num_cores=8))
    nde.genes(level="I", index="tumor_epithelial", niche="myeloid")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import downstream as _dn
from . import niche_de as _nde
from . import object_creation as _oc
from . import pvalues as _pv

__all__ = ["NicheDE"]


class NicheDE:
    """Chainable wrapper around a :class:`~pynichede.object_creation.NicheDEObject`."""

    def __init__(self, obj):
        self.obj = obj

    # ---- constructors -----------------------------------------------------
    @classmethod
    def from_matrices(cls, counts_mat, coordinate_mat, library_mat, deconv_mat,
                      sigma=(), Int: bool = True, random_state=None) -> "NicheDE":
        """``CreateNicheDEObject``."""
        return cls(_oc.create_nichede_object(counts_mat, coordinate_mat, library_mat,
                                             deconv_mat, sigma=sigma, Int=Int,
                                             random_state=random_state))

    @classmethod
    def from_anndata(cls, adata, library_mat, deconv_mat, sigma=(), layer=None,
                     spatial_key: str = "spatial", Int: bool = True,
                     random_state=None) -> "NicheDE":
        """``CreateNicheDEObjectFromSeurat``, AnnData flavour."""
        return cls(_oc.create_nichede_object_from_anndata(
            adata, library_mat, deconv_mat, sigma=sigma, layer=layer,
            spatial_key=spatial_key, Int=Int, random_state=random_state))

    # ---- slots ------------------------------------------------------------
    @property
    def counts(self):
        return self.obj.counts

    @property
    def coord(self):
        return self.obj.coord

    @property
    def sigma(self):
        return self.obj.sigma

    @property
    def num_cells(self):
        return self.obj.num_cells

    @property
    def ref_expr(self):
        return self.obj.ref_expr

    @property
    def effective_niche_(self):
        return self.obj.effective_niche

    @property
    def cell_types(self):
        return self.obj.cell_types

    @property
    def gene_names(self):
        return self.obj.gene_names

    @property
    def niche_DE_pval_pos(self):
        return self.obj.niche_DE_pval_pos

    @property
    def niche_DE_pval_neg(self):
        return self.obj.niche_DE_pval_neg

    def __repr__(self):
        return repr(self.obj)

    # ---- pipeline ---------------------------------------------------------
    def effective_niche(self, cutoff: float = 0.05, large_scale: bool = False,
                        batch_size: int = 1000, standardize: bool = True) -> "NicheDE":
        """``CalculateEffectiveNiche`` / ``CalculateEffectiveNicheLargeScale``."""
        if large_scale:
            self.obj = _oc.calculate_effective_niche_large_scale(
                self.obj, batch_size=batch_size, cutoff=cutoff, standardize=standardize)
        else:
            self.obj = _oc.calculate_effective_niche(self.obj, cutoff=cutoff)
        return self

    def run(self, num_cores: int = 1, C: float = 150, M: int = 10,
            gamma: float = 0.8, Int: bool = True, batch: bool = True,
            self_EN: bool = False, verbose: bool = True) -> "NicheDE":
        """``niche_DE`` — the main regression plus both p-value tables."""
        self.obj = _nde.niche_DE(self.obj, num_cores=num_cores, C=C, M=M,
                                 gamma=gamma, Int=Int, batch=batch,
                                 self_EN=self_EN, verbose=verbose)
        return self

    def filter(self, cell_names) -> "NicheDE":
        """``Filter_NDE``."""
        self.obj = _oc.filter_nde(self.obj, cell_names)
        return self

    @staticmethod
    def merge(objects) -> "NicheDE":
        """``MergeObjects``."""
        raw = [o.obj if isinstance(o, NicheDE) else o for o in objects]
        return NicheDE(_oc.merge_objects(raw))

    # ---- downstream -------------------------------------------------------
    def genes(self, level: str, index=None, niche=None, positive: bool = True,
              alpha: float = 0.05) -> pd.DataFrame:
        """``get_niche_DE_genes``."""
        return _dn.get_niche_DE_genes(self.obj, level, index=index, niche=niche,
                                      positive=positive, alpha=alpha)

    def markers(self, index: str, niche1: str, niche2: str,
                alpha: float = 0.05) -> pd.DataFrame:
        """``niche_DE_markers``."""
        return _dn.niche_DE_markers(self.obj, index, niche1, niche2, alpha=alpha)

    def ligand_receptor(self, ligand_cell: str, receptor_cell: str,
                        ligand_target_matrix, lr_mat, resolution: str = "spot",
                        **kwargs) -> pd.DataFrame:
        """``niche_LR_spot`` (``resolution='spot'``) / ``niche_LR_cell`` (``'cell'``)."""
        fn = _dn.niche_LR_spot if resolution == "spot" else _dn.niche_LR_cell
        return fn(self.obj, ligand_cell, receptor_cell, ligand_target_matrix,
                  lr_mat, **kwargs)

    def check_colloc(self, index: str, niche: str) -> np.ndarray:
        """``check_colloc``, by cell-type *name*."""
        cts = list(self.obj.cell_types)
        return _pv.check_colloc(self.obj, cts.index(index), cts.index(niche))

    def pval_raw(self, pos: bool = True) -> dict:
        """``get_niche_DE_pval_raw`` — un-BH-adjusted p-values (does not mutate)."""
        return _pv._pval_core(self.obj, pos=pos, adjust=False, verbose=False)

    # ---- interop ----------------------------------------------------------
    def to_anndata(self, adata=None):
        """Write the results back into an AnnData.

        ``.obsm['nichede_effective_niche_<sigma>']`` gets each effective niche;
        ``.varm['nichede_pval_pos_ct']`` / ``'..._neg_ct'`` get the cell-type
        level p-values; ``.uns['nichede']`` gets the gene-level vectors, the
        interaction-level arrays and the run metadata.
        """
        import anndata as ad
        o = self.obj
        if adata is None:
            adata = ad.AnnData(np.asarray(o.counts, dtype=np.float64))
            adata.obs_names = list(o.cell_names)
            adata.var_names = list(o.gene_names)
            adata.obsm["spatial"] = np.asarray(o.coord, dtype=np.float64)
        adata.obsm["nichede_num_cells"] = np.asarray(o.num_cells, dtype=np.float64)
        for k, s in enumerate(o.sigma):
            adata.obsm[f"nichede_effective_niche_{s:g}"] = o.effective_niche[k]
        uns = {"sigma": np.asarray(o.sigma), "cell_types": list(o.cell_types)}
        for tag, pv in (("pos", o.niche_DE_pval_pos), ("neg", o.niche_DE_pval_neg)):
            if pv:
                adata.varm[f"nichede_pval_{tag}_ct"] = np.asarray(pv["cell_type_level"])
                uns[f"pval_{tag}_gene"] = np.asarray(pv["gene_level"])
                uns[f"pval_{tag}_interaction"] = np.asarray(pv["interaction_level"])
        adata.uns["nichede"] = uns
        return adata
