"""py-nichede — a pure-Python port of the R package `nicheDE`.

Niche-DE (Mason et al., *Genome Biology* 2024) answers the question one step up
from spatially-variable genes: **does cell type A express different genes when
it sits next to cell type B?**  It regresses each gene's spot-level counts on
the interaction between (a) the posterior share of that gene's expression
attributable to each *index* cell type and (b) a kernel-smoothed "effective
niche" describing which cell types surround the spot, and reports a Wald
statistic per (index, niche, gene) triple.

Two equivalent APIs are exposed:

* a **functional** API that mirrors the R exports one-to-one
  (``create_nichede_object``, ``calculate_effective_niche``, ``niche_DE``,
  ``get_niche_DE_genes``, ``niche_LR_spot`` …), and
* a **class** API, :class:`NicheDE`, for method chaining.

Parity with the R reference is enforced by ``tests/test_exact_match.py``
against the gate pre-registered in ``data/manifest.yaml``.
"""

from .core import NicheDE
from .downstream import (
    get_niche_DE_genes,
    niche_DE_markers,
    niche_LR_cell,
    niche_LR_spot,
)
from .niche_de import niche_DE, niche_DE_no_parallel
from .object_creation import (
    NicheDEObject,
    calculate_effective_niche,
    calculate_effective_niche_large_scale,
    create_library_matrix,
    create_library_matrix_from_anndata,
    create_nichede_object,
    create_nichede_object_from_anndata,
    filter_nde,
    merge_objects,
)
from .poolr import fisher_generalized, mvnconv
from .pvalues import (
    T_to_p,
    celltype_level,
    celltype_level_fisher,
    check_colloc,
    contrast_post,
    gene_level,
    gene_level_fisher,
    get_niche_DE_pval_fisher,
    get_niche_DE_pval_raw,
    ultosymmetric,
)
from .rstats import nb_lik

__all__ = [
    # class API
    "NicheDE",
    "NicheDEObject",
    # object creation
    "create_library_matrix",
    "create_library_matrix_from_anndata",
    "create_nichede_object",
    "create_nichede_object_from_anndata",
    "merge_objects",
    "filter_nde",
    "calculate_effective_niche",
    "calculate_effective_niche_large_scale",
    # main
    "niche_DE",
    "niche_DE_no_parallel",
    # p-values / helpers
    "get_niche_DE_pval_fisher",
    "get_niche_DE_pval_raw",
    "T_to_p",
    "gene_level",
    "celltype_level",
    "gene_level_fisher",
    "celltype_level_fisher",
    "contrast_post",
    "check_colloc",
    "ultosymmetric",
    "nb_lik",
    # downstream
    "get_niche_DE_genes",
    "niche_DE_markers",
    "niche_LR_spot",
    "niche_LR_cell",
    # poolr replacement
    "mvnconv",
    "fisher_generalized",
]

__version__ = "0.1.0"
