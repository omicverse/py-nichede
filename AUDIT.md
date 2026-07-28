## R function coverage audit

### Coverage summary

| Category | Ported | Total | % |
|---|---|---|---|
| Exported R functions | 22 | 26 | 84.6% |
| Internal helpers (reachable) | 0 | 0 | 0.0% |

_Python package exposes 90 unique names._

### Exported R functions

| R function | Python equivalent | Status |
|---|---|---|
| `CalculateEffectiveNiche` | `calculate_effective_niche` | ✅ ported |
| `CalculateEffectiveNicheLargeScale` | `calculate_effective_niche_large_scale` | ✅ ported |
| `CreateLibraryMatrix` | `create_library_matrix` | ✅ ported |
| `CreateLibraryMatrixFromSeurat` | `—` | ❌ MISSING |
| `CreateNicheDEObject` | `—` | ❌ MISSING |
| `CreateNicheDEObjectFromSeurat` | `—` | ❌ MISSING |
| `Filter_NDE` | `filter_nde` | ✅ ported |
| `MergeObjects` | `merge_objects` | ✅ ported |
| `T_to_p` | `T_to_p` | ✅ ported |
| `celltype_level` | `celltype_level` | ✅ ported |
| `celltype_level_fisher` | `celltype_level_fisher` | ✅ ported |
| `check_colloc` | `check_colloc` | ✅ ported |
| `contrast_post` | `contrast_post` | ✅ ported |
| `gene_level` | `gene_level` | ✅ ported |
| `gene_level_fisher` | `gene_level_fisher` | ✅ ported |
| `get_niche_DE_genes` | `get_niche_DE_genes` | ✅ ported |
| `get_niche_DE_pval_fisher` | `get_niche_DE_pval_fisher` | ✅ ported |
| `get_niche_DE_pval_raw` | `get_niche_DE_pval_raw` | ✅ ported |
| `nb_lik` | `nb_lik` | ✅ ported |
| `niche_DE` | `niche_DE` | ✅ ported |
| `niche_DE_markers` | `niche_DE_markers` | ✅ ported |
| `niche_DE_no_parallel` | `niche_DE_no_parallel` | ✅ ported |
| `niche_LR_cell` | `niche_LR_cell` | ✅ ported |
| `niche_LR_spot` | `niche_LR_spot` | ✅ ported |
| `print` | `—` | ❌ MISSING |
| `ultosymmetric` | `ultosymmetric` | ✅ ported |

### Internal helpers reachable from exports

| R helper | File | Python equivalent | Status |
|---|---|---|---|
