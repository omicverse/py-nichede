<h1 align="center">py-nichede</h1>

<p align="center">
  <b>Niche-differential expression for spatial transcriptomics — a pure-Python port of the R package <code>nicheDE</code>.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/pynichede/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pynichede.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue.svg">
</p>

---

## What it does

Spatial tools can already tell you *which genes vary across a tissue*. Niche-DE
answers the question one level up:

> **Does cell type A express different genes when it sits next to cell type B?**

For every gene, Niche-DE regresses the spot-level counts on the interaction
between (a) the share of that gene's expression attributable to each **index**
cell type and (b) a kernel-smoothed **effective niche** describing which cell
types surround the spot, and reports a Wald statistic per
*(index cell type, niche cell type, gene)* triple. Those statistics are then
pooled — with Brown's method across interactions and a Cauchy combination across
kernel bandwidths — into gene-level, cell-type-level and interaction-level
p-values, and turned into niche marker genes and context-dependent
ligand–receptor pairs.

`py-nichede` is a line-by-line port of the reference R implementation
([kaishumason/NicheDE](https://github.com/kaishumason/NicheDE)), validated
against it on the upstream vignette dataset under a parity gate that was
committed **before** any Python was written.

**Paper**: Mason K., Sathe A., Hess P.R., Rong J., Wu C.-Y., Furth E., Susztak K.,
Levinsohn J., Ji H.P., Zhang N.R. *Niche-DE: niche-differential gene expression
analysis in spatial transcriptomics data identifies context-dependent cell–cell
interactions.* **Genome Biology** 25, 14 (2024).
[PMC10785550](https://pmc.ncbi.nlm.nih.gov/articles/PMC10785550/)

---

## Install

```bash
pip install pynichede
```

Runtime dependencies are `numpy`, `scipy`, `pandas`, `anndata`, `joblib`,
`threadpoolctl`. **No R and no `rpy2`.**

---

## Quick start (class API)

```python
import pynichede as nde

nd = (
    nde.NicheDE.from_matrices(
        counts_mat=counts,          # spots x genes, raw integer counts
        coordinate_mat=coord,       # spots x 2
        library_mat=libmat,         # cell types x genes (reference profile)
        deconv_mat=deconv,          # spots x cell types (e.g. RCTD output)
        sigma=[1, 100, 250],        # kernel bandwidths
    )
    .effective_niche(cutoff=0.05)
    .run(num_cores=8, C=150, M=10, gamma=0.8)
)

# genes the tumour compartment turns on specifically next to myeloid cells
nd.genes("I", index="tumor_epithelial", niche="myeloid", positive=True, alpha=0.05)

# genes that distinguish a myeloid niche from a stromal one
nd.markers("tumor_epithelial", "myeloid", "stromal")

# context-dependent ligand-receptor pairs
nd.ligand_receptor("myeloid", "tumor_epithelial",
                   ligand_target_matrix=ltm, lr_mat=lr, resolution="spot")

adata = nd.to_anndata()   # results into .obsm / .varm / .uns
```

Starting from an `AnnData` instead:

```python
nd = nde.NicheDE.from_anndata(adata, library_mat=libmat, deconv_mat=deconv,
                              sigma=[1, 100, 250], spatial_key="spatial")
```

## Low-level functional API (mirrors R one-to-one)

```python
obj = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=[1, 100, 250])
obj = nde.calculate_effective_niche(obj, cutoff=0.05)
obj = nde.niche_DE(obj, num_cores=8, C=150, M=10, gamma=0.8)
res = nde.get_niche_DE_genes(obj, "I", index="tumor_epithelial", niche="myeloid")
```

---

## What's included

Every symbol exported by the R `NAMESPACE` has a Python counterpart.

| R (`nicheDE`) | Python (`pynichede`) | Notes |
|---|---|---|
| `CreateLibraryMatrix` | `create_library_matrix` | |
| `CreateLibraryMatrixFromSeurat` | `create_library_matrix_from_anndata` | Seurat → AnnData |
| `CreateNicheDEObject` | `create_nichede_object` | |
| `CreateNicheDEObjectFromSeurat` | `create_nichede_object_from_anndata` | Seurat → AnnData |
| `MergeObjects` | `merge_objects` | |
| `Filter_NDE` | `filter_nde` | |
| `CalculateEffectiveNiche` | `calculate_effective_niche` | |
| `CalculateEffectiveNicheLargeScale` | `calculate_effective_niche_large_scale` | see *Known divergences* |
| `niche_DE` | `niche_DE` | |
| `niche_DE_no_parallel` | `niche_DE_no_parallel` | |
| `get_niche_DE_pval_fisher` | `get_niche_DE_pval_fisher` | |
| `get_niche_DE_pval_raw` | `get_niche_DE_pval_raw` | |
| `get_niche_DE_genes` | `get_niche_DE_genes` | |
| `niche_DE_markers` | `niche_DE_markers` | |
| `niche_LR_spot` | `niche_LR_spot` | |
| `niche_LR_cell` | `niche_LR_cell` | |
| `contrast_post` | `contrast_post` | 0-based cell-type indices |
| `check_colloc` | `check_colloc` | 0-based cell-type indices |
| `gene_level` | `gene_level` | |
| `celltype_level` | `celltype_level` | |
| `gene_level_fisher` | `gene_level_fisher` | |
| `celltype_level_fisher` | `celltype_level_fisher` | |
| `T_to_p` | `T_to_p` | |
| `ultosymmetric` | `ultosymmetric` | |
| `nb_lik` | `nb_lik` | |
| `print.Niche_DE` | `NicheDEObject.__repr__` | |
| *(poolr dependency)* | `mvnconv`, `fisher_generalized` | clean-room, MIT |

Two extra dependency-free modules are exposed for other ports to reuse:

- `pynichede.rstats` — R-faithful `glm.fit` (LINPACK `dqrdc2` limited-pivot rank
  detection, `NA`-for-aliased coefficients), `optimize` (Brent `fmin`),
  `p.adjust`, type-7 `quantile`, `weighted.mean`.
- `pynichede.poolr` — Brown's method and `mvnconv`, re-derived from first
  principles rather than vendored (poolr is GPL-2+, this port is MIT).

---

## Reproducing the R results exactly

The parity gate is a runnable test. Generate the R reference, run the Python
candidate, and assert the pre-registered thresholds:

```bash
export R_LIBS_USER=/path/to/rlibs           # where nicheDE + poolr are installed
export REF=/tmp/nichede_ref

Rscript tests/r_reference_driver.R      $REF 0 16    # 0 = full gene set
Rscript tests/r_reference_supplement.R  $REF 0
python  tests/_run_candidate.py         $REF   16

NICHEDE_REF_DIR=$REF pytest -q                        # the gate
python tests/parity_report.py $REF                    # the numbers
```

`pytest -q` without `NICHEDE_REF_DIR` skips the R-dependent tests and runs the
31 R-free unit tests, so a bare checkout stays green.

### Measured parity — canonical fixture

10x Visium human liver metastasis (the dataset shipped with the R package):
**848 spots x 21 708 genes x 7 cell types x 3 kernel bandwidths**.

| Output | Class | Metric | Threshold | **Measured** |
|---|---|---|---|---|
| `T_stat` (Wald statistic, all 3 kernels) | ordinal | Pearson / Spearman | ≥ 0.99 | **1.000000 / 1.000000** |
| `betas` | ordinal | Pearson | ≥ 0.99 | **1.000000** |
| `log_likelihood` | ordinal | Pearson | ≥ 0.99 | **1.000000** |
| `valid` gene flags | exact | agreement | 1.0 | **1.000000** (4768 / 4788 / 4858) |
| `nulls` (dropped interactions) | exact | max abs err | 0 | **0** |
| `num_cells` | deterministic | max abs err | < 1e-8 | **4.7e-14** |
| `effective_niche` | deterministic | max abs err | < 1e-8 | **2.9e-13** |
| `pval_pos` gene level | inference | Spearman(-log10 p) / top-50 J | ≥ 0.90 / 0.70 | **0.9966 / 1.000** |
| `pval_pos` cell-type level | inference | Spearman / top-50 J | ≥ 0.90 / 0.70 | **0.9960 / 0.923** |
| `pval_pos` interaction level | inference | Spearman / top-50 J | ≥ 0.90 / 0.70 | **1.000000 / 1.000** |
| `pval_neg` gene level | inference | Spearman / top-50 J | ≥ 0.90 / 0.70 | **0.9960 / 1.000** |
| `get_niche_DE_genes` output sets | ranked | Jaccard | ≥ 0.70 | **0.938 – 0.972** |
| `niche_DE_markers` output set | ranked | Jaccard | ≥ 0.70 | **1.000** |

Full breakdown, including the `Int = FALSE` branch and the helper functions, in
[`RECONSTRUCTION_REPORT.md`](RECONSTRUCTION_REPORT.md).

### Speed

`niche_DE` on the canonical fixture (848 x 21 708 x 7 x 3):

| | wall clock |
|---|---|
| R `nicheDE::niche_DE`, 16 cores | **852 s** |
| `pynichede.niche_DE`, 16 cores | **33 s** |

≈ **26x faster**, entirely from exact algebraic rewrites and scheduling — no
approximation.

`niche_LR_spot` on the same fixture (579 candidate ligands against the
16 968 x 579 NicheNet matrix) went from **314.5 s to 3.0 s (105x)** by
memoising a per-kernel slice that the R source recomputes once per ligand —
byte-identical output. See [`ITERATION_LOG.md`](ITERATION_LOG.md) and
[`MATH.md`](MATH.md).

---

## Known divergences from the shipped R package

These are measured, not hand-waved. See [`MATH.md`](MATH.md) §3.

1. **`CalculateEffectiveNicheLargeScale` — Python is correct, R is not.**
   With `Rfast >= 2.1.5.2`, `Rfast::dista()` returns an **all-zero matrix**
   whenever `nrow(xnew) >= 4`, so every Gaussian kernel weight in the tiled path
   collapses to 1. The shipped R `CalculateEffectiveNicheLargeScale` therefore
   disagrees with the shipped R `CalculateEffectiveNiche` by up to **16.4
   z-units** on the canonical fixture. `py-nichede` implements the intended
   algorithm and matches the exact effective niche to `2.9e-13`; it is
   parity-tested against a **repaired** R version.

2. **`poolr::mvnconv` — Python is the more accurate side.**
   poolr's shipped `mvnlookup` is a numerically-integrated table stored to four
   decimals; its `chisq1_2` column deviates from its own exact closed form
   `2 rho^2` by up to `6.28e-4`. `py-nichede` re-derives the table exactly (via
   Mehler's formula) instead of vendoring GPL data. Net effect on a pooled
   Brown p-value: **3.06e-4 relative**.

3. **Bug-compatible by design.** Four defects in the R source change *which
   genes get reported*, so they are reproduced rather than fixed — including
   the `new_nul` / `var` typo that invalidates a gene whenever `X'WX` has a zero
   diagonal, and the `X[, -null]` dimension drop that invalidates a gene
   whenever exactly one interaction survives the `M` filter. Reproducing them is
   what takes the `valid`-flag agreement from 0.9933 to exactly 1.0.

---

## Notebooks

| Notebook | For |
|---|---|
| [`examples/compare_R_vs_Python.ipynb`](examples/compare_R_vs_Python.ipynb) | reviewers — pipeline-level parity vs R, one figure per gated output |
| [`examples/tutorial_liver_met_visium.ipynb`](examples/tutorial_liver_met_visium.ipynb) | new users — Python-only walkthrough of every public function |
| [`examples/function_by_function_R_parity.ipynb`](examples/function_by_function_R_parity.ipynb) | R users migrating — per-function parameter dictionary + side-by-side calls |
| [`examples/evolution.ipynb`](examples/evolution.ipynb) | auditors — per-iteration narrative of how the port was built |

---

## Relationship to omicverse

`py-nichede` is a standalone package. It is also the niche-differential-expression
backend for [omicverse](https://github.com/Starlitnightly/omicverse), filling a
gap no other Python tool covers: `squidpy` and `liana-py` provide spatially
variable genes and ligand–receptor scoring, but neither models
index-cell-type x niche-cell-type interaction effects on expression.

It was produced with the
[Omicverse-RebuildR](https://github.com/omicverse/omicverse-rebuildr) protocol:
the R source is treated as the executable spec, the parity metric and threshold
are pre-registered before any Python is written, and every acceleration rewrite
must ship an admissibility proof.

---

## Citation

If you use `py-nichede`, please cite the original method:

```bibtex
@article{mason2024nichede,
  title   = {Niche-DE: niche-differential gene expression analysis in spatial
             transcriptomics data identifies context-dependent cell-cell interactions},
  author  = {Mason, Kaishu and Sathe, Anuja and Hess, Paul R. and Rong, Jiazhen and
             Wu, Chi-Yun and Furth, Emma and Susztak, Katalin and Levinsohn, Jonathan and
             Ji, Hanlee P. and Zhang, Nancy R.},
  journal = {Genome Biology},
  volume  = {25},
  number  = {1},
  pages   = {14},
  year    = {2024},
  doi     = {10.1186/s13059-023-03159-6}
}
```

and, if the Python port itself was load-bearing, link back to this repository.

## License

MIT, matching the upstream R package. See [LICENSE](LICENSE).
`pynichede.poolr` is an independent clean-room implementation and contains no
GPL-licensed code or data from `poolr`.
