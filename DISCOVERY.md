# Discovery — `py-nichede`

> Omicverse-RebuildR PROTOCOL.md **Step 0**. Written and committed *before* any
> algorithmic Python code.

## 1. Target identification

The paper is **Mason K., Sathe A., Hess P.R., Rong J., Wu C.-Y., Furth E.,
Susztak K., Levinsohn J., Ji H.P., Zhang N.R. — "Niche-DE: niche-differential
gene expression analysis in spatial transcriptomics data identifies
context-dependent cell–cell interactions", *Genome Biology* 25:14 (2024)**
(<https://pmc.ncbi.nlm.nih.gov/articles/PMC10785550/>).

The paper's Availability-of-data section points at
<https://github.com/kaishumason/NicheDE> (the pkgdown site advertised in
`DESCRIPTION` is `https://kmason23.github.io/NicheDE/`; `kmason23/NicheDE` is a
0-star empty mirror of the same author, `kaishumason/NicheDE` is the live 23-star
repo the paper links and the one the docs site is built from).

**The R source treated as the executable spec for this port:**

| field | value |
|---|---|
| repo | `https://github.com/kaishumason/NicheDE` |
| commit | `87e0e89bb066702a54fa47638965b61dc6f24d05` |
| date | 2025-06-10 |
| version (`DESCRIPTION`) | `0.0.0.9000` |
| license | MIT (repo `LICENSE`; the `DESCRIPTION` `License:` field is an unfilled `usethis` placeholder) |
| R files | `R/niche_DE_object_creation.R` (703 L), `R/niche_DE_main_functions.R` (1571 L), `R/niche_DE_helper_functions.R` (556 L) — 2828 L total |
| exported symbols | 26 (`NAMESPACE`) |

Cloned into `nichede-ref/` (gitignored). The R package is installed into a
private library `/scratch/users/steorra/Rlibs_nichede` so that the shared
`CMAP` env is not modified.

## 2. Is it already ported?

```
$ python -m engine.discover_omicverse_deps --check NicheDE
[discover] using cached repo list (95 repos, age 5 min)
## Discovery — `NicheDE`
**No existing omicverse port found.** Safe to start a new port.
```

Manual cross-check of the 95 `github.com/omicverse` repos: no `py-nicheDE`,
`py-nichede`, `py-nicheDE-R`, or alias. Nothing in `squidpy`, `scanpy`,
`liana-py`, or `omicverse` implements niche-differential expression (they cover
spatially-variable genes and ligand–receptor scoring, not the
index-cell-type × niche-cell-type interaction regression). **Proceed.**

## 3. Upstream R dependency audit

`DESCRIPTION::Imports` = `abind, Matrix, Seurat, foreach, stats, fastDummies,
parallel, poolr, spatstat.utils`. Plus two undeclared but `::`-called packages
found by grepping the source: `Rfast` (`Rfast::dista`) and `doParallel`.

| R dep | Used for | omicverse mirror | Decision |
|---|---|---|---|
| `Matrix` | sparse counts, `chol`, `solve`, `t` | [`anndata-oom`](https://github.com/omicverse/anndata-oom) (sparse AnnData, not a `Matrix` mirror) | **native**: `scipy.sparse` + `scipy.linalg.cholesky`. The omicverse match is a false positive — `anndata-oom` mirrors `anndata`, not R `Matrix`. |
| `Seurat` | `GetAssay`, `Idents`, `@images` extraction in the two `*FromSeurat` constructors | [`py-cca`](https://github.com/omicverse/py-cca) (mirrors `RunCCA` only) | **native**: replaced by an `AnnData` constructor (`from_anndata`), which is the Python-ecosystem equivalent of "a Seurat object". `py-cca` is a false positive — it mirrors CCA, not the Seurat object model. |
| `stats` | `glm`, `lm`, `optimize`, `dnbinom`, `pnorm`, `pcauchy`, `pchisq`, `p.adjust`, `quantile`, `dist` | — | **native**, but *not* off-the-shelf: R's `glm.fit` (LINPACK `dqrdc2` limited-pivot rank detection, `NA` aliased coefficients), R's `optimize` (Brent `fmin`), R's `p.adjust(method="BH")` NA convention and R's type-7 `quantile` are all reimplemented bit-faithfully in `pynichede/rstats.py`. `statsmodels.GLM` does **not** reproduce R's rank-deficiency handling, so it is deliberately not used. |
| `poolr` | `mvnconv(target="m2lp", side=1)` + `fisher(adjust="generalized")` (Brown's method) | — | **clean-room native reimplementation** in `pynichede/poolr.py`. poolr is GPL-2+ and this port is MIT, so poolr's `mvnlookup` data table is *not* vendored; it is **re-derived from its mathematical definition** (Gauss–Hermite quadrature of `Cov(g(Z₁), g(Z₂))` under a bivariate normal) and verified to reproduce all 1991 × 8 published entries exactly after poolr's own 4-decimal rounding. See §4. |
| `fastDummies` | `dummy_cols(remove_first_dummy=TRUE)` on the batch factor | — | **native**: `pandas.get_dummies(drop_first=True)`, 6 lines. |
| `abind` | `abind(..., along=3)` to stack per-σ cell-type p-value matrices | — | **native**: `numpy.stack(..., axis=2)`. |
| `spatstat.utils` | `inside.range()` inside the large-scale binned effective-niche | — | **native**: two numpy comparisons. |
| `Rfast` | `dista()` — cross Euclidean distance matrix | — | **native**: `scipy.spatial.distance.cdist`. |
| `foreach` / `doParallel` / `parallel` | `%dopar%` over genes | — | **native**: `joblib.Parallel`. Parallelism is an implementation detail; the R result is order-independent by construction (one independent GLM per gene). |

### Ecosystem-reuse accounting

- **0 of 11** upstream R dependencies had a usable `omicverse/py-*` mirror. The
  two matches the automated scan reported (`anndata-oom` ← `Matrix`,
  `py-cca` ← `Seurat`) are name-level false positives; both were inspected and
  rejected on function coverage.
- Consequently `py-nichede` takes **no `omicverse/py-*` hard dependency**. Its
  runtime dependency set is `numpy, scipy, pandas, anndata, joblib` only.
- Reuse still happened at the *ecosystem* level in the opposite direction:
  `pynichede.rstats` (R `glm.fit` with `dqrdc2` pivoting, R `optimize`/Brent
  `fmin`, R `p.adjust`, R type-7 `quantile`) and `pynichede.poolr` (Brown's
  method / `mvnconv`) are written as standalone, dependency-free modules
  precisely so the next port that needs R-faithful GLM or Brown-combination can
  lift them. Both are flagged in §5 as reusable.

## 4. The one genuinely hard dependency: `poolr`

Niche-DE combines the up-to-49 correlated per-interaction p-values of one gene
into gene-level and cell-type-level p-values with **Brown's method**
(`poolr::fisher(..., adjust = "generalized")`), which needs the covariance of
the `-2 log p` statistics implied by the correlation matrix of the β̂ vector.
`poolr::mvnconv` obtains that covariance from a shipped lookup table
`mvnlookup` (1991 rows × 9 columns: ρ from 1.000 down to −0.990 in steps of
0.001, times 4 targets × 2 sides).

Because that table is GPL-2+ data and this port is MIT, it is **not copied**.
Instead `pynichede/poolr.py` re-derives it: for target `t` and side `s`, define
the transform `g` mapping a standard normal draw `Z` to the target statistic
(`side=1`: `p = 1 − Φ(Z)`; `side=2`: `p = 2(1 − Φ(|Z|))`; then
`p`→`p`, `z`→`Φ⁻¹(1−p)`, `chisq1`→`qchisq(1−p, 1)`, `m2lp`→`−2 log p`) and
compute `Cov(g(Z₁), g(Z₂))` for `(Z₁,Z₂) ~ BVN(ρ)` by tensor Gauss–Hermite
quadrature. `tests/test_poolr_table.py` asserts the re-derived table equals the
published poolr table on all 1991 × 8 entries after poolr's own 4-decimal
rounding, so the port is numerically identical to poolr without carrying
poolr's code or data.

## 5. Modules written to be reusable by later ports

| module | what it gives the next port |
|---|---|
| `pynichede/rstats.py` | `r_glm_fit` (Poisson/Gaussian IRLS with LINPACK `dqrdc2` limited-pivot rank detection and R's `NA`-for-aliased-coefficient convention), `r_optimize` (Brent `fmin`, R's `.Machine$double.eps^0.25` default tol), `p_adjust` (`BH`/`bonferroni`/`holm` with R's NA convention `n = length(p)` ≠ `length(p[!is.na])`), `r_quantile` (type 7), `weighted_mean` (R's `na.rm` + `w != 0` rule) |
| `pynichede/poolr.py` | `mvnconv`, `fisher_generalized` (Brown's method), and the re-derived `mvnlookup` generator |

Both are importable without touching any niche-DE state.

## 6. Roadmap additions found during this audit

None of the R deps is itself worth porting as a standalone omicverse package
(they are utility packages). `poolr` is the only statistically interesting one
and it is fully covered by `pynichede/poolr.py` — if another port needs it,
lift that module rather than opening `py-poolr`.
