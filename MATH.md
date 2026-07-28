# MATH.md — derivations behind `py-nichede`

Three things need a written derivation: the algebraic identities the
Acceleration Agent relied on (§1), the clean-room re-derivation of poolr's
`mvnconv` table (§2), and an honest accounting of every place where the Python
answer is *not* bit-identical to R (§3).

**There are no (B) bounded-epsilon approximations in this port.** Every accepted
rewrite in `ITERATION_LOG.md` is admissibility class (E) — an exact algebraic
identity — or a pure scheduling change. So there is no perturbation budget to
sum against the parity threshold.

---

## 1. Exact identities used by the Acceleration Agent

### 1.1 Design matrix as a stacked outer product (iter 4)

R builds the niche design one spot at a time:

```r
X[k, ] = as.vector( round(effective_niche[k, ], 2) %*% t(as.matrix(pstg[k, ])) )
```

`v %*% t(w)` for length-`p` vectors is the rank-1 matrix `M` with
`M[a, b] = v[a] w[b]`, and `as.vector` flattens column-major, so

```
X[k, (b-1)p + a] = round(EN[k, a], 2) * pstg[k, b].
```

Stacking over `k` is the broadcast `EN[:, :, None] * pstg[:, None, :]`, whose
`[k, a, b]` entry is the same product. Re-ordering the trailing two axes so `a`
varies fastest (`transpose(0, 2, 1)` then a C-order reshape) reproduces the
column-major flattening. Each output entry is a **single multiplication of the
same two f64 operands**, so the result is bit-identical, not merely equal to
rounding. Measured max abs deviation on the fixture: `0.0`.

### 1.2 `cho_solve` instead of an explicit triangular inverse (iter 5)

R computes

```r
A <- Matrix::chol(var_mat, LDL = FALSE, perm = FALSE)   # upper, A'A = var_mat
V <- Matrix::solve(A) %*% Matrix::t(Matrix::solve(A))
```

With `A` upper triangular and `A'A = X'WX`,

```
A^-1 (A^-1)' = A^-1 A^-T = (A' A)^-1 = (X'WX)^-1,
```

which is precisely `cho_solve((A, upper=True), I)`, obtained by two triangular
solves instead of forming `A^-1` and multiplying. Exact in exact arithmetic;
measured max **relative** deviation `1.49e-15` (one to two ulp).

The Cholesky factorisation itself is retained rather than replaced by a general
solve, because R's `Matrix::chol` **raising** on a non-positive-definite
`X'WX` is load-bearing: it is what makes `nicheDE` mark such a gene invalid.
`scipy.linalg.cholesky` raises `LinAlgError` in the same situation, so the
control flow is preserved.

### 1.3 Memoising the `mvnconv` table (iter 6)

`mvnconv(R, side, target)` reads a fixed 1991 x 9 table indexed by
`round(rho, 3)`. The table depends on nothing but `(side, target)` and the rho
grid, so caching it is textbook memoisation of a pure function: bit-identical
output, measured max abs deviation `0.0`.

### 1.4 Thread-count and partition changes (iters 7, 9, 10)

Limiting the BLAS pool to one thread does not change which floating-point
operations are performed for `syrk` / `potrf` / `trsm` on a ~848 x 50 design —
those routines partition by output block, not by reduction, so no summation
order changes. Splitting the gene loop across processes is admissible because
each gene's fit reads only shared read-only arrays and writes only its own
result; R itself parallelises over exactly this axis with `foreach %dopar%`.
Hoisting the `sum(counts) > C && !all-below-filter` predicate into the parent
evaluates the same predicate on the same values at the same point in the
control flow. All three were additionally verified empirically: the complete
parity report is unchanged, digit for digit, before and after.

---

## 2. Clean-room derivation of poolr's `mvnconv` table

Niche-DE pools a gene's up-to-49 correlated interaction p-values with **Brown's
method**, `poolr::fisher(p, side = 1, R = ..., adjust = "generalized")`. Brown's
method needs `Var(sum_i -2 log p_i)`, i.e. the pairwise covariances
`Cov(-2 log p_i, -2 log p_j)` implied by the correlation `rho_ij` of the
underlying Wald statistics. `poolr` supplies those from a shipped lookup table
`mvnlookup`.

`poolr` is GPL-2+ and this port is MIT (matching upstream `nicheDE`), so the
table is **not** copied. It is re-derived from its own definition.

### 2.1 Setup

Let `(Z1, Z2) ~ N(0, [[1, rho], [rho, 1]])`. Define the p-value

- `side = 1`: `p = 1 - Phi(Z)`
- `side = 2`: `p = 2 (1 - Phi(|Z|))`

and the target transform `g`

| target | `g(Z)` |
|---|---|
| `p` | `p` |
| `z` | `Phi^-1(1 - p)` |
| `chisq1` | `F_{chi^2_1}^-1(1 - p)` |
| `m2lp` | `-2 log p` |

The table entry is `Cov(g(Z1), g(Z2))` as a function of `rho`.

### 2.2 Mehler's formula

Let `He_n` be the probabilists' Hermite polynomials and
`psi_n = He_n / sqrt(n!)` the orthonormal basis of `L^2(phi)`. Mehler's kernel
expansion of the bivariate normal density gives

```
phi_rho(x, y) / (phi(x) phi(y)) = sum_{n >= 0} (rho^n / n!) He_n(x) He_n(y).
```

Writing `alpha_n = E[g(Z) psi_n(Z)]` and taking expectations term by term,

```
E[g(Z1) g(Z2)] = sum_{n >= 0} alpha_n^2 rho^n ,
```

and since `E[g(Z1)] E[g(Z2)] = alpha_0^2`,

```
    Cov( g(Z1), g(Z2) ) = sum_{n >= 1} alpha_n^2 rho^n .            (*)
```

This is a **power series in rho with non-negative coefficients**. One
one-dimensional quadrature for the `alpha_n` therefore yields the covariance at
every rho on poolr's grid simultaneously — which is also why the Python version
builds the whole 1991-row table in 0.16 s where the naive per-rho 2-D
quadrature took 263 s.

The `alpha_n` are computed with a composite 40-point Gauss-Legendre rule on
panels packed geometrically towards `z = 0` (the `side = 2` targets have a
`|z|`-induced corner there), using the stable three-term recurrence
`psi_{n+1}(z) = (z psi_n(z) - sqrt(n) psi_{n-1}(z)) / sqrt(n+1)`.

### 2.3 Convergence, checked against closed forms

`(*)` at `rho = 1` collapses to `Var(g)`, giving a free convergence test.
For `m2lp` with `side = 1`, `p` is exactly `Uniform(0,1)` so `-2 log p ~ chi^2_2`,
hence `E = 2` and `Var = 4`. The implementation reproduces both to `1e-16` with
50 terms and is run with 2000.

Two entries have exact closed forms, and both are reproduced to `4.4e-16`:

| column | closed form | why |
|---|---|---|
| `z_1` | `rho` | `side = 1` gives `Phi^-1(1 - p) = Z`, so `g` is the identity |
| `chisq1_2` | `2 rho^2` | `side = 2` gives `1 - p = P(|Z| <= |t|) = F_{chi^2_1}(t^2)`, so `g(Z) = Z^2`, and `Cov(Z1^2, Z2^2) = 2 rho^2` |

The only slowly-converging column is `m2lp` with `side = 2`, where `|z|` puts a
corner at the origin and the Hermite coefficients decay polynomially; 2000 terms
leave `~5e-6`, still an order of magnitude inside poolr's own 4-decimal
storage. Niche-DE uses only `side = 1`.

### 2.4 Brown's method itself

With `X = -2 sum_i log p_i`, `E[X] = 2k` and `Var[X] = sum_{i,j} Cov_ij`,
Brown matches the first two moments of a scaled chi-square:

```
c = Var / (2 E),      f = 2 E^2 / Var,      p_pooled = P( chi^2_f > X / c ).
```

Reproduced verbatim, including poolr's `Matrix::nearPD` repair of a
non-positive-definite converted covariance (Higham's alternating projections).

---

## 3. Where the Python answer is *not* bit-identical to R, and why

| # | Source | Magnitude (canonical fixture) | Direction |
|---|---|---|---|
| 1 | poolr's `mvnlookup` is itself a ~1e-3-accurate numerical table | `<= 4.74e-4` on the `m2lp_1` column; **3.06e-4 relative** on an end-to-end Brown p-value | **Python is the more accurate side** |
| 2 | `(X'WX)^-1` computed by LAPACK `dpotrf` + `cho_solve` vs CHOLMOD's sparse supernodal `chol` + `solve` | `2.1e-4` max relative on `Varcov` entries spanning `2.7e-8 .. 2.1e4`; `4.6e-7` max relative on `T_stat` | conditioning, no preferred side |
| 3 | `dnbinom(mu=)` via `lgamma` vs R's Loader saddle-point `dbinom_raw` | `3e-14` max relative on the fitted log-likelihood | negligible |
| 4 | `Rfast::dista` is broken in the installed R stack | see §3.1 | **Python is correct, R is wrong** |

### 3.0 Why #1 is a divergence we chose

poolr's `mvnlookup` was produced by numerical integration and stored to four
decimals. Its `chisq1_2` column, which has the exact closed form `2 rho^2`,
deviates from it by up to `6.28e-4` — proof that the table carries ~1e-3 of
numerical error of its own. Reproducing that error would require copying the
GPL table verbatim. We instead ship the exact values and **measure** the
consequence: on a 6-p-value Brown probe the pooled p-value moves by `3.06e-4`
relative (`0.00291637` vs R's `0.00291548`). Downstream, the gene-level
BH-adjusted p-values still agree with R at Spearman `0.9999` on `-log10 p`.

### 3.1 `Rfast::dista` returns zeros (affects `CalculateEffectiveNicheLargeScale`)

With `Rfast 2.1.5.2` under R 4.4.3,
`Rfast::dista(xnew, x, type = "euclidean", trans = TRUE)` returns an **all-zero
matrix whenever `nrow(xnew) >= 4`** (verified in isolation, independent of
`nicheDE`). `CalculateEffectiveNicheLargeScale` feeds that zero matrix into
`exp(-D^2 / sigma^2)`, so every kernel weight becomes `1` and the "effective
niche" degenerates into an unweighted cell count over the tile's bounding box.
Consequence: the shipped R `CalculateEffectiveNicheLargeScale` disagrees with
the shipped R `CalculateEffectiveNiche` by up to **16.4 z-units** on the
canonical fixture.

The Python port implements the intended algorithm. Its output matches
`CalculateEffectiveNiche` to `8.3e-14` and matches a **repaired** R
`CalculateEffectiveNicheLargeScale` (same R code with base-R `dist`
substituted for `Rfast::dista`, dumped by `tests/r_reference_driver.R` as
`ref_effective_niche_lsfix_*`) to within the deterministic gate.

The tiling is also provably exact rather than approximate: a tile's candidate
box is padded by `sigma * sqrt(-log(cutoff))`, and

```
exp(-d^2 / sigma^2) >= cutoff   <=>   d <= sigma * sqrt(-log cutoff),
```

so every neighbour that survives the `cutoff` truncation is inside the box and
no contribution can be missed. `tests/test_smoke.py::test_large_scale_matches_exact`
pins this.

### 3.2 Deliberately reproduced R defects ("bug compatibility")

These change *which genes get reported*, so the port reproduces them rather
than fixing them. Each is flagged in the source with the R line it mirrors.

| R defect | Effect | Python |
|---|---|---|
| `new_null = new_nul[new_null <= var]` — two undefined symbols | any zero diagonal in `X'WX` throws inside the `tryCatch`; gene marked invalid | `_RErrorCompat` raised at the same point |
| `X_partial = X[, -null]` drops to a vector when one column survives, so `nvar` is `NULL` and `beta[...] <- coeff[-1]` raises "replacement has length zero" | gene marked invalid | same guard (`r_dropped_to_vector`) |
| the invalid branch always returns `nulls = c(1:n_type^2)` regardless of the computed `null` | reported null count is 49 for every invalid gene | same |
| `optimize(nb_lik, x = counts, mu = mu_hat, ...)` passes the full count vector with a `mu_hat` that had zero-expected-expression spots removed | R recycles `mu_hat` | `nb_lik` recycles with `np.resize` |

Without these, the port disagreed with R on 2 of 300 genes in the dev fixture.
With them, `valid`-flag agreement and the `nulls` sets are **exact**.
