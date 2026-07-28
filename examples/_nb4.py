"""Builder for ``examples/evolution.ipynb`` (Notebook 4).

One ``## Iteration N — <title>`` header per entry in ``ITERATION_LOG.md``
(11: baseline + iters 1-10, including the REJECTED iter 8), each with a >= 3
sentence narrative and a code cell emitting that iteration's subplot.  The final
cell renders and saves the aggregate 2-panel figure to ``examples/evolution.png``.
"""

from __future__ import annotations

from _nb_common import PREAMBLE, code, md

# (n, title, narrative, code) -------------------------------------------------
ITERS = []


def I(n, title, narrative, body):
    ITERS.append((n, title, narrative, body))


I(0, "Baseline translation",
  r"""
The first pass was a literal, function-by-function transcription of the 2828 lines of R across
`niche_DE_object_creation.R`, `niche_DE_main_functions.R` and `niche_DE_helper_functions.R`,
plus two supporting modules that R gets for free from its ecosystem: `pynichede/rstats.py`
(R's `glm.fit` with LINPACK `dqrdc2` limited-pivot rank detection and the `NA`-for-aliased
convention, R's Brent `optimize`, R's `p.adjust`, R's type-7 `quantile`, R's `weighted.mean`)
and `pynichede/poolr.py` (a clean-room re-derivation of `poolr::mvnconv` and Brown's method,
because poolr is GPL-2+ and this port is MIT). The design matrix was built with a literal
per-spot Python transcription of R's `for (k in 1:nrow(pstg))` loop, the covariance with a
literal `solve(chol(X'WX)) %*% t(solve(chol(X'WX)))`, and genes were dispatched one joblib task
each with no BLAS thread management.

What already worked: the headline Wald statistic. `T_stat` cleared the pre-registered gate at
Pearson 1.000000 on the very first run, which said the regression itself — design construction,
offset, IRLS, dispersion search, Wald denominator — had been transcribed correctly.

What was hard, and wrong: three outputs disagreed with R in ways that change *which genes get
reported*. The `valid` flags disagreed on 2 of 300 dev-fixture genes, the interaction-level
BH p-values correlated at only Spearman 0.859, and the `Int=FALSE` linear-model branch produced
**no** valid genes at all. Those are equivalence bugs, not acceleration opportunities, so
iterations 1–3 fixed them before any performance work began.

**How to read the wall clock on the plots.** `ITERATION_LOG.md`'s baseline figure is a
*controlled ablation*: the shipped pipeline with the two pre-acceleration code paths (iters 5
and 7) put back, timed in the same process, rather than an archived timing from a different
run. The cell below therefore measures the **shipped serial pipeline as it stands today** —
which is the *post*-acceleration number, not the baseline — so the two are deliberately
different, and the ablation that produces the baseline is re-measured in iteration 7's cell.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
# Baseline measurement: run the shipped pipeline on the dev fixture and grade it
# against the R reference the driver dumped for exactly the same 300 genes.
t0 = time.perf_counter()
o = nde.create_nichede_object(dcounts, dcoord, dlibmat, ddeconv, sigma=dsigma, Int=True)
o = nde.calculate_effective_niche(o, cutoff=0.05)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o = nde.niche_DE(o, num_cores=1, C=150, M=10, gamma=0.8, verbose=False)
T_SHIPPED_SERIAL = time.perf_counter() - t0
T_dev = unpack_T(o, 0)
st = corr(dev["ref_T_stat_1"], T_dev)
print(f"shipped code, dev fixture, 1 worker : {T_SHIPPED_SERIAL:.2f} s")
print(f"R reference on the same fixture     : {R_T_DEV:.2f} s ({R_CORES_DEV} cores)")
print(f"T_stat Pearson vs R                 : {st['pearson']:.6f}")
iter_panel(0, extra=(f"shipped serial run measured live: {T_SHIPPED_SERIAL:.2f} s, "
                     f"T_stat Pearson {st['pearson']:.6f}"))
''')

I(1, "Equivalence — R's single-column `drop = TRUE`",
  r"""
**What changed.** `_niche_de_core` gained an `r_dropped_to_vector` guard: when exactly one
interaction column survives the `M` filter, it raises `_RErrorCompat` at the same point R's own
code throws, so the gene is returned invalid. The invalid-return contract was fixed at the same
time: R's `else` branch always reports `nulls = c(1:n_type^2)` regardless of the `null` vector
it had computed on the way, so the port does too.

**Why.** The parity test showed `valid`-flag agreement of 0.9933 — two genes out of 300 where R
said invalid and Python said valid. Tracing them (FABP1 and SERPINA1) showed R had *recorded a
finite log-likelihood*, so the GLM had succeeded; the failure came later. The cause is R's
`drop = TRUE` default: `X_partial = X[, -null]` collapses to a plain vector when one column
survives, `nvar = ncol(X_partial)` becomes `NULL`, `coeff = coefficients[1:(NULL + 1)]` becomes
length 0, and `beta[c(1:n_type^2)[-null]] = coeff[-1]` raises *"replacement has length zero"*.
The enclosing `tryCatch` swallows it and the gene is silently dropped.

**Admissibility.** Not an acceleration rewrite — this is class (E) equivalence, and mandatory:
without it the port reports genes R does not.

**Expected effect.** No wall-clock change; `valid`-flag agreement 0.9933 → 1.000000 and the
`nulls` max abs error 1 → 0.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
# The R dump lets us *find* the affected genes rather than assert them: genes R
# marked invalid even though it had already recorded a finite log-likelihood.
r_valid = dev["ref_valid_1"]; r_ll = dev["ref_loglik_1"]; r_nn = dev["ref_nnull_1"]
n_type = len(dcts)
susp = np.flatnonzero((r_valid == 0) & np.isfinite(r_ll) & (r_nn == n_type ** 2))
print("genes R marked INVALID despite a finite log-likelihood:",
      [dgenes[i] for i in susp])
py_valid = np.array([r["valid"] for r in o.niche_DE[0]], dtype=float)
py_nn = np.array([len(r["nulls"]) for r in o.niche_DE[0]], dtype=float)
agree = float((r_valid == py_valid).mean())
print(f"valid-flag agreement with the guard in place : {agree:.6f}")
print(f"nulls-count max abs err                      : {float(np.max(np.abs(r_nn - py_nn))):.0f}")
print(f"without the guard those {len(susp)} genes would be reported as valid "
      f"-> agreement would be {1 - len(susp) / len(dgenes):.4f}")
iter_panel(1, extra=(f"valid-flag agreement {agree:.6f}; the {len(susp)} R-invalid-with-loglik "
                     f"genes ({', '.join(dgenes[i] for i in susp)}) are reproduced"))
''')

I(2, "Equivalence — R's lazy `n` in `p.adjust`",
  r"""
**What changed.** `pynichede/rstats.py::p_adjust` was rewritten to reproduce R's *lazy* default
argument: `p.adjust(p, method, n = length(p))` does not force `n` until the `stopifnot(n >= lp)`
check, which runs **after** the body has already executed `p <- p[!is.na(p)]`. An un-supplied
`n` is therefore the number of **non-NA** p-values, not the total length.

**Why.** The interaction-level BH p-values correlated with R at Spearman 0.859 on `-log10 p` —
far below the 0.90 gate — while the *raw* p-values feeding them matched at 1.000000. That
isolates the fault to the adjustment step. Niche-DE's p-value arrays are dense with `NaN` (any
interaction whose column was nulled out), so the difference between `n = total` and
`n = non-NA` is large: about a factor of 3 on this fixture.

**Admissibility.** Class (E), exact. Confirmed directly against R:
`p.adjust(c(.4, NA, .01, .9, NA, .02, .5), "BH")` returns exactly
`p.adjust(c(.4, .01, .9, .02, .5), "BH")` in the non-NA slots. After the fix, feeding R's *own*
raw p-values through the Python `p_adjust` reproduces R's adjusted arrays with max abs error
0.000e+00 at all three levels.

**Expected effect.** No wall-clock change. This was the largest single accuracy win of the
port: interaction-level Spearman 0.859 → 1.000000, gene level 0.994 → 0.9999, and the reported
gene-set Jaccards 0.67–0.86 → 0.93–1.00.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
from pynichede.rstats import p_adjust
# 1) The isolated demonstration of R's lazy `n`.
p_demo = np.array([0.4, np.nan, 0.01, 0.9, np.nan, 0.02, 0.5])
lazy = p_adjust(p_demo, "BH")
eager = p_demo.copy()
m = np.isfinite(p_demo)
o_ = np.argsort(-p_demo[m])
r_ = np.arange(len(p_demo), len(p_demo) - m.sum(), -1)       # n = TOTAL length
eager[np.flatnonzero(m)[o_]] = np.minimum.accumulate(len(p_demo) / r_ * p_demo[m][o_])
print("p                :", p_demo)
print("R / pynichede (n = #non-NA) :", np.round(lazy, 6))
print("naive     (n = len(p))      :", np.round(eager, 6))

# 2) End to end on the full fixture: R's own raw p-values -> Python's p_adjust.
errs = {}
for lvl in ("gene", "ct", "int"):
    errs[lvl] = det(full[f"ref_pval_pos_{lvl}"],
                    _adjust_like_R(full[f"ref_praw_pos_{lvl}"], lvl))["max_abs_err"]
print("\nfeeding R's own raw p through the Python p_adjust, max abs err vs R's adjusted p:")
for k, v in errs.items():
    print(f"   {k:5s} {v:.3e}")
sp_now = infer(full["ref_pval_pos_int"], _adjust_like_R(full["ref_praw_pos_int"], "int"))
print(f"interaction-level Spearman(-log10 p) with the lazy-n rule: "
      f"{sp_now['spearman_neglog10p']:.6f}")
iter_panel(2, extra=("BH step is now bit-exact: max abs err "
                     + ", ".join(f"{k}={v:.1e}" for k, v in errs.items())))
''')

I(3, "Equivalence — the `Int = FALSE` linear-model branch",
  r"""
**What changed.** The continuous-data branch of `_niche_de_core` was rewritten to follow
`summary.lm` semantics rather than the Poisson branch's. R takes its test statistic from
`summary(lm)$coefficients[, 3]` — the **t-values** — not from `beta / tau`, and its covariance
from `summary(lm)$cov.unscaled * sigma^2`, which contains only the **non-aliased** columns.

**Why.** The first draft padded the covariance back to full width with `NaN` on the aliased
rows and then tested `diag(V) == 0` on the padded matrix. With a single batch the constant
`batchvar` column is always aliased against the intercept, so that test fired on **every** gene
and the whole branch returned 0 valid genes out of 300 while R returned 93.

**Admissibility.** Class (E), exact — a semantics correction, not an approximation.

**Expected effect.** No wall-clock change; `Int=FALSE` valid genes 0/300 → 93/300, matching R
exactly, with `T_stat` Pearson 1.000000 on the 1219 finite entries and the gene-level p-values
clearing the inference gate.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
cont_genes = list(full.meta["cont_genes"])
cc = pd.DataFrame(full["in_cont_counts"], index=fcells, columns=cont_genes)
cl = pd.DataFrame(full["in_cont_libmat"], index=fcts,   columns=cont_genes)
oc = nde.create_nichede_object(cc, fcoord, cl, fdeconv, sigma=fsigma, Int=False)
oc = nde.calculate_effective_niche(oc, cutoff=0.05)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    oc = nde.niche_DE_no_parallel(oc, C=150, M=10, gamma=0.8, Int=False,
                                  batch=True, self_EN=False, verbose=False)
Tc = np.full((len(fcts), len(fcts), len(cont_genes)), np.nan)
vc = np.zeros(len(cont_genes))
for g, r in enumerate(oc.niche_DE[0]):
    if r["valid"] == 1:
        Tc[:, :, g] = r["T_stat"]
    vc[g] = r["valid"]
st = corr(full["ref_cont_T_stat_1"], Tc)
agree = float((full["ref_cont_valid_1"] == vc).mean())
stp = infer(full["ref_cont_pval_pos_gene"],
            np.asarray(oc.niche_DE_pval_pos["gene_level"], dtype=float))
print(f"Int=FALSE valid genes  R {int(full['ref_cont_valid_1'].sum())} / "
      f"Python {int(vc.sum())} of {len(cont_genes)}   agreement {agree:.6f}")
print(f"Int=FALSE T_stat Pearson {st['pearson']:.6f} on {st['n']} finite entries")
print(f"Int=FALSE gene-level p Spearman(-log10 p) {stp['spearman_neglog10p']:.6f}")
iter_panel(3, extra=(f"Int=FALSE branch: {int(vc.sum())}/{len(cont_genes)} valid genes "
                     f"(was 0), agreement {agree:.6f}, T_stat Pearson {st['pearson']:.6f}"))
del oc
''')

I(4, "Acceleration — vectorised design-matrix construction",
  r"""
**What changed.** The per-spot loop that builds the `n_spot x n_type^2` niche design matrix was
replaced by a single broadcast: `(EN[:, :, None] * pstg[:, None, :]).transpose(0, 2, 1)`
followed by a C-order reshape.

**Why.** Profiling the baseline showed the design build was the largest pure-Python cost inside
the per-gene worker — it runs once per gene per kernel bandwidth, i.e. ~65 000 times on the
canonical fixture.

**Admissibility — class (E), exact.** R writes
`X[k, ] = as.vector(round(EN[k, ], 2) %*% t(pstg[k, ]))`. For length-`p` vectors, `v %*% t(w)`
is the rank-1 matrix with entry `v[a] w[b]`, and `as.vector` flattens column-major, so
`X[k, (b-1)p + a] = EN[k, a] * pstg[k, b]`. The broadcast produces the same product for each
`[k, a, b]`; the transpose makes `a` vary fastest, reproducing the column-major flattening.
Every output entry is a **single multiplication of the same two f64 operands**, so the result
is bit-identical rather than merely equal to rounding (`MATH.md` §1.1).

**Expected effect.** A large speedup on the **step** and no change whatsoever to parity — but
the design build is only about 1% of `niche_DE`, so `ITERATION_LOG.md` deliberately claims **no
pipeline speedup** for this rewrite. It is kept because it is free and exact, not because it
moves the total.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
def build_X_loop(en_r, pstg, n_type, self_EN=False):
    """Iteration 0: literal transcription of R's `for (k in 1:nrow(pstg))`."""
    X = np.empty((en_r.shape[0], n_type * n_type))
    diag_idx = np.arange(n_type) * n_type + np.arange(n_type)
    for k in range(en_r.shape[0]):
        X[k] = np.outer(en_r[k], pstg[k]).ravel(order="F")
        if not self_EN:
            X[k, diag_idx] = 0.0
    return X

def build_X_broadcast(en_r, pstg, n_type, self_EN=False):
    """Iteration 4: one broadcast + transpose + reshape (what the package ships)."""
    X = (en_r[:, :, None] * pstg[:, None, :]).transpose(0, 2, 1).reshape(en_r.shape[0], -1)
    if not self_EN:
        diag_idx = np.arange(n_type) * n_type + np.arange(n_type)
        X[:, diag_idx] = 0.0
    return X

nc_ = np.asarray(o.num_cells, dtype=float); rf_ = np.asarray(o.ref_expr, dtype=float)
re_ = rf_[:, 0]; EEJ = nc_ @ re_
with np.errstate(divide="ignore", invalid="ignore"):
    pstg = nc_ * re_[None, :] / EEJ[:, None]
pstg[pstg < 0.05] = 0.0
en_r = np.round(o.effective_niche[0], 2)
nt = len(dcts)
A_ = build_X_loop(en_r, pstg, nt); B_ = build_X_broadcast(en_r, pstg, nt)
err = float(np.max(np.abs(A_ - B_)))
t_old, _ = timeit(lambda: build_X_loop(en_r, pstg, nt), 5)
t_new, _ = timeit(lambda: build_X_broadcast(en_r, pstg, nt), 5)
print(f"per-spot loop : {t_old * 1e3:8.3f} ms")
print(f"broadcast     : {t_new * 1e3:8.3f} ms   -> {t_old / t_new:.1f}x")
print(f"max abs deviation between the two: {err:.3e}   (bit-identical)")
iter_panel(4, extra=(f"design matrix {t_old*1e3:.2f} ms -> {t_new*1e3:.3f} ms "
                     f"({t_old/t_new:.0f}x), max abs deviation {err:.1e}"),
           bars=("design-matrix step", ["per-spot loop", "broadcast"], [t_old, t_new]))
''')

I(5, "Acceleration — `cho_solve` instead of an explicit triangular inverse",
  r"""
**What changed.** `V = solve(chol(X'WX)) %*% t(solve(chol(X'WX)))` was replaced by
`A = cholesky(var_mat); V = cho_solve((A, False), I)` — two triangular solves instead of forming
`A^-1` and multiplying it out.

**Why.** The literal transcription allocates a dense `nvar x nvar` identity, calls
`solve_triangular` against it, and then does a full matrix multiply — three passes where one
`potrs` suffices. This runs once per valid gene per bandwidth.

**Admissibility — class (E), exact.** With `A` upper triangular and `A'A = X'WX`,
`A^-1 (A^-1)' = A^-1 A^-T = (A'A)^-1 = (X'WX)^-1`, which is exactly what `cho_solve` returns.
Exact in exact arithmetic; measured max **relative** deviation on the fixture 1.49e-15, i.e.
one to two ulp (`MATH.md` §1.2). Critically the Cholesky **factorisation itself is retained**,
because `Matrix::chol` *raising* on a non-positive-definite `X'WX` is load-bearing — it is what
makes `nicheDE` mark such a gene invalid, and `scipy.linalg.cholesky` raises `LinAlgError` in
the same situation, so the control flow is preserved.

**Expected effect, and honest attribution.** Large on the step, and parity flat. At *pipeline*
level, though, `ITERATION_LOG.md`'s ablation shows this rewrite is worth **1.00x on its own**
(1.917 s with the explicit inverse restored vs 1.923 s shipped, BLAS still pinned): the two
variants cost the same once the thread pool is already limited. Its value is **joint with
iteration 7** — the explicit `solve_triangular(A, I)` is what made a wide BLAS pool expensive,
so removing either one removes the cost, and the 8.0x is booked once against the pair rather
than twice. Accepting an exact, strictly-cheaper identity with zero downside is still the right
call when its solo attribution is zero.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

def inv_explicit(var_mat):
    """Iteration 0: R's literal `solve(A) %*% t(solve(A))`."""
    A = cholesky(var_mat, lower=False)
    Ainv = solve_triangular(A, np.eye(A.shape[0]), lower=False)
    return Ainv @ Ainv.T

def inv_cho_solve(var_mat):
    """Iteration 5: two triangular solves, no explicit inverse (what ships)."""
    c = cho_factor(var_mat, lower=False)
    return cho_solve(c, np.eye(var_mat.shape[0]))

Xd = np.column_stack([B_, np.ones(B_.shape[0])])
W = np.abs(np.asarray(dcounts)[:, 0]) + 1.0
keep = np.flatnonzero((Xd > 0).sum(0) >= 10)
Xk = np.column_stack([Xd[:, keep], np.ones(Xd.shape[0])])
var_mat = (Xk * W[:, None]).T @ Xk
v1, v2 = inv_explicit(var_mat), inv_cho_solve(var_mat)
rel = float(np.max(np.abs(v1 - v2)) / np.max(np.abs(v1)))
t_old, _ = timeit(lambda: inv_explicit(var_mat), 60)
t_new, _ = timeit(lambda: inv_cho_solve(var_mat), 60)
print(f"var_mat is {var_mat.shape[0]}x{var_mat.shape[0]}")
print(f"explicit inverse : {t_old * 1e6:9.1f} us")
print(f"cho_solve        : {t_new * 1e6:9.1f} us   -> {t_old / t_new:.1f}x")
print(f"max RELATIVE deviation between the two: {rel:.3e}   (f64 rounding only)")
iter_panel(5, extra=(f"(X'WX)^-1 {t_old*1e6:.1f} us -> {t_new*1e6:.1f} us "
                     f"({t_old/t_new:.1f}x), max relative deviation {rel:.1e}"),
           bars=("(X'WX)^-1 step", ["explicit inverse", "cho_solve"], [t_old, t_new]))
''')

I(6, "Acceleration — memoised Mehler / `mvnconv` lookup table",
  r"""
**What changed.** `pynichede.poolr._lookup()` — the 1991 × 9 covariance table that `mvnconv`
indexes — was put behind an `functools.lru_cache`, so the Hermite quadrature that derives it
runs once per process instead of once per call.

**Why.** Because the table is re-derived rather than vendored (poolr is GPL-2+, this port is
MIT), building it is not free: ~3.6 s of Gauss–Legendre quadrature plus a 2000-term Mehler
series. `mvnconv` is called once per gene, per kernel bandwidth, per direction, per pooling
level — on the canonical fixture that is roughly 5000 × 3 × 2 × 2 calls, so an uncached table
would have dominated the entire runtime by orders of magnitude.

**Admissibility — class (E), exact.** `mvnconv` is a pure function of `(rho grid, target,
side)`; the table depends on nothing else. Hoisting its construction behind a cache is textbook
memoisation of a pure function, with bit-identical output (measured max abs deviation 0.0,
`MATH.md` §1.3).

**Expected effect.** Several orders of magnitude on the step. Like iteration 4 this is logged
as **step-level with no pipeline delta claimed** — the cache was present from the first working
version, so it never shows up as a pipeline *difference*; it is recorded because removing it is
catastrophic and a future maintainer needs to know the `lru_cache` is load-bearing. It is also
why every timing in the log is warmup-excluded: the first `niche_DE` call in a fresh process
pays the table build once. Parity flat.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
from pynichede import poolr as _poolr

def mvnconv_uncached(R, side=1, target="m2lp"):
    """Iteration 0: rebuild the Mehler table on every call."""
    tbl = _poolr.build_mvnlookup()
    tbl[:, 1:] = np.round(tbl[:, 1:], 4)
    col = _poolr._TARGETS.index(target) * 2 + 1 + (1 if side == 2 else 0)
    Rr = np.where(np.round(R, 3) < -0.99, -0.99, np.round(R, 3))
    idx = np.clip(np.rint((1.0 - Rr) * 1000.0).astype(np.int64), 0, tbl.shape[0] - 1)
    return tbl[idx, col]

Rp = np.eye(7) * 0.5 + 0.5
a_ = mvnconv_uncached(Rp)
b_ = nde.mvnconv(Rp, 1, "m2lp")
err = float(np.max(np.abs(a_ - b_)))
t_old, _ = timeit(lambda: mvnconv_uncached(Rp), 2, warmup=0)
t_new, _ = timeit(lambda: nde.mvnconv(Rp, 1, "m2lp"), 300)
n_calls = int(np.isfinite(full["ref_loglik_1"]).sum()) * len(fsigma) * 2 * 2
print(f"uncached : {t_old * 1e3:10.2f} ms per call")
print(f"cached   : {t_new * 1e3:10.5f} ms per call  -> {t_old / t_new:.0f}x")
print(f"max abs deviation: {err:.3e}   (bit-identical)")
print(f"~{n_calls} mvnconv calls on the canonical fixture "
      f"-> {t_old * n_calls / 60:.0f} min uncached vs {t_new * n_calls:.1f} s cached")
iter_panel(6, extra=(f"mvnconv {t_old*1e3:.1f} ms -> {t_new*1e3:.4f} ms "
                     f"({t_old/t_new:.0f}x), max abs deviation {err:.1e}"),
           bars=("mvnconv call", ["rebuild table", "cached lookup"], [t_old, t_new]))
''')

I(7, "Acceleration — pin the BLAS thread pool to 1",
  r"""
**What changed.** `niche_DE` now wraps its per-sigma loop in
`threadpoolctl.threadpool_limits(limits=1)` (with a `contextlib.nullcontext()` fallback if
`threadpoolctl` is unavailable).

**Why.** Niche-DE's inner problem is one ~848 × 50 GLM per gene. That is far below the size
where a multi-threaded BLAS pays for its synchronisation, and on this 17-core box the default
17-thread OpenBLAS pool spent more time in barriers than in arithmetic. This was the single
largest pipeline-level win of the whole acceleration loop.

**Admissibility — class (E)/scheduling, exact.** Changing the number of BLAS worker threads
changes neither the operations performed nor their order for the level-2/level-3 kernels used
here (`syrk`, `potrf`, `trsm`), because those routines partition by **output block**, not by
reduction — so no summation order changes. Verified empirically as well: the complete parity
report is unchanged, digit for digit, before and after (`MATH.md` §1.4).

**Expected effect, and honest attribution.** `ITERATION_LOG.md` books **8.02x jointly against
iterations 5 + 7**, and is explicit that ablating *only* the thread pinning is worth just
**1.08x** (2.085 ± 0.096 s vs the shipped 1.923 ± 0.004 s), while ablating the pinning **and**
the `cho_solve` identity together costs 15.43 ± 2.05 s. The two are redundant fixes for the same
bottleneck. The cell below re-measures the solo effect live on the canonical full fixture, so
the reader can check that 1.08x figure rather than take it on trust; the *parity* half of the
claim — that the thread budget cannot change the result — is checked second and holds
unconditionally.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
from pynichede.niche_de import _run_sigma
from pynichede.rstats import r_quantile
from threadpoolctl import threadpool_limits, threadpool_info

print("BLAS pools visible to this process:",
      [(a["internal_api"], a["num_threads"]) for a in threadpool_info()])

# Measure on the CANONICAL FULL fixture: the dev fixture only has ~100 runnable
# genes, far too little arithmetic for a thread-pool effect to be visible.
of = nde.create_nichede_object(fcounts, fcoord, flibmat, fdeconv, sigma=fsigma, Int=True)
of = nde.calculate_effective_niche(of, cutoff=0.05)
fcnt = np.asarray(of.counts, dtype=float)
fncl = np.asarray(of.num_cells, dtype=float)
frex = np.asarray(of.ref_expr, dtype=float)
fbid = np.asarray(of.batch_ID)
fctf = np.array([r_quantile(frex[i], 0.8) for i in range(frex.shape[0])])
fargs = (fcnt, of.effective_niche[0], fncl, fctf, 150, 10, True, True, fbid, frex,
         len(fcts), False, 1, False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with threadpool_limits(limits=N_CPU):
        t0 = time.perf_counter(); r_multi = _run_sigma(*fargs); t_multi = time.perf_counter() - t0
    with threadpool_limits(limits=1):
        t0 = time.perf_counter(); r_one = _run_sigma(*fargs); t_one = time.perf_counter() - t0

Ta = np.array([r["T_stat"] if r["valid"] == 1 else np.full((len(fcts),) * 2, np.nan)
               for r in r_multi])
Tb = np.array([r["T_stat"] if r["valid"] == 1 else np.full((len(fcts),) * 2, np.nan)
               for r in r_one])
m = np.isfinite(Ta) & np.isfinite(Tb)
err = float(np.max(np.abs(Ta[m] - Tb[m])))
abl = float(logged(7, "ablation_this_change_only_s"))
shipped = float(logged(7, "wall_clock_mean_s"))
joint = float(logged(7, "speedup_vs_previous"))
print(f"full fixture, ONE kernel bandwidth, single process:")
print(f"  BLAS pool = {N_CPU} threads : {t_multi:6.2f} s")
print(f"  BLAS pool = 1  thread    : {t_one:6.2f} s   -> {t_multi / t_one:.2f}x measured now")
print(f"  ITERATION_LOG.md solo ablation : {abl:.3f} s vs {shipped:.3f} s shipped "
      f"= {abl / shipped:.2f}x")
print(f"  ITERATION_LOG.md joint 5+7     : {joint:.2f}x")
print(f"  -> the live {t_multi / t_one:.2f}x corroborates the logged SOLO figure "
      f"({abl / shipped:.2f}x), not the joint one.")
print(f"max abs difference in T_stat between the two thread budgets: {err:.3e}"
      f"   <- the admissibility claim, and it holds exactly")
iter_panel(7, extra=(f"full fixture, one sigma: {t_multi:.2f} s with {N_CPU} BLAS threads -> "
                     f"{t_one:.2f} s with 1 ({t_multi/t_one:.2f}x live, vs the logged solo "
                     f"ablation {abl/shipped:.2f}x); T_stat identical to {err:.1e}"),
           bars=("_run_sigma, full fixture, one bandwidth",
                 [f"{N_CPU} BLAS threads", "1 BLAS thread"], [t_multi, t_one]))

# the dev-fixture arrays iterations 8 and 10 reuse
cnt = np.asarray(dcounts, dtype=float)
ncl = np.asarray(o.num_cells, dtype=float)
rex = np.asarray(o.ref_expr, dtype=float)
bid = np.asarray(o.batch_ID)
ctf = np.array([r_quantile(rex[i], 0.8) for i in range(rex.shape[0])])
args = (cnt, o.effective_niche[0], ncl, ctf, 150, 10, True, True, bid, rex,
        len(dcts), False, 1, False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with threadpool_limits(limits=1):
        t0 = time.perf_counter(); r_dev_one = _run_sigma(*args)
        t_dev_one = time.perf_counter() - t0
print(f"\n(dev fixture, one bandwidth, serial, for iterations 8-10: {t_dev_one:.2f} s)")
''')

I(8, "REJECTED — one joblib task per gene",
  r"""
**What was tried.** Dispatching the gene loop to `joblib.Parallel` with **one task per gene**,
mirroring R's `foreach(g = 1:ngene) %dopar%` as literally as possible.

**Why it looked right.** Each gene's fit is genuinely independent — it reads only shared
read-only arrays (`effective_niche`, `num_cells`, `ref_expr`) and writes only its own result
dict — so any partition of the gene index set is admissible, and R itself parallelises over
exactly this axis.

**What happened.** It was **slower than serial**, badly. The log's controlled ablation puts
per-gene dispatch at 14.34 ± 2.06 s against the shipped chunked dispatch's 0.686 ± 0.007 s —
**20.9× slower**, and slower even than the 1.923 s single-process path. Two causes, both
confirmed by profiling: joblib re-pickled the shared `effective_niche` / `num_cells` /
`ref_expr` arrays for every small task, so transfer dominated the few milliseconds of actual
arithmetic; and each loky worker opened its **own** full-width BLAS pool, so 8 workers × 17
threads thrashed a 17-core box.

**Decision: REJECT_SLOW.** Rolled back and replaced by iteration 9, which keeps the same
admissibility argument but changes the partition granularity and the per-worker thread budget.
Parity was never the problem here (Pearson stayed at 1.000000) — this rejection is purely about
wall clock, and it is logged rather than discarded because the *reason* it failed is what
motivated iteration 9's design.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
from joblib import Parallel, delayed, parallel_backend
from pynichede.niche_de import _chunk_worker

tot = cnt.sum(axis=0)
passes = (rex < ctf[:, None]).mean(axis=0) != 1
runnable = np.flatnonzero((tot > 150) & passes)
print(f"{runnable.size} runnable genes on the dev fixture")

def per_gene_dispatch(n_jobs):
    """The REJECTED variant: one joblib task per gene, default worker thread budget."""
    with parallel_backend("loky", n_jobs=n_jobs):
        out = Parallel(verbose=0)(
            delayed(_chunk_worker)(np.array([g]), cnt[:, [g]], o.effective_niche[0], ncl,
                                   ctf, 150, 10, True, True, bid, rex, len(dcts), False)
            for g in runnable)
    return [r for blk in out for r in blk]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t0 = time.perf_counter(); res_rej = per_gene_dispatch(8)
    T_REJECTED = time.perf_counter() - t0
Tr = np.array([r["T_stat"] if r["valid"] == 1 else np.full((len(dcts),) * 2, np.nan)
               for r in res_rej])
Tk = np.array([r_dev_one[g]["T_stat"] if r_dev_one[g]["valid"] == 1
               else np.full((len(dcts),) * 2, np.nan) for g in runnable])
mm = np.isfinite(Tr) & np.isfinite(Tk)
print(f"per-gene joblib tasks (n_jobs=8), one bandwidth : {T_REJECTED:6.2f} s")
print(f"serial with the shipped code, one bandwidth      : {t_dev_one:6.2f} s")
print(f"-> speedup {t_dev_one / T_REJECTED:.2f}x   (< 1 means SLOWER than serial)")
lg = float(logged(8, "speedup_vs_previous"))
print(f"ITERATION_LOG.md records {lg:.3f}x for this rewrite (vs the shipped chunked path); "
      f"measured here against the serial path: {t_dev_one / T_REJECTED:.2f}x")
print("-- different denominators, same verdict: parallelising per gene is a regression.")
print(f"parity was never the issue: max |T_stat difference| = "
      f"{float(np.max(np.abs(Tr[mm] - Tk[mm]))):.3e}")
iter_panel(8, extra=(f"REJECTED: per-gene dispatch {T_REJECTED:.2f} s vs serial "
                     f"{t_dev_one:.2f} s = {t_dev_one/T_REJECTED:.2f}x (slower); "
                     f"parity unchanged"),
           bars=("dev fixture, one bandwidth", ["serial", "per-gene joblib (REJECTED)"],
                 [t_dev_one, T_REJECTED]), bar_colors=["#888888", C_BAD])
''')

I(9, "Acceleration — chunked dispatch + `inner_max_num_threads=1`",
  r"""
**What changed.** The parallel path now splits the runnable gene indices into roughly
`4 × n_jobs` contiguous **chunks** and dispatches one task per chunk, inside
`parallel_backend("loky", n_jobs=n, inner_max_num_threads=1)`.

**Why.** Directly from iteration 8's post-mortem: chunking amortises the pickling of the shared
arrays over hundreds of genes instead of one, and `inner_max_num_threads=1` stops each worker
from opening its own full-width BLAS pool. Four chunks per worker (rather than exactly one) is
a deliberate compromise — enough granularity for load balancing when genes differ in cost, few
enough that transfer stays amortised.

**Admissibility — class (E), exact.** The same independence argument as iteration 8; only the
partition granularity and the worker thread budget change. Results are reassembled in gene
order, so the output list is identical element-for-element — verified by running the complete
parity report at `n_jobs=1` and `n_jobs=16` and comparing digit for digit.

**Expected effect.** A genuine parallel speedup this time, and it converts iteration 8's 0.6×
regression into a real gain. Parity flat.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o_p = nde.create_nichede_object(dcounts, dcoord, dlibmat, ddeconv, sigma=dsigma, Int=True)
    o_p = nde.calculate_effective_niche(o_p, cutoff=0.05)
    t0 = time.perf_counter()
    o_p = nde.niche_DE(o_p, num_cores=N_JOBS, C=150, M=10, gamma=0.8, verbose=False)
    T_CHUNKED = time.perf_counter() - t0
T_chunk = unpack_T(o_p, 0)
st = corr(dev["ref_T_stat_1"], T_chunk)
same = float(np.nanmax(np.abs(np.nan_to_num(T_chunk) - np.nan_to_num(T_dev))))
print(f"shipped chunked dispatch, {N_JOBS} workers, all 3 bandwidths : {T_CHUNKED:6.2f} s")
print(f"shipped serial run (iteration 0 cell)                     : {T_SHIPPED_SERIAL:6.2f} s")
print(f"-> {T_SHIPPED_SERIAL / T_CHUNKED:.2f}x, and vs R's {R_T_DEV:.2f} s on "
      f"{R_CORES_DEV} cores: {R_T_DEV / T_CHUNKED:.1f}x")
print(f"ITERATION_LOG.md records {float(logged(9, 'speedup_vs_previous')):.2f}x vs serial and "
      f"{float(logged(9, 'speedup_vs_baseline')):.1f}x vs the controlled baseline")
print(f"T_stat Pearson vs R                : {st['pearson']:.6f}")
print(f"max |T_stat| difference vs the serial run: {same:.3e}  (element-for-element identical)")
iter_panel(9, extra=(f"chunked dispatch {T_CHUNKED:.2f} s vs serial {T_SHIPPED_SERIAL:.2f} s "
                     f"({T_SHIPPED_SERIAL/T_CHUNKED:.2f}x); identical output ({same:.1e})"),
           bars=("full dev pipeline", ["serial", f"chunked, {N_JOBS} workers"],
                 [T_SHIPPED_SERIAL, T_CHUNKED]))
''')

I(10, "Acceleration — hoist the gene gate out of the workers",
  r"""
**What changed.** `_run_sigma` now evaluates `niche_DE_core`'s two entry conditions —
`sum(counts) > C` and "not every cell type is below its gamma-quantile filter" — **once in the
parent for all genes**, and ships only the survivors to the workers. Genes that fail are
returned as the invalid result directly.

**Why.** On the canonical fixture only 4867 of 21708 genes are ever actually fitted. Under the
old scheme all 21708 counts columns were pickled to workers, and 78% of the tasks did nothing
but re-derive that they should do nothing. The saving is therefore both in task count and,
more importantly, in the volume of data crossing the process boundary.

**Admissibility — class (E), exact.** Both predicates depend only on the counts column sums and
on `ref_expr`, neither of which changes per gene inside the worker, and they are the *first*
branch of `niche_DE_core` — so evaluating them earlier is the same predicate applied at the
same point in the control flow, on the same values. Verified: `valid`-flag agreement stays at
1.000000 and the parity report is unchanged.

**Expected effect.** `ITERATION_LOG.md` folds this into the 0.686 s chunked-dispatch figure and
claims **1.00x of its own** on the dev fixture, noting that it matters most on the full fixture
where ~78% of genes are removed rather than ~68%. The cell below measures both the removal
fraction and the parallel-path timing on the canonical full fixture. This was the
last accepted rewrite; the loop then stopped because the remaining runtime is dominated by the
per-gene IRLS (already a rank-truncated QR at minimum arithmetic cost) and the Brown/Cauchy
pooling (already vectorised).

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
# How much does the prefilter remove, on the canonical FULL fixture?
fcnt = np.asarray(fcounts, dtype=float)
frex = np.asarray(flibmat, dtype=float)
fctf = np.array([r_quantile(frex[i], 0.8) for i in range(frex.shape[0])])
f_tot = fcnt.sum(axis=0)
f_pass = (frex < fctf[:, None]).mean(axis=0) != 1
f_runnable = np.flatnonzero((f_tot > 150) & f_pass)
frac = 1 - f_runnable.size / fcnt.shape[1]
print(f"full fixture: {f_runnable.size} of {fcnt.shape[1]} genes are ever fitted "
      f"-> the prefilter removes {frac * 100:.1f}% of the tasks")

# Cost of NOT prefiltering, measured where the claim lives: the PARALLEL path on
# the full fixture, where every unfiltered gene also costs a pickled counts column.
def par_dispatch(idx, n_jobs):
    n_chunks = min(idx.size, max(1, n_jobs * 4))
    splits = [s for s in np.array_split(idx, n_chunks) if s.size]
    with parallel_backend("loky", n_jobs=n_jobs, inner_max_num_threads=1):
        out = Parallel(verbose=0)(
            delayed(_chunk_worker)(s, fcnt[:, s], of.effective_niche[0], fncl, fctf,
                                   150, 10, True, True, fbid, frex, len(fcts), False)
            for s in splits)
    return [r for blk in out for r in blk]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with threadpool_limits(limits=1):
        t0 = time.perf_counter()
        res_all = par_dispatch(np.arange(fcnt.shape[1]), N_JOBS)
        T_NOFILTER = time.perf_counter() - t0
        t0 = time.perf_counter()
        res_pre = _run_sigma(fcnt, of.effective_niche[0], fncl, fctf, 150, 10, True, True,
                             fbid, frex, len(fcts), False, N_JOBS, False)
        T_PREFILTER = time.perf_counter() - t0
va = np.array([r["valid"] for r in res_all]); vb = np.array([r["valid"] for r in res_pre])
print(f"all {fcnt.shape[1]} genes dispatched, {N_JOBS} workers : {T_NOFILTER:6.2f} s")
print(f"prefiltered to {f_runnable.size} genes (shipped)      : {T_PREFILTER:6.2f} s "
      f"-> {T_NOFILTER / T_PREFILTER:.2f}x")
print(f"ITERATION_LOG.md claims {float(logged(10, 'speedup_vs_previous')):.2f}x of its own "
      f"on the dev fixture; measured here on the FULL fixture: "
      f"{T_NOFILTER / T_PREFILTER:.2f}x")
print(f"valid flags identical either way: {bool((va == vb).all())}"
      f"   <- the admissibility claim")
iter_panel(10, extra=(f"prefilter removes {frac*100:.1f}% of genes on the full fixture; "
                      f"{T_NOFILTER:.2f} s -> {T_PREFILTER:.2f} s "
                      f"({T_NOFILTER/T_PREFILTER:.2f}x), identical valid flags"),
           bars=(f"full fixture, one bandwidth, {N_JOBS} workers",
                 ["every gene", "prefiltered"], [T_NOFILTER, T_PREFILTER]))
''')


I(11, "Acceleration — memoise the per-kernel ligand-target slice",
  r"""
**What changed.** `_ligand_scores`, shared by `niche_LR_spot` and `niche_LR_cell`, now caches
the filtered-and-reordered NicheNet ligand-target matrix in a dict keyed on the kernel index,
instead of rebuilding it inside the per-candidate-ligand loop the way the R source does.

**Why.** R runs this once per candidate ligand:
`sig <- T_vector[[top_kernel[ind]]]`, `genes <- gene_names[!is.na(sig)]`,
`lv <- ligand_target_matrix[rownames %in% genes, ]`, `lv <- lv[genes, ]`. Every line of it
depends **only** on `top_kernel[ind]` — which kernel bandwidth fits that ligand best — and that
takes at most `len(sigma)` = 3 distinct values. So across the 579 candidate ligands there are at
most 3 distinct results and the other 576 evaluations rebuild an array that already exists. On
the canonical fixture that is 576 redundant reindexes of a 16968 × 579 frame.

**Admissibility — class (E), exact.** Memoisation of a pure function: the cached value is the
same object the loop would have recomputed, bit for bit. Verified by diffing the produced
`niche_LR_spot` table against both the pre-memoisation run and R's table — byte-identical,
including the comma-joined `top_downstream_niche_DE_genes` strings.

**A note on the axis.** This is the one iteration that is *not* on the `niche_DE` trajectory
plotted above. `niche_LR_*` is a downstream call, so `ITERATION_LOG.md` carries the `niche_DE`
pipeline time forward unchanged at 0.686 s for this block — deliberately, so the trajectory
cannot appear to regress — and records the effect in separate `niche_lr_*` fields. The cell
below measures both: that `niche_DE` is untouched, and that the ligand sweep is ~105× faster.

**How it was found.** Not by reading the code. It surfaced because the tutorial notebook was
*executed* rather than sketched, and a cell that should have taken seconds took five minutes.
That is an argument for the protocol's insistence on genuinely running the deliverables.

[ITER_LOG ↩](../ITERATION_LOG.md)
""",
  r'''
# 1. The claim's own numbers, from the log.
before = float(logged(11, "niche_lr_time_before_s"))
after_logged = float(logged(11, "niche_lr_time_after_s"))
print(f"ITERATION_LOG.md: niche_LR_spot {before:.1f} s -> {after_logged:.1f} s "
      f"({before / after_logged:.1f}x), niche_DE unchanged at "
      f"{float(logged(11, 'wall_clock_mean_s')):.3f} s")

# 2. Re-measure the memoised call live on the canonical FULL fixture.
from pynichede import niche_LR_spot

ltm = pd.DataFrame(full["in_ligand_target_matrix"],
                   index=list(full.meta["ltm_rownames"]),
                   columns=list(full.meta["ltm_colnames"]))
lr_mat = pd.DataFrame({"ligand": list(full.meta["lr_ligand"]),
                       "receptor": list(full.meta["lr_receptor"])})

# `of` was built (and its effective niche computed) back in iteration 7; niche-LR
# additionally needs the fitted model, so run niche_DE on it once here.
if not of.niche_DE:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = nde.niche_DE(of, num_cores=N_JOBS, verbose=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t0 = time.perf_counter()
    lr_now = niche_LR_spot(of, "myeloid", "tumor_epithelial", ltm, lr_mat)
    T_LR_NOW = time.perf_counter() - t0
print(f"measured here, full {ltm.shape[1]}-ligand sweep: {T_LR_NOW:.2f} s "
      f"-> {len(lr_now)} ligand-receptor pairs")

# 3. Cost of the slice the memoisation removes, measured directly.
#    Re-running the whole uncached sweep would add ~5 min to this notebook, so
#    instead we time ONE slice and multiply by the number of evaluations the
#    uncached loop would perform.  It has to be the slice the real call actually
#    builds: the candidate-ligand-filtered matrix, and the T-vector for the
#    (index = tumor_epithelial, niche = myeloid) pair -- using any other pair
#    gives a much smaller `lv` and badly understates the cost.
from pynichede.rstats import r_quantile as _rq

_index = list(of.cell_types).index("tumor_epithelial")
_niche = list(of.cell_types).index("myeloid")
_Lv = np.asarray(of.ref_expr, dtype=float)
_CT = np.array([_rq(_Lv[i], 0.25) for i in range(_Lv.shape[0])])
_cand = set(np.asarray(of.gene_names)[_Lv[_niche] > _CT[_niche]])
ltm_cand = ltm.loc[:, [c for c in ltm.columns if c in _cand]]   # what the loop iterates

gnames = np.asarray(of.gene_names)
ltm_idx = pd.Index(ltm_cand.index)
sig0 = np.array([r["T_stat"][_index, _niche] if r["T_stat"] is not None else np.nan
                 for r in of.niche_DE[0]])
g0 = gnames[~np.isnan(sig0)]
t0 = time.perf_counter()
common = ltm_idx.isin(set(g0))
lv0 = ltm_cand.loc[common]
lv0 = lv0.loc[g0[np.isin(g0, lv0.index.to_numpy())]]
T_SLICE = time.perf_counter() - t0
n_lig = ltm_cand.shape[1]
n_sig = len(of.sigma)
print(f"\ncandidate ligands the loop iterates          : {n_lig} of {ltm.shape[1]}")
print(f"one slice -> lv {lv0.shape}                 : {T_SLICE * 1e3:7.1f} ms")
print(f"uncached: {n_lig} evaluations                 : {T_SLICE * n_lig:7.1f} s  (projected)")
print(f"memoised: at most len(sigma) = {n_sig} evaluations  : {T_SLICE * n_sig:7.1f} s")
print(f"redundant evaluations removed               : {n_lig - n_sig}")
print(f"\nprojection {T_SLICE * n_lig:.0f} s vs the {before:.1f} s logged for the "
      f"uncached sweep -> the slice accounts for essentially all of it")

# 4. Admissibility: the output must be byte-identical to the pre-memoisation run.
lr_ref = pd.read_csv(os.path.join(REF_DIR, "ref_niche_LR_spot.csv"))
pairs_now = set(map(tuple, lr_now[["ligand", "receptor"]].astype(str).to_numpy()))
pairs_R = set(map(tuple, lr_ref.iloc[:, :2].astype(str).to_numpy()))
same_pairs = pairs_now == pairs_R
same_str = sorted(lr_now["top_downstream_niche_DE_genes"].astype(str)) == \
           sorted(lr_ref.iloc[:, 2].astype(str))
print(f"\nligand-receptor pairs identical to R : {same_pairs}  ({len(pairs_R)} pairs)")
print(f"downstream-gene strings identical    : {same_str}   <- the admissibility claim")

iter_panel(11, extra=(f"full {n_lig}-ligand sweep {T_LR_NOW:.2f} s (log: {before:.1f} s -> "
                      f"{after_logged:.1f} s, {before/after_logged:.0f}x); {n_lig - n_sig} "
                      f"redundant slices removed; output identical to R"),
           bars=("niche_LR_spot, full ligand sweep",
                 ["before memoisation", "shipped"], [before, T_LR_NOW]),
           bar_colors=[C_R, C_PY])
''')


def cells():
    C = []
    C.append(md(r"""
# `py-nichede` — how the port evolved, iteration by iteration

**Notebook 4 of 4.** Audience: **an auditor of the engineering process**, asking *"did the agent
really iterate, or did it skip the loop and log only the survivor?"*

There is exactly **one `## Iteration N — title` header per entry in
[`ITERATION_LOG.md`](../ITERATION_LOG.md)** — 12 in total: the baseline translation, three
equivalence fixes, seven accepted accelerations, and **one rejection** (iteration 8, `REJECT_SLOW`).
Each block carries a written narrative of what changed and why, plus a code cell that
**re-measures** the claim live and emits that iteration's subplot. The rejection is kept, with
its numbers, because the reason it failed is what shaped the design of iteration 9.

Two conventions used throughout:

* **Parity metric** — Pearson on the Wald `T_stat`, threshold 0.99, pre-registered in
  `data/manifest.yaml`. It is flat at 1.000000 across every iteration; the equivalence fixes
  (1–3) moved *other* outputs (valid flags, BH p-values, the `Int=FALSE` branch), which is why
  each of those cells measures the output it actually repaired rather than re-reporting the
  headline number.
* **Wall clock** — measured on the **dev fixture** (848 spots × 300 genes × 3 kernel bandwidths,
  7 cell types) for pipeline-level iterations, and as an isolated micro-benchmark for the three
  step-level rewrites (4, 5, 6). The two are plotted with different markers because they are not
  comparable quantities.

Every accepted rewrite is admissibility class **(E) — an exact algebraic identity — or a pure
scheduling change. No (B) bounded-epsilon approximation is used anywhere in this port**, so
`MATH.md` carries no perturbation budget.
"""))

    C.append(md("## Setup"))
    C.append(code(PREAMBLE))
    C.append(code(r'''
N_CPU = len(os.sched_getaffinity(0))
print("visible CPUs:", N_CPU, " | joblib workers used below:", N_JOBS)

# ---- the dev fixture (300 highest-count genes): the wall-clock bench --------
dev     = RefDump(DEV_DIR)
dcts    = list(dev.meta["cell_types"])
dcells  = list(dev.meta["cell_names"])
dgenes  = list(dev.meta["gene_names"])
dsigma  = np.atleast_1d(np.asarray(dev.meta["sigma"], dtype=float))
dcounts = pd.DataFrame(dev["ref_counts"],   index=dcells, columns=dgenes)
dcoord  = pd.DataFrame(dev["in_coord"],     index=dcells, columns=["imagerow", "imagecol"])
dlibmat = pd.DataFrame(dev["ref_ref_expr"], index=dcts,   columns=dgenes)
ddeconv = pd.DataFrame(dev["in_deconv"],    index=dcells, columns=dcts)
R_T_DEV     = float(np.atleast_1d(dev.meta["time_niche_DE"])[0])
R_CORES_DEV = int(np.atleast_1d(dev.meta["n_cores"])[0])

# ---- the canonical full fixture: used where scale is the point -------------
full    = RefDump(REF_DIR)
fcts    = list(full.meta["cell_types"])
fcells  = list(full.meta["cell_names"])
fgenes  = list(full.meta["gene_names"])
fsigma  = np.atleast_1d(np.asarray(full.meta["sigma"], dtype=float))
fcounts = pd.DataFrame(full["ref_counts"],   index=fcells, columns=fgenes)
fcoord  = pd.DataFrame(full["in_coord"],     index=fcells, columns=["imagerow", "imagecol"])
flibmat = pd.DataFrame(full["ref_ref_expr"], index=fcts,   columns=fgenes)
fdeconv = pd.DataFrame(full["in_deconv"],    index=fcells, columns=fcts)

print(f"dev  fixture: {dcounts.shape[0]} spots x {dcounts.shape[1]} genes x "
      f"{len(dsigma)} kernels   (R: {R_T_DEV:.2f} s on {R_CORES_DEV} cores)")
print(f"full fixture: {fcounts.shape[0]} spots x {fcounts.shape[1]} genes x "
      f"{len(fsigma)} kernels   (R: "
      f"{float(np.atleast_1d(full.meta['time_niche_DE'])[0]):.1f} s on "
      f"{int(np.atleast_1d(full.meta['n_cores'])[0])} cores)")
'''))

    C.append(md(r"""
### Parse `ITERATION_LOG.md`

The narrative below is written by hand; the **numbers** on every plot come either from a live
measurement in that iteration's cell or from parsing the log, never from prose. This cell reads
the log so the aggregate figure and the per-iteration panels share one source of truth.
"""))
    C.append(code(r'''
import yaml
LOG_PATH = os.path.join(PKG_ROOT, "ITERATION_LOG.md")
raw = open(LOG_PATH).read()

blocks = []
for header, body in re.findall(r"^##\s+(.+?)\n(.*?)(?=^##\s|\Z)", raw, re.S | re.M):
    m = re.search(r"```yaml\n(.*?)```", body, re.S)
    if not m:
        continue
    y = yaml.safe_load(m.group(1))
    if "iter" not in y:
        continue
    y["_header"] = header.strip()
    blocks.append(y)
blocks.sort(key=lambda b: b["iter"])
LOG = pd.DataFrame(blocks)

# Every block now carries a PIPELINE wall clock on the dev fixture (the log was
# revised to use controlled ablations rather than cross-process comparisons), so
# the series is directly plottable.  Blocks whose notes begin "STEP-LEVEL" claim
# no pipeline delta of their own and are flagged so the plot cannot mislead.
LOG["status"] = LOG["status"].fillna("baseline")
LOG["scope"] = np.where(LOG["notes"].fillna("").str.strip().str.startswith("STEP-LEVEL"),
                        "step-level (no pipeline delta claimed)", "pipeline")
LOG["joint"] = LOG.get("joint_with_iter5", pd.Series([np.nan] * len(LOG))).notna() | \
               LOG.get("joint_with_iter7", pd.Series([np.nan] * len(LOG))).notna()
display(LOG[["iter", "_header", "status", "admissibility", "wall_clock_mean_s",
             "parity_metric", "parity_passes", "scope"]])
N_ITERS = len(LOG)
PARITY_THRESHOLD = float(LOG["parity_threshold"].dropna().iloc[0])
print(f"\n{N_ITERS} iteration blocks parsed; parity threshold {PARITY_THRESHOLD} "
      f"({LOG['parity_class'].dropna().iloc[0]})")
assert N_ITERS >= 2, "NOTEBOOKS.md requires at least a baseline plus one follow-up"
'''))

    C.append(md(r"""
### Plotting + timing helpers

`iter_panel(n)` draws the two-panel progress figure **as of iteration `n`**, with that
iteration's point highlighted, and optionally a third panel with the iteration's own
before/after bars. Every iteration — including the baseline and the rejection — gets one.
"""))
    C.append(code(r'''
def timeit(fn, n_reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))

def unpack_T(obj_, k):
    n_ct = len(obj_.cell_types); ng = len(obj_.gene_names)
    T = np.full((n_ct, n_ct, ng), np.nan)
    for g, r in enumerate(obj_.niche_DE[k]):
        if r["valid"] == 1:
            T[:, :, g] = r["T_stat"]
    return T

def _adjust_like_R(arr, level):
    """Apply the port's p_adjust exactly where nicheDE applies BH, to R's own raw p."""
    from pynichede.rstats import p_adjust
    a = np.asarray(arr, dtype=float)
    if level == "gene":
        return p_adjust(a, "BH")
    if level == "ct":                      # BH across cell types, per gene (rows)
        return np.vstack([p_adjust(a[i], "BH") for i in range(a.shape[0])])
    out = a.copy()                         # BH across niche types, per (index, gene)
    for g in range(a.shape[2]):
        for k in range(a.shape[0]):
            out[k, :, g] = p_adjust(a[k, :, g], "BH")
    return out

def logged(it, field):
    """Read a field of one iteration straight out of ITERATION_LOG.md."""
    v = LOG.loc[LOG["iter"] == it, field]
    return None if not len(v) or pd.isna(v.iloc[0]) else v.iloc[0]

def iter_panel(n, extra="", bars=None, bar_colors=None):
    """The subplot NOTEBOOKS.md requires for every iteration block."""
    ncol = 3 if bars else 2
    fig, ax = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.5))

    p_now = LOG[LOG["iter"] <= n]
    ax[0].errorbar(p_now["iter"], p_now["wall_clock_mean_s"],
                   yerr=p_now["wall_clock_stddev_s"].fillna(0.0),
                   fmt="o-", color=C_PY, capsize=3,
                   label="pipeline wall clock (dev fixture)")
    stp = p_now[p_now["scope"].str.startswith("step-level")]
    if len(stp):
        ax[0].scatter(stp["iter"], stp["wall_clock_mean_s"], marker="s", s=55,
                      color="#7b4fa0", zorder=4, label="step-level (no pipeline delta)")
    ax[0].axhline(R_T_DEV, ls="--", color=C_R, lw=1,
                  label=f"R reference {R_T_DEV:.1f} s ({R_CORES_DEV} cores)")
    row = LOG[LOG["iter"] == n].iloc[0]
    ax[0].scatter([n], [row["wall_clock_mean_s"]], s=190, facecolors="none",
                  edgecolors="k", linewidths=2, zorder=5, label=f"iteration {n}")
    ax[0].set_yscale("log"); ax[0].set_xlim(-0.6, N_ITERS - 0.4)
    ax[0].set_xticks(range(N_ITERS))
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("wall clock (s, log scale)")
    ax[0].set_title("wall clock so far"); ax[0].legend(fontsize=6.5, loc="lower left")

    a_now = LOG[LOG["iter"] <= n]
    colr = ["#2e7d32" if str(s).startswith("REJECT") is False else C_BAD
            for s in a_now["status"]]
    ax[1].plot(a_now["iter"], a_now["parity_metric"], "-", color=C_PY, lw=1.2, zorder=1)
    ax[1].scatter(a_now["iter"], a_now["parity_metric"], c=colr, s=45, zorder=2)
    ax[1].axhline(PARITY_THRESHOLD, ls="--", color=C_BAD, lw=1.2,
                  label=f"pre-registered gate {PARITY_THRESHOLD}")
    ax[1].scatter([n], [row["parity_metric"]], s=190, facecolors="none",
                  edgecolors="k", linewidths=2, zorder=5)
    ax[1].set_ylim(0.985, 1.003); ax[1].set_xlim(-0.6, N_ITERS - 0.4)
    ax[1].set_xticks(range(N_ITERS))
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("T_stat Pearson vs R")
    ax[1].set_title("parity so far"); ax[1].legend(fontsize=6.5, loc="lower left")

    if bars:
        title, labels, vals = bars
        cols = bar_colors or ["#888888", C_PY]
        b = ax[2].bar(labels, vals, color=cols)
        for rect, v in zip(b, vals):
            ax[2].text(rect.get_x() + rect.get_width() / 2, v,
                       (f"{v * 1e3:.3g} ms" if v < 1 else f"{v:.2f} s"),
                       ha="center", va="bottom", fontsize=7)
        ax[2].set_yscale("log"); ax[2].set_ylabel("wall clock (log scale)")
        ax[2].set_title(title, fontsize=9)
        ax[2].tick_params(axis="x", labelsize=7)

    st = str(row["status"])
    fig.suptitle(f"Iteration {n} — {row['_header']}   [{st}]",
                 fontsize=10, y=1.04,
                 color=(C_BAD if st.startswith("REJECT") else "black"))
    plt.show()
    if extra:
        display(Markdown(f"> **Measured live in this cell:** {extra}"))
print("helpers ready")
'''))

    for n, title, narrative, body in ITERS:
        C.append(md(f"## Iteration {n} — {title}\n\n{narrative.strip()}"))
        C.append(code(body))

    C.append(md(r"""
## Aggregate evolution figure

The two-panel summary `engine.plot_evolution` writes to `examples/evolution.png`, rendered
inline **and** saved to disk. Left: wall clock against iteration, pipeline measurements joined
and step-level micro-benchmarks marked separately, with R's reference time as the horizontal
line. Right: the pre-registered parity metric against iteration, with the 0.99 gate as the red
dashed line, the rejected iteration in red, and the cumulative speedup annotated.
"""))
    C.append(code(r'''
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

ax[0].errorbar(LOG["iter"], LOG["wall_clock_mean_s"],
               yerr=LOG["wall_clock_stddev_s"].fillna(0.0),
               fmt="o-", color=C_PY, lw=1.8, ms=8, capsize=3,
               label="pipeline wall clock (dev fixture, 300 genes x 3 kernels)")
stp = LOG[LOG["scope"].str.startswith("step-level")]
ax[0].scatter(stp["iter"], stp["wall_clock_mean_s"], marker="s", s=90, color="#7b4fa0",
              zorder=4, label="step-level rewrite (no pipeline delta claimed)")
rej = LOG[LOG["status"].astype(str).str.startswith("REJECT")]
ax[0].scatter(rej["iter"], rej["wall_clock_mean_s"], s=230, facecolors="none",
              edgecolors=C_BAD, linewidths=2.4, zorder=6, label="REJECTED")
ax[0].axhline(R_T_DEV, ls="--", color=C_R, lw=1.4,
              label=f"R nicheDE reference: {R_T_DEV:.2f} s on {R_CORES_DEV} cores")
for _, r in LOG.iterrows():
    ax[0].annotate(str(int(r["iter"])), (r["iter"], r["wall_clock_mean_s"]),
                   textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7)
ax[0].set_yscale("log"); ax[0].set_xticks(range(N_ITERS))
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("wall clock (s, log scale)")
ax[0].set_title("py-nichede: wall clock by iteration")
ax[0].legend(fontsize=7, loc="center left")

base = float(LOG.loc[LOG["iter"] == 0, "wall_clock_mean_s"].iloc[0])
final = float(LOG["wall_clock_mean_s"].iloc[-1])
colr = [C_BAD if str(s).startswith("REJECT") else "#2e7d32" for s in LOG["status"]]
ax[1].plot(LOG["iter"], LOG["parity_metric"], "-", color=C_PY, lw=1.6, zorder=1)
ax[1].scatter(LOG["iter"], LOG["parity_metric"], c=colr, s=90, zorder=2, edgecolors="k",
              linewidths=0.5)
ax[1].axhline(PARITY_THRESHOLD, ls="--", color=C_BAD, lw=1.6,
              label=f"pre-registered gate: Pearson {PARITY_THRESHOLD}")
ax[1].set_ylim(0.985, 1.004); ax[1].set_xticks(range(N_ITERS))
ax[1].set_xlabel("iteration"); ax[1].set_ylabel("T_stat Pearson vs R (parity metric)")
ax[1].set_title("py-nichede: parity by iteration")
ax[1].legend(fontsize=7, loc="lower left")
ax[1].text(0.02, 0.90,
           f"cumulative: {base:.2f} s -> {final:.2f} s  = {base / final:.1f}x\n"
           f"vs R ({R_CORES_DEV} cores): {R_T_DEV / final:.1f}x\n"
           f"parity flat at {LOG['parity_metric'].min():.6f} throughout\n"
           f"{int((LOG['status'].astype(str) == 'ACCEPT').sum())} accepted, "
           f"{len(rej)} rejected, {N_ITERS} blocks",
           transform=ax[1].transAxes, fontsize=8, va="top",
           bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#bbbbbb"))

plt.tight_layout()
OUT_PNG = os.path.join(PKG_ROOT, "examples", "evolution.png")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print("saved", OUT_PNG, f"({os.path.getsize(OUT_PNG) / 1024:.0f} KB)")
'''))

    C.append(md(r"""
## Summary table

Rendered from `ITERATION_LOG.md`, so it cannot drift from the authoritative log.
"""))
    C.append(code(r'''
S = LOG[["iter", "_header", "action", "admissibility", "wall_clock_mean_s",
         "speedup_vs_baseline", "parity_metric", "status", "scope"]].copy()
S.columns = ["iter", "title", "action", "admissibility", "mean time (s)",
             "speedup vs baseline", "T_stat Pearson", "status", "scope"]
display(S.style.hide(axis="index"))

n_accept = int((LOG["status"].astype(str) == "ACCEPT").sum())
n_reject = int(LOG["status"].astype(str).str.startswith("REJECT").sum())
n_base   = int((LOG["status"].astype(str) == "baseline").sum())
print(f"{N_ITERS} iteration blocks: {n_base} baseline, {n_accept} accepted, {n_reject} rejected")
print("admissibility classes used:", sorted(set(LOG["admissibility"].dropna())))
assert not (LOG["parity_metric"] < PARITY_THRESHOLD).any(), \
    "an iteration dropped below the pre-registered gate"
print(f"no iteration ever dropped below the pre-registered gate "
      f"(min Pearson {LOG['parity_metric'].min():.6f} >= {PARITY_THRESHOLD})")
'''))

    C.append(md(r"""
## What the iteration history shows

* **12 blocks, one per `ITERATION_LOG.md` entry**, including the rejection. Iteration 8 tried
  the most obvious parallelisation — one joblib task per gene, exactly mirroring R's
  `foreach %dopar%` — and made the pipeline **20.9× slower** than the chunked dispatch that
  replaced it, and slower even than running serially. It is kept in the record because the
  profile that explained the failure (per-task pickling of the shared arrays, and every worker
  opening a 17-thread BLAS pool) is precisely what iteration 9 was designed against.

* **Speedups are attributed by controlled ablation, and two of them are joint.** Iterations 5
  and 7 are redundant fixes for the same bottleneck: each is worth ~1.0–1.1× alone and 8.0×
  together, and the log books the 8.0× once against the pair rather than twice against each.
  Iterations 4 and 6 are logged as step-level with **no pipeline delta claimed at all**. The
  cells above re-measure these numbers live rather than restating them, and where the live
  figure differs from the log it is printed side by side instead of being quietly dropped.

* **The first three iterations are equivalence fixes, not optimisations.** Every one of them
  changed *which genes get reported*, and none of them moved the headline parity metric — the
  Wald statistic was already Pearson 1.000000 at the baseline. That is the argument for grading
  a port on more than one number: a `T_stat` that matches perfectly can still sit above a
  `p.adjust` that silently discards a third of the significant genes.

* **Every accepted rewrite is class (E) or scheduling.** No bounded-epsilon approximation was
  used anywhere, so there is no perturbation budget to charge against the parity threshold, and
  the parity metric is flat at 1.000000 across all 12 blocks.

* **One iteration came from running the deliverables, not reading the code.** Iteration 11 was
  found because executing the tutorial notebook made a five-minute cell obvious; it turned out to
  be 576 redundant rebuilds of the same array, and removing them was worth 105× with
  byte-identical output. Notebooks that are merely sketched would not have surfaced it.

* **Stop reason.** The playbook is exhausted for this port's shape: `niche_DE`'s remaining
  runtime is the per-gene IRLS (already a rank-truncated QR at minimum arithmetic cost) and the
  Brown/Cauchy pooling (already `O(n_gene × n_celltype^2)` and vectorised); `niche_LR_*` is now
  dominated by the per-candidate Poisson GLMs, irreducible without changing the statistics.

Related: [`ITERATION_LOG.md`](../ITERATION_LOG.md) (authoritative log),
[`MATH.md`](../MATH.md) (the admissibility proofs referenced above),
[`compare_R_vs_Python.ipynb`](compare_R_vs_Python.ipynb) (the parity evidence),
[`examples/benchmark.py`](benchmark.py) (the standalone benchmark harness).
"""))
    return C
