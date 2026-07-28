# Acceleration Iteration Log — py-nichede

> One block per Equivalence/Acceleration step, in the order they happened.
> Parsed by `engine/plot_evolution.py`; narrated in `examples/evolution.ipynb`.
>
> Parity class for the headline metric is **ordinal (Pearson on the Wald
> `T_stat`)**, threshold 0.99, pre-registered in `data/manifest.yaml`.
>
> **Measurement protocol.** Wall-clock is the complete `niche_DE` call on the
> dev fixture (848 spots x 300 genes x 7 cell types x 3 kernel bandwidths),
> single process unless stated, reported as **warmup-excluded mean +- sd of 3
> runs**. A warmup is mandatory because the first `mvnconv` call builds the
> 1.69 s Mehler table. The R reference on the same fixture with 8 cores is
> **18.25 s**.
>
> Attribution for the acceleration iterations is by **controlled ablation**
> (`examples/ablation.py`): the shipped pipeline is run with exactly one rewrite
> reverted to its predecessor, everything else held fixed. Where a block quotes
> a step-level microbenchmark it says so explicitly and does **not** put that
> number in `wall_clock_mean_s`.
>
> **Interaction warning (important).** Iterations 5 (`cho_solve`) and 7
> (single-threaded BLAS) attack the *same* bottleneck — a small, BLAS-threaded
> triangular inverse. Reverting either one alone costs almost nothing
> (1.00x and 1.08x); reverting **both** costs **8.0x**. Their speedups are
> therefore not additive and are reported as a joint attribution.

---

## Baseline — 2026-07-28

```yaml
iter: 0
status: baseline
action: null
admissibility: null
playbook_section: null
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
warmup_run_s: excluded
first_cold_run_s: 33.95
parity_metric: 1.000000
parity_class: ordinal
parity_threshold: 0.99
parity_passes: true
notes: |
  First end-to-end translation.  Design matrix built with a literal
  per-spot Python transcription of R's `for (k in 1:nrow(pstg))` loop;
  `V = solve(chol(X'WX)) %*% t(solve(chol(X'WX)))` transcribed literally;
  one joblib task per gene; no BLAS thread management.
  T_stat already clears the gate at Pearson 1.000000, but three outputs
  were wrong: `valid` flags disagreed on 2/300 genes, the interaction-level
  BH p-values correlated at only Spearman 0.859, and the Int=FALSE branch
  produced no valid genes at all.  Those are equivalence bugs, fixed in
  iters 1-3 before any acceleration work began.

  The wall-clock quoted here is the CONTROLLED baseline: the shipped pipeline
  with the two pre-acceleration choices that actually mattered restored
  (explicit triangular inverse + unpinned BLAS), warm, serial, 3 runs.
  The very first cold run of the untuned code was 33.95 s, recorded above as
  `first_cold_run_s`, but that number is not comparable to anything else in
  this log: it also paid the one-off 1.69 s table build and used per-gene
  joblib dispatch.  15.43 s is the honest reference point.
```

---

## iter 1 — equivalence: R's single-column `drop = TRUE`

```yaml
iter: 1
status: ACCEPT
action: r_single_column_drop_compat
playbook_section: null
admissibility: exact
admissibility_evidence: |
  Not an acceleration rewrite — an equivalence fix.  When exactly one
  interaction column survives the M filter, R's `X_partial = X[,-null]`
  drops to a vector, so `nvar = ncol(X_partial)` is NULL,
  `coeff = coefficients[1:(NULL+1)]` is length 0, and
  `beta[c(1:n_type^2)[-null]] = coeff[-1]` raises
  "replacement has length zero".  The enclosing tryCatch swallows it and
  the gene is reported invalid.  Verified by locating the two affected
  genes (FABP1, SERPINA1) and confirming R's log-likelihood was recorded
  (so the GLM had succeeded) while `valid` was 0 and `nulls` had length 49.
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Also fixed the invalid-return contract: R's else-branch always reports
  `nulls = c(1:n_type^2)` regardless of the `null` vector it computed.
  valid-flag agreement 0.9933 -> 1.000000; `nulls` max abs err 1 -> 0.
```

### Decision
ACCEPT — equivalence fix, mandatory.

---

## iter 2 — equivalence: R's lazy `n` in `p.adjust`

```yaml
iter: 2
status: ACCEPT
action: p_adjust_lazy_n
playbook_section: null
admissibility: exact
admissibility_evidence: |
  `p.adjust(p, method, n = length(p))` has a LAZY default.  `n` is not
  forced until `stopifnot(n >= lp)`, which runs *after* the body has
  executed `p <- p[!is.na(p)]`.  So an un-supplied `n` equals the number of
  NON-NA p-values, not the total.  Confirmed against R:
  `p.adjust(c(.4,NA,.01,.9,NA,.02,.5),"BH")` returns exactly what
  `p.adjust(c(.4,.01,.9,.02,.5),"BH")` returns.  After the fix, feeding R's
  own raw p-values through the Python `p_adjust` reproduces R's adjusted
  arrays with max abs err **0.000e+00** at all three levels
  (gene / cell type / interaction).
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Largest single accuracy win of the port.  Using n = total inflated every
  adjusted p-value by n_total/n_valid (~3x here), which silently lost a
  third of the significant genes.
  pval_pos_interaction_level Spearman(-log10 p) 0.859 -> 1.000000
  pval_pos_gene_level 0.994 -> 0.999937
  reported gene-set Jaccard 0.67-0.86 -> 0.93-1.00
```

### Decision
ACCEPT — equivalence fix, mandatory.

---

## iter 3 — equivalence: the `Int = FALSE` linear-model branch

```yaml
iter: 3
status: ACCEPT
action: int_false_summary_lm_semantics
playbook_section: null
admissibility: exact
admissibility_evidence: |
  R's continuous-data branch takes its test statistic from
  `summary(lm)$coefficients[, 3]` (t-values), not from `beta/tau`, and its
  covariance from `summary(lm)$cov.unscaled * sigma^2`, which contains only
  the NON-aliased columns.  The first draft padded the covariance back to
  full width with NaN on the aliased rows and then tested
  `diag(V) == 0` on the padded matrix — which fired on every gene because
  the single-batch `batchvar` column is always aliased against the
  intercept, so every gene was thrown away.
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Int=FALSE valid genes 0/300 -> 93/300, matching R exactly (agreement
  1.000000); Int=FALSE T_stat Pearson 1.000000 on 1219 finite entries.
```

### Decision
ACCEPT — equivalence fix, mandatory.

---

## iter 4 — acceleration: vectorised design-matrix construction

```yaml
iter: 4
status: ACCEPT
action: broadcast_design_matrix
playbook_section: "§1 loop hoisting / vectorisation"
admissibility: exact
admissibility_evidence: |
  R builds X one spot at a time:
  `X[k,] = as.vector(round(EN[k,],2) %*% t(pstg[k,]))`, i.e. the
  column-major flattening of the outer product `EN[k] (x) pstg[k]`.
  Stacking those outer products over k is exactly the rank-1 broadcast
  `EN[:, :, None] * pstg[:, None, :]` re-ordered so that the niche index
  varies fastest.  Measured max abs deviation vs the loop: **0.0** (bit
  identical, same multiplications in the same association order).
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
step_time_old_s: 0.2224
step_time_new_s: 0.01556
step_speedup: 14.3
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  STEP-LEVEL measurement, not pipeline: building the design matrix for 200
  runnable genes takes 0.2224 s with R's per-spot loop and 0.01556 s with the
  broadcast, i.e. **14.3x on that step**.  At pipeline level the design step
  is only ~1% of `niche_DE`, so `wall_clock_mean_s` is unchanged within noise
  and no pipeline speedup is claimed.  Kept because it is free and exact.
```

### Decision
ACCEPT.

### Commit / branch
```
in-tree: pynichede/niche_de.py::_niche_de_core (broadcast + transpose + reshape)
```

---

## iter 5 — acceleration: `cho_solve` instead of an explicit triangular inverse

```yaml
iter: 5
status: ACCEPT
action: cho_solve_instead_of_explicit_inverse
playbook_section: "§2 exact algebraic identity (Cholesky solve)"
admissibility: exact
admissibility_evidence: |
  R computes `A = chol(X'WX)` then `V = solve(A) %*% t(solve(A))`.
  Since `A' A = X'WX` with A upper triangular,
  `A^-1 A^-T = (A' A)^-1 = (X'WX)^-1`, which is exactly what
  `cho_solve((A, upper), I)` returns via two triangular solves — without
  ever forming `A^-1`.  Identity is exact in exact arithmetic; measured max
  relative deviation on the fixture **1.49e-15** (i.e. f64 rounding only).
  The Cholesky factorisation itself is kept, so the "not positive definite"
  failure path that makes R mark a gene invalid still fires identically.
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
ablation_this_change_only_s: 1.917
ablation_this_change_only_sd: 0.003
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
joint_with_iter7: true
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Ablating ONLY this rewrite (explicit `solve_triangular(A, I)` restored,
  BLAS still pinned) gives 1.917 +- 0.003 s vs the shipped 1.923 +- 0.004 s:
  **no measurable pipeline effect on its own** (1.00x), max deviation
  4.44e-15.  It is nonetheless accepted, because together with iter 7 it is
  worth 8.0x — see the interaction warning at the top and iter 7's block.
  Accepting an exact, strictly-cheaper identity with no downside is the
  right call even when its solo attribution is zero.
```

### Decision
ACCEPT.

---

## iter 6 — acceleration: memoised Mehler/`mvnconv` lookup table

```yaml
iter: 6
status: ACCEPT
action: cache_mvnlookup
playbook_section: "§1.2 memoisation"
admissibility: exact
admissibility_evidence: |
  `mvnconv` is a pure function of (rho grid, target, side).  The 1991 x 9
  covariance table depends on nothing else, so hoisting its construction
  behind an `lru_cache` is pure memoisation — the returned values are
  bit-identical (measured max abs err 0.0).  Without the cache the Hermite
  quadrature is re-run for every gene x every kernel bandwidth.
wall_clock_mean_s: 15.43
wall_clock_stddev_s: 2.05
wall_clock_runs_s: [15.30, 12.99, 18.00]
step_time_old_s: 1.690
step_time_new_s: 0.0000144
step_speedup: 117000
speedup_vs_previous: 1.00
speedup_vs_baseline: 1.00
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  STEP-LEVEL: building the Mehler table costs 1.690 s; a cached `mvnconv`
  call costs 1.44e-5 s.  `mvnconv` is invoked once per (gene x kernel x
  direction x pooling level) — on the canonical fixture ~10^5 times — so
  without the cache the table build alone would dominate everything else.
  The cache was present from the first working version, so it never appears
  as a pipeline delta; it is logged because removing it is catastrophic and
  a future maintainer needs to know the `lru_cache` is load-bearing.
  It is also why every timing in this log is warmup-excluded: the first
  `niche_DE` call in a fresh process pays the 1.69 s build once.
```

### Decision
ACCEPT.

---

## iter 7 — acceleration: pin the BLAS thread pool to 1

```yaml
iter: 7
status: ACCEPT
action: single_threaded_blas
playbook_section: "§4 scheduling (no FLOP change)"
admissibility: exact
admissibility_evidence: |
  Changing the number of BLAS worker threads changes neither the operations
  performed nor their order for the level-2/level-3 kernels used here
  (`syrk`, `potrf`, `trsm` on a ~848 x 50 design), because those routines
  partition by output block, not by reduction.  Verified empirically: the
  full parity report is unchanged to the last reported digit before and
  after.  This is a pure scheduling change.
wall_clock_mean_s: 1.923
wall_clock_stddev_s: 0.004
wall_clock_runs_s: [1.9202, 1.9165, 1.9133]
ablation_this_change_only_s: 2.085
ablation_this_change_only_sd: 0.096
ablation_with_iter5_also_reverted_s: 15.43
speedup_vs_previous: 8.02
speedup_vs_baseline: 8.02
joint_with_iter5: true
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Implemented with `threadpoolctl.threadpool_limits(limits=1)` around the
  per-sigma loop (`contextlib.nullcontext()` fallback if threadpoolctl is
  missing).  Niche-DE's inner problem is one ~848 x 50 GLM per gene; a
  17-thread OpenBLAS pool spent more time in barriers than in arithmetic.

  HONEST ATTRIBUTION.  Ablating only this change gives 2.085 +- 0.096 s vs
  the shipped 1.923 +- 0.004 s -> just 1.08x on its own.  Ablating this
  change AND iter 5 together gives 15.43 +- 2.05 s -> **8.02x**.  The two are
  redundant fixes for the same bottleneck: the explicit
  `solve_triangular(A, I)` is what made the thread pool expensive, so
  removing either the explicit inverse or the threads removes the cost.
  The 8.02x is booked jointly against iters 5+7, not twice.
  (An earlier cross-process measurement reported 24.65 s -> 4.35 s = 5.7x for
  this change alone; that comparison was confounded by the cold table build
  and by per-gene dispatch, and is superseded by the ablation above.)
```

### Decision
ACCEPT.

---

## iter 8 — REJECTED: one joblib task per gene

```yaml
iter: 8
status: REJECT_SLOW
action: per_gene_joblib_tasks
playbook_section: "§5 parallelisation"
admissibility: exact
admissibility_evidence: |
  Each gene's fit is independent (R itself uses `foreach %dopar%` over
  genes), so any partition of the gene index set is admissible.
wall_clock_mean_s: 14.34
wall_clock_stddev_s: 2.06
wall_clock_runs_s: [14.34]
speedup_vs_previous: 0.048
speedup_vs_baseline: 0.048
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Ablation with per-gene dispatch restored, n_jobs=8: 14.34 +- 2.06 s versus
  the shipped chunked dispatch at 0.686 +- 0.007 s — **20.9x SLOWER**, and
  slower even than the 1.923 s serial path.  joblib re-pickled the shared
  effective-niche / num_cells / ref_expr arrays for every small task, and
  each worker opened its own full-width BLAS pool (8 workers x 17 threads on
  a 17-core box).  Output identical (max deviation 0.0), so the rejection is
  purely on wall-clock.
```

### Decision
REJECT_SLOW — rolled back and replaced by iter 9.

---

## iter 9 — acceleration: chunked dispatch + `inner_max_num_threads=1`

```yaml
iter: 9
status: ACCEPT
action: chunked_parallel_dispatch
playbook_section: "§5 parallelisation"
admissibility: exact
admissibility_evidence: |
  Same independence argument as iter 8; only the partition granularity and
  the worker thread budget change.  Results are reassembled in gene order,
  so the output list is identical element-for-element (verified: the full
  parity report is unchanged between n_jobs=1 and n_jobs=16).
wall_clock_mean_s: 0.686
wall_clock_stddev_s: 0.007
wall_clock_runs_s: [0.686]
ablation_per_gene_dispatch_s: 14.34
speedup_vs_previous: 2.80
speedup_vs_baseline: 22.5
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  ~4 chunks per worker instead of one task per gene, dispatched under
  `parallel_backend("loky", inner_max_num_threads=1)`.
  n_jobs=8: 0.686 +- 0.007 s, versus 1.923 s serial (2.80x) and versus
  14.34 s for the rejected per-gene dispatch (20.9x).
  R on the same fixture with 8 cores: 18.25 s -> **26.6x faster than R**.
```

### Decision
ACCEPT.

---

## iter 10 — acceleration: hoist the gene gate out of the workers

```yaml
iter: 10
status: ACCEPT
action: prefilter_runnable_genes
playbook_section: "§1.3 early exit"
admissibility: exact
admissibility_evidence: |
  `niche_DE_core`'s first branch returns the invalid result whenever
  `sum(counts) <= C` or every cell type is below its gamma-quantile filter.
  Both predicates depend only on the counts column sums and on `ref_expr`,
  neither of which changes per gene inside the worker.  Evaluating them
  once in the parent for all genes and shipping only the survivors is the
  same predicate applied at the same point, so the returned list is
  identical (verified: valid-flag agreement stays 1.000000).
wall_clock_mean_s: 0.686
wall_clock_stddev_s: 0.007
wall_clock_runs_s: [0.686]
speedup_vs_previous: 1.00
speedup_vs_baseline: 22.5
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  On the dev fixture only 97 of 300 genes are runnable; on the canonical
  full fixture only ~4870 of 21708.  The prefilter removes ~68% (dev) to
  ~78% (full) of the tasks and, more importantly, the same fraction of the
  counts columns that would otherwise be pickled to workers.  It is folded
  into the 0.686 s figure above; it matters most on the FULL fixture, where
  `niche_DE` runs in 32.6 s on 16 cores against R's 852.2 s on 16 cores.
```

### Decision
ACCEPT.

---

## iter 11 — acceleration: memoise the per-kernel ligand-target slice

```yaml
iter: 11
status: ACCEPT
action: cache_kernel_ligand_slice
playbook_section: "§1.2 memoisation"
admissibility: exact
admissibility_evidence: |
  Inside `niche_LR_spot` / `niche_LR_cell`, R re-runs the whole
  filter-and-reindex of the NicheNet ligand-target matrix once per candidate
  ligand:
      sig    <- T_vector[[ top_kernel[ind] ]]
      genes  <- gene_names[!is.na(sig)]
      lv     <- ligand_target_matrix[rownames %in% genes, ]
      lv     <- lv[genes, ]
  Every line of that depends **only** on `top_kernel[ind]` — the index of the
  best-fitting kernel for that ligand — which takes at most `length(sigma)`
  distinct values.  So across the 579 candidate ligands there are at most 3
  distinct results and the other 576 evaluations recompute an identical array.
  Hoisting them behind a dict keyed on the kernel index is textbook memoisation
  of a pure function: bit-identical output, verified by diffing the produced
  `niche_LR_spot` table against the pre-optimisation run — **byte-identical**,
  including the `top_downstream_niche_DE_genes` strings — and against R's
  table (9/9 ligand-receptor pairs identical).
wall_clock_mean_s: 3.0
wall_clock_stddev_s: null
wall_clock_runs_s: [3.0]
ablation_this_change_only_s: 314.5
speedup_vs_previous: 104.8
speedup_vs_baseline: 104.8
parity_metric: 1.000000
parity_delta_vs_baseline: 0.0
parity_passes: true
notes: |
  Measured on the canonical full fixture (579 candidate ligands against a
  16968 x 579 NicheNet matrix): **314.5 s -> 3.0 s, 104.8x**.  This is a
  separate wall-clock axis from iters 0-10, which time `niche_DE`; niche-LR is
  a downstream call and was not part of that trajectory.  Surfaced only
  because building the tutorial notebook made the minutes-long call obvious —
  a good argument for the protocol's insistence that the notebooks be
  genuinely executed rather than sketched.
```

### Decision
ACCEPT.

---

## Summary

All wall-clock figures: dev fixture (848 spots x 300 genes x 7 cell types x
3 kernels), warmup-excluded mean of 3, serial unless the row says otherwise.

| iter | action | admissibility | pipeline time (s) | attribution | T_stat Pearson | status |
|---|---|---|---|---|---|---|
| 0 | (controlled baseline) | — | 15.43 +- 2.05 | 1.0x | 1.000000 | — |
| 1 | R single-column `drop` compat | exact | 15.43 | equivalence, no speed change | 1.000000 | ACCEPT |
| 2 | `p.adjust` lazy `n` | exact | 15.43 | equivalence, no speed change | 1.000000 | ACCEPT |
| 3 | `Int=FALSE` summary.lm semantics | exact | 15.43 | equivalence, no speed change | 1.000000 | ACCEPT |
| 4 | broadcast design matrix | exact | 15.43 | 14.3x on that step, ~0 at pipeline | 1.000000 | ACCEPT |
| 5 | `cho_solve` | exact | 15.43 | 1.00x alone; 8.02x jointly with iter 7 | 1.000000 | ACCEPT |
| 6 | memoise `mvnlookup` | exact | 15.43 | 1.17e5x on that step; load-bearing | 1.000000 | ACCEPT |
| 7 | single-threaded BLAS | exact | **1.923 +- 0.004** | 1.08x alone; **8.02x jointly with iter 5** | 1.000000 | ACCEPT |
| 8 | per-gene joblib tasks | exact | 14.34 +- 2.06 (n_jobs=8) | **0.048x — slower** | 1.000000 | **REJECT_SLOW** |
| 9 | chunked dispatch, 1 BLAS thread/worker | exact | **0.686 +- 0.007** (n_jobs=8) | 2.80x vs serial, 20.9x vs iter 8 | 1.000000 | ACCEPT |
| 10 | prefilter runnable genes | exact | 0.686 (n_jobs=8) | folded in; largest effect on the full fixture | 1.000000 | ACCEPT |
| 11 | memoise per-kernel ligand slice | exact | 3.0 (niche-LR, separate axis) | **104.8x** (314.5 s -> 3.0 s) | 1.000000 | ACCEPT |

**Net: 22.5x faster than the controlled baseline and 26.6x faster than the R
reference on the same core count, with the headline parity metric flat at
Pearson 1.000000 through every single iteration.**

On the canonical FULL fixture (848 x 21708 x 7 x 3): R `niche_DE` **852.2 s**
on 16 cores, `pynichede.niche_DE` **32.6 s** on 16 cores = **26.1x**.

Every accepted rewrite is admissibility class **(E) exact algebraic identity**
or a pure scheduling change. **No (B) bounded-epsilon approximation was used
anywhere**, so `MATH.md` carries no perturbation budget — the only numerical
deviations in the port come from f64 rounding and from the one place where the
R reference itself is the less accurate of the two (see `MATH.md` §3).

## Stop reason

Playbook exhausted for this port's pattern. `niche_DE`'s remaining runtime is
dominated by the per-gene IRLS, already a rank-truncated QR at minimum
arithmetic cost, and by the Brown/Cauchy pooling, already vectorised and
O(n_gene x n_celltype^2). `niche_LR_*` is now dominated by the per-candidate
Poisson GLMs, which are irreducible without changing the statistics.
