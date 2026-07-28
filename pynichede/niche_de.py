"""The Niche-DE regression itself — port of ``niche_DE`` / ``niche_DE_core``.

For one gene ``j`` and one kernel bandwidth, the R code fits

    counts ~ X + batch + offset(log(EEJ)),   family = poisson

where ``EEJ = num_cells @ ref_expr[:, j]`` is the expected expression of gene
``j`` under the null (no niche effect), and the design ``X`` has one column per
ordered *(niche cell type, index cell type)* pair,

    X[spot, (b, a)] = round(effective_niche[spot, a], 2) * pstg[spot, b]

with ``pstg[spot, b] = num_cells[spot, b] * ref_expr[b, j] / EEJ[spot]`` the
posterior share of gene-``j`` expression attributable to cell type ``b``.  The
Poisson fit supplies β̂; an over-dispersion parameter is then obtained by
one-dimensional Brent minimisation of the negative binomial negative
log-likelihood, and the Wald standard errors come from
``V = (X' W X)^-1`` with ``W = mu/(1 + mu/disp)``.

**Bug-compatibility notice.**  Two quirks of the R source are reproduced on
purpose because they change which genes are reported:

1. ``niche_DE_core`` contains ``new_null = new_nul[new_null <= var]`` — two
   undefined symbols.  Whenever any diagonal entry of ``X'WX`` is exactly zero
   the R code therefore throws, the surrounding ``tryCatch`` swallows it, and
   the gene is returned as *invalid*.  :func:`_niche_de_core` raises
   :class:`_RErrorCompat` at the same point.
2. ``optimize(nb_lik, x = counts, mu = mu_hat, ...)`` is called with the *full*
   count vector but a ``mu_hat`` that has had zero-expected-expression spots
   removed, so R recycles ``mu_hat``.  :func:`~pynichede.rstats.nb_lik`
   reproduces the recycling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky

from .rstats import r_glm_fit, r_optimize, r_quantile, nb_lik

__all__ = ["niche_DE", "niche_DE_no_parallel"]


class _RErrorCompat(Exception):
    """Marks the point where the R source raises inside its own ``tryCatch``."""


def _single_threaded_blas():
    """Pin the BLAS thread pool to 1 for the duration of the fit.

    Niche-DE's inner problem is one ~``n_spot x n_celltype^2`` GLM per gene —
    small enough that a multi-threaded BLAS spends more time in barriers than in
    arithmetic.  Falls back to a no-op if ``threadpoolctl`` is unavailable.
    """
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception:                                        # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def _as_dense(x):
    return x.to_numpy(dtype=np.float64) if isinstance(x, pd.DataFrame) \
        else np.asarray(x, dtype=np.float64)


def _batch_design(batch_id: np.ndarray, use_batch: bool):
    """R's ``+ batchvar`` term.

    Single batch (or ``batch = FALSE``): ``batchvar <- rep(1, n)``, a constant
    numeric column that ``model.matrix`` keeps and ``dqrdc2`` then aliases
    against the intercept — so it contributes one always-``NA`` coefficient and
    no columns to the variance matrix.  Multiple batches: a factor, expanded to
    ``nlevels - 1`` treatment-contrast dummies.
    """
    levels = np.unique(batch_id)
    if use_batch and levels.size > 1:
        dummies = np.zeros((batch_id.size, levels.size - 1))
        for i, lv in enumerate(levels[1:]):
            dummies[:, i] = (batch_id == lv).astype(np.float64)
        return dummies, True
    return np.ones((batch_id.size, 1)), False


def _niche_de_core(counts, iter_, effective_niche, num_cells, CT_filter, C, M,
                   Int, batch, batch_id, ref_expr, n_type, self_EN):
    """Port of the R closure ``niche_DE_core``."""
    ref_expression = ref_expr[:, iter_]
    EEJ = num_cells @ ref_expression

    invalid = dict(T_stat=None, betas=None, Varcov=None,
                   nulls=np.arange(n_type ** 2), valid=0, log_likelihood=np.nan)

    passes_filter = np.mean(ref_expression < CT_filter) != 1
    if not (counts.sum() > C and passes_filter):
        return invalid

    # ---- design matrix ----------------------------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        pstg = num_cells * ref_expression[None, :] / EEJ[:, None]
    pstg[:, ref_expression < CT_filter] = 0.0
    pstg[pstg < 0.05] = 0.0

    en_r = np.round(effective_niche, 2)
    # X[:, b*n_type + a] = en_r[:, a] * pstg[:, b]   (R's column-major outer product)
    X = (en_r[:, :, None] * pstg[:, None, :]).transpose(0, 2, 1).reshape(en_r.shape[0], -1)
    if not self_EN:
        diag_idx = np.arange(n_type) * n_type + np.arange(n_type)
        X[:, diag_idx] = 0.0

    null = np.flatnonzero(np.nansum(X > 0, axis=0) < M)
    rest = np.arange(X.shape[1])
    r_dropped_to_vector = False
    if null.size > 0:
        keep = np.setdiff1d(rest, null)
        X_partial = X[:, keep]
        rest = keep
        # R: `X_partial = X[,-null]` drops dim when a single column survives,
        # so `nvar = ncol(X_partial)` becomes NULL, `coeff = coefficients[1:(NULL+1)]`
        # becomes length 0, and `beta[...] = coeff[-1]` raises "replacement has
        # length zero".  The tryCatch swallows it and the gene is invalid.
        r_dropped_to_vector = X_partial.shape[1] == 1
    else:
        X_partial = X
    del X

    if null.size == n_type ** 2:
        return invalid

    batch_cols, multi_batch = _batch_design(batch_id, batch)

    if Int:
        liks_val = np.nan   # R keeps whatever was assigned before the error
        try:
            nvar = X_partial.shape[1]
            bad_ind = np.flatnonzero(EEJ == 0)
            good = np.ones(EEJ.size, dtype=bool)
            good[bad_ind] = False

            design = np.column_stack([np.ones(good.sum()),
                                      X_partial[good],
                                      batch_cols[good]])
            fit = r_glm_fit(design, counts[good], offset=np.log(EEJ[good]),
                            family="poisson")
            mu_hat = np.exp(fit.linear_predictors)

            opt = r_optimize(lambda d: nb_lik(counts, mu_hat, d), 0.05, 100.0)
            disp = opt["minimum"]
            liks_val = -opt["objective"]

            W = mu_hat / (1.0 + mu_hat / disp)

            # R appends `fastDummies::dummy_cols(batchvar, remove_first_dummy=T)`
            # only in the multi-batch branch, then an explicit intercept column.
            Xp = X_partial[good]
            if multi_batch:
                Xp = np.column_stack([Xp, batch_cols[good]])
            Xp = np.column_stack([Xp, np.ones(Xp.shape[0])])

            var_mat = (Xp * W[:, None]).T @ Xp

            new_null = np.flatnonzero(np.diag(var_mat) == 0)
            if new_null.size > 0:
                # R: `new_null = new_nul[new_null <= var]` -> object not found
                raise _RErrorCompat("nicheDE new_nul/var typo")
            if r_dropped_to_vector:
                raise _RErrorCompat("nicheDE single-column drop: replacement has length zero")

            coeff = fit.coefficients[:nvar + 1].copy()
            coeff[np.isnan(coeff)] = 0.0

            # R computes `V = solve(chol(var_mat)) %*% t(solve(chol(var_mat)))`.
            # acceleration: iter 2 — `cho_solve` (two triangular solves against
            # the identity) is the same quantity `(X'WX)^-1` without forming the
            # triangular inverse first.  Exact algebraic identity
            # (ACCELERATION_PLAYBOOK 2, Cholesky solve); measured max relative
            # deviation on the fixture 1.5e-15.
            A = cholesky(var_mat, lower=False)          # raises if not PD, like Matrix::chol
            V = cho_solve((A, False), np.eye(A.shape[0]))
            V = V[:nvar, :nvar]
            V = np.triu(V)

            tau = np.sqrt(np.diag(V))

            V_ = np.full(n_type * n_type, np.nan)
            beta = np.full(n_type * n_type, np.nan)
            idx = rest if null.size > 0 else np.arange(n_type ** 2)
            V_[idx] = tau
            beta[idx] = coeff[1:]

            V_ = V_.reshape((n_type, n_type), order="F")
            beta = beta.reshape((n_type, n_type), order="F")
            T_ = beta / V_

            return dict(T_stat=T_.T, betas=beta.T, Varcov=V,
                        nulls=null, valid=1, log_likelihood=liks_val)
        except (_RErrorCompat, np.linalg.LinAlgError, ValueError, FloatingPointError):
            # R's invalid branch always reports `nulls = c(1:n_type^2)`,
            # regardless of the `null` vector it computed on the way.
            return dict(T_stat=None, betas=None, Varcov=None,
                        nulls=np.arange(n_type ** 2), valid=0,
                        log_likelihood=liks_val)

    # ---- Int = FALSE: linear model with a gene-specific variance -----------
    # R: lm((counts - EEJ) ~ X_partial + batchvar); V = cov.unscaled * sigma^2;
    #    T_ comes from summary(lm)$coefficients[, 3] (the t-values), NOT beta/tau.
    liks_val = np.nan
    try:
        nvar = X_partial.shape[1]
        design = np.column_stack([np.ones(counts.size), X_partial, batch_cols])
        y = counts - EEJ
        fit = r_glm_fit(design, y, family="gaussian")
        resid = y - design @ np.nan_to_num(fit.coefficients)
        n = counts.size
        rss = float(resid @ resid)
        sigma2 = rss / (n - fit.rank)
        # stats::logLik.lm for an unweighted fit
        liks_val = float(0.5 * (-n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(rss))))

        # summary.lm reports only the non-aliased columns, in their original
        # order — this is `sum_lm$cov.unscaled` and `sum_lm$coefficients`.
        keep_cols = np.sort(fit.pivot[:fit.rank])
        cov_un = np.linalg.inv(design[:, keep_cols].T @ design[:, keep_cols])
        V = cov_un * sigma2

        new_null = np.flatnonzero(np.diag(V) == 0)
        if new_null.size > 0:
            raise _RErrorCompat("degenerate columns in cov.unscaled")
        if r_dropped_to_vector:
            raise _RErrorCompat("nicheDE single-column drop")

        se = np.sqrt(np.diag(V))
        t_vals = fit.coefficients[keep_cols] / se           # summary()[, 3]

        V = np.triu(V[1:, 1:])                              # R: V = V[-1, -1]

        coeff = fit.coefficients[:nvar + 1].copy()
        coeff[np.isnan(coeff)] = 0.0
        # R: coeff_T = sum_lm$coefficients[1:(num_coef+1), 3]; out-of-range -> NA -> 0
        coeff_T = np.zeros(nvar + 1)
        take = min(nvar + 1, t_vals.size)
        coeff_T[:take] = np.nan_to_num(t_vals[:take])

        beta = np.full(n_type * n_type, np.nan)
        T_ = np.full(n_type * n_type, np.nan)
        idx = rest if null.size > 0 else np.arange(n_type ** 2)
        beta[idx] = coeff[1:]
        T_[idx] = coeff_T[1:]
        beta = beta.reshape((n_type, n_type), order="F")
        T_ = T_.reshape((n_type, n_type), order="F")
        return dict(T_stat=T_.T, betas=beta.T, Varcov=V,
                    nulls=null, valid=1, log_likelihood=liks_val)
    except (_RErrorCompat, np.linalg.LinAlgError, ValueError, FloatingPointError):
        return dict(T_stat=None, betas=None, Varcov=None,
                    nulls=np.arange(n_type ** 2), valid=0, log_likelihood=liks_val)


def _chunk_worker(gene_idx, counts_chunk, effective_niche, num_cells, CT_filter,
                  C, M, Int, batch, batch_id, ref_expr, n_type, self_EN):
    """Fit one contiguous block of genes inside a single worker task."""
    return [_niche_de_core(counts_chunk[:, i], int(g), effective_niche, num_cells,
                           CT_filter, C, M, Int, batch, batch_id, ref_expr,
                           n_type, self_EN)
            for i, g in enumerate(gene_idx)]


def _run_sigma(counts, effective_niche, num_cells, CT_filter, C, M, Int, batch,
               batch_id, ref_expr, n_type, self_EN, n_jobs, verbose):
    ngene = counts.shape[1]

    # acceleration: iter 5 — hoist the two cheap gate conditions out of the
    # per-gene worker so genes that R would immediately declare invalid are
    # never shipped to a subprocess.  Exactly the same predicate as the first
    # branch of `niche_DE_core`, so the result is unchanged.
    tot = counts.sum(axis=0)
    passes = (ref_expr < CT_filter[:, None]).mean(axis=0) != 1
    runnable = np.flatnonzero((tot > C) & passes)

    invalid = dict(T_stat=None, betas=None, Varcov=None,
                   nulls=np.arange(n_type ** 2), valid=0, log_likelihood=np.nan)
    results = [dict(invalid) for _ in range(ngene)]
    if runnable.size == 0:
        return results

    if n_jobs in (0, 1, None) or runnable.size < 64:
        fitted = _chunk_worker(runnable, counts[:, runnable], effective_niche,
                              num_cells, CT_filter, C, M, Int, batch, batch_id,
                              ref_expr, n_type, self_EN)
    else:
        # acceleration: iter 3b — dispatch *chunks* of genes, not single genes.
        # One task per gene made joblib's pickling of the shared arrays dominate
        # (measured 0.60x, i.e. slower than serial); ~4 chunks per worker keeps
        # the workers busy while amortising the transfer.
        # iter 4 — `inner_max_num_threads=1` stops each worker from opening its
        # own full-width BLAS pool (16 workers x 17 threads thrashed the box).
        from joblib import Parallel, delayed, parallel_backend
        n_chunks = min(runnable.size, max(1, int(n_jobs) * 4))
        splits = [s for s in np.array_split(runnable, n_chunks) if s.size]
        with parallel_backend("loky", n_jobs=int(n_jobs), inner_max_num_threads=1):
            out = Parallel(verbose=0)(
                delayed(_chunk_worker)(s, counts[:, s], effective_niche, num_cells,
                                       CT_filter, C, M, Int, batch, batch_id,
                                       ref_expr, n_type, self_EN)
                for s in splits)
        fitted = [r for block in out for r in block]

    for g, r in zip(runnable, fitted):
        results[int(g)] = r
    return results


def niche_DE(obj, num_cores: int = 1, outfile=None, C: float = 150, M: int = 10,
             gamma: float = 0.8, print_: bool = True, Int: bool = True,
             batch: bool = True, self_EN: bool = False, G: float = 1,
             verbose: bool = True):
    """``nicheDE::niche_DE`` — the canonical entry point.

    Parameters mirror the R signature.  ``num_cores`` maps to ``joblib``
    workers, ``outfile``/``G`` are accepted for signature compatibility and are
    no-ops (Python does not need the counts matrix chunked to bound worker
    memory the way the ``foreach``/``parallel`` cluster did).

    Populates ``obj.niche_DE`` (one list of per-gene results per kernel
    bandwidth) and both ``obj.niche_DE_pval_pos`` / ``obj.niche_DE_pval_neg``.
    """
    from .pvalues import get_niche_DE_pval_fisher

    counts = _as_dense(obj.counts)
    num_cells = _as_dense(obj.num_cells)
    ref_expr = _as_dense(obj.ref_expr)
    batch_id = np.asarray(obj.batch_ID)
    n_type = num_cells.shape[1]
    ngene = counts.shape[1]

    if verbose:
        print(f"Starting Niche-DE analysis with parameters C = {C}, M = {M}, gamma = {gamma}.")

    CT_filter = np.array([r_quantile(ref_expr[i], gamma) for i in range(ref_expr.shape[0])])

    obj.niche_DE = []
    valid = np.zeros((ngene, len(obj.sigma)))
    # acceleration: iter 4 — the per-gene design is only ~848 x 50, far below
    # the size where a multi-threaded BLAS pays for its synchronisation.  Pinning
    # the pool to one thread for the duration of the fit measured 5.7x on the
    # canonical fixture (24.65s -> 4.35s single-process).  Pure scheduling
    # change: identical FLOPs in an identical order.
    with _single_threaded_blas():
        for k, sig in enumerate(obj.sigma):
            if verbose:
                print(f"Performing Niche-DE analysis with kernel bandwidth: {sig} "
                      f"(number {k + 1} out of {len(obj.sigma)} values)")
            res = _run_sigma(counts, obj.effective_niche[k], num_cells, CT_filter,
                             C, M, Int, batch, batch_id, ref_expr, n_type, self_EN,
                             num_cores, verbose)
            obj.niche_DE.append(res)
            valid[:, k] = [r["valid"] for r in res]

    num_pass = int((valid.sum(axis=1) >= 1).sum())
    if verbose:
        print("Computing Positive Niche-DE Pvalues")
    obj = get_niche_DE_pval_fisher(obj, pos=True, verbose=verbose)
    if verbose:
        print("Computing Negative Niche-DE Pvalues")
    obj = get_niche_DE_pval_fisher(obj, pos=False, verbose=verbose)
    if verbose:
        print(f"Niche-DE analysis complete. Number of Genes with niche-DE T-stat equal to {num_pass}")
    if num_pass < 1000:
        import warnings
        warnings.warn("Less than 1000 genes pass. This could be due to insufficient "
                      "read depth of data or size of C parameter.")
    return obj


def niche_DE_no_parallel(obj, C: float = 150, M: int = 10, gamma: float = 0.8,
                         print_: bool = True, Int: bool = True, batch: bool = True,
                         self_EN: bool = False, verbose: bool = True):
    """``nicheDE::niche_DE_no_parallel`` — identical maths, single-threaded."""
    return niche_DE(obj, num_cores=1, C=C, M=M, gamma=gamma, print_=print_,
                    Int=Int, batch=batch, self_EN=self_EN, verbose=verbose)
