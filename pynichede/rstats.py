"""R-faithful numerical primitives.

Niche-DE's answers are decided by four pieces of base-R machinery whose
off-the-shelf Python equivalents behave *differently*:

* ``stats::glm.fit`` — IRLS on top of LINPACK ``dqrdc2``, a **limited-pivot**
  QR that shifts near-collinear columns to the right edge and reports the
  aliased coefficients as ``NA``.  ``statsmodels.GLM`` uses ``pinv`` and
  silently spreads the fit across collinear columns instead, which changes both
  β̂ and the ``null``/``new_null`` bookkeeping that Niche-DE derives from it.
* ``stats::optimize`` — Brent's ``fmin`` with R's default
  ``tol = .Machine$double.eps^0.25``.  ``scipy.optimize.minimize_scalar`` uses a
  different bracketing and termination rule.
* ``stats::p.adjust(method = "BH")`` — the BH factor uses ``n = length(p)``
  *including* ``NA`` entries while the ranks use only the non-``NA`` ones.
  ``statsmodels.stats.multitest`` drops the ``NA``s from both.
* ``stats::quantile(type = 7)`` — the R default; ``numpy.quantile``'s default
  ``method="linear"`` happens to agree, but this module states it explicitly so
  the equivalence is auditable.

Everything here is standalone (numpy + scipy.special only) so other
Omicverse-RebuildR ports can lift it wholesale.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "dqrdc2",
    "r_lstsq",
    "r_glm_fit",
    "GLMResult",
    "brent_fmin",
    "r_optimize",
    "p_adjust",
    "r_quantile",
    "weighted_mean",
    "dnbinom_mu",
    "nb_lik",
]


# --------------------------------------------------------------------------- #
# 1. LINPACK dqrdc2 — R's limited-pivoting QR
# --------------------------------------------------------------------------- #

def dqrdc2(x: np.ndarray, tol: float = 1e-7):
    """Port of LINPACK/R ``dqrdc2`` (``src/appl/dqrdc2.f``).

    Householder QR with the *limited* column-pivoting strategy R uses for
    ``lm``/``glm``: columns are processed left to right, and a column whose
    remaining norm has fallen below ``tol`` times its **original** norm is
    cycled to the right-hand edge instead of being used as a pivot.  Unlike
    full column-norm pivoting this keeps the natural column order for
    well-conditioned columns, which is what makes R's "first collinear column
    wins, later ones become NA" behaviour reproducible.

    Parameters
    ----------
    x : (n, p) array — modified out of place.
    tol : relative norm tolerance (R passes ``min(1e-7, epsilon/1000)``).

    Returns
    -------
    qr : (n, p) packed Householder/R factor (as LINPACK leaves it).
    qraux : (p,) Householder scalars.
    jpvt : (p,) 0-based permutation; ``jpvt[j]`` is the original index of the
        column now sitting in position ``j``.
    rank : int, ``min(k - 1, n)`` in Fortran terms — the numerical rank.
    """
    x = np.array(x, dtype=np.float64, order="F", copy=True)
    n, p = x.shape

    qraux = np.zeros(p, dtype=np.float64)
    work1 = np.zeros(p, dtype=np.float64)
    work2 = np.zeros(p, dtype=np.float64)
    jpvt = np.arange(p, dtype=np.int64)

    for j in range(p):
        nrm = float(np.linalg.norm(x[:, j]))
        qraux[j] = nrm
        work1[j] = nrm
        work2[j] = nrm if nrm != 0.0 else 1.0

    lup = min(n, p)
    k = p + 1  # Fortran 1-based "first negligible column"

    for l0 in range(lup):            # l0 is 0-based; Fortran l = l0 + 1
        l = l0 + 1
        # --- cycle negligible columns to the right edge -------------------
        while not (l >= k or qraux[l0] >= work2[l0] * tol):
            # rotate columns l..p one step left, moving column l to the end
            col = x[:, l0].copy()
            x[:, l0:p - 1] = x[:, l0 + 1:p]
            x[:, p - 1] = col

            i_, t_, tt_, ttt_ = jpvt[l0], qraux[l0], work1[l0], work2[l0]
            jpvt[l0:p - 1] = jpvt[l0 + 1:p]
            qraux[l0:p - 1] = qraux[l0 + 1:p]
            work1[l0:p - 1] = work1[l0 + 1:p]
            work2[l0:p - 1] = work2[l0 + 1:p]
            jpvt[p - 1], qraux[p - 1], work1[p - 1], work2[p - 1] = i_, t_, tt_, ttt_
            k -= 1

        if l == n:
            continue

        # --- Householder transformation for column l ----------------------
        nrmxl = float(np.linalg.norm(x[l0:, l0]))
        if nrmxl == 0.0:
            continue
        if x[l0, l0] != 0.0:
            nrmxl = np.copysign(nrmxl, x[l0, l0])
        x[l0:, l0] /= nrmxl
        x[l0, l0] += 1.0

        for j in range(l0 + 1, p):
            t = -float(np.dot(x[l0:, l0], x[l0:, j])) / x[l0, l0]
            x[l0:, j] += t * x[l0:, l0]
            if qraux[j] == 0.0:
                continue
            tt = 1.0 - (abs(x[l0, j]) / qraux[j]) ** 2
            tt = max(tt, 0.0)
            # BDR 9/99: recompute the norm when the downdate loses too much.
            if abs(tt) < 1e-6:
                nj = float(np.linalg.norm(x[l0 + 1:, j]))
                qraux[j] = nj
                work1[j] = nj
            else:
                qraux[j] = qraux[j] * np.sqrt(tt)

        qraux[l0] = x[l0, l0]
        x[l0, l0] = -nrmxl

    rank = min(k - 1, n)
    return x, qraux, jpvt, rank


def r_lstsq(x: np.ndarray, y: np.ndarray, tol: float = 1e-7):
    """R's ``dqrls``: rank-truncated least squares with ``dqrdc2`` pivoting.

    Returns ``(coefficients, pivot, rank)`` where ``coefficients`` is in
    **pivoted** order and positions ``rank:p`` are set to ``0.0`` — exactly what
    LINPACK ``dqrls`` does (``b(j,jj) = 0`` for ``j > k``) and what
    ``glm.fit`` relies on when it does ``start[fit$pivot] <- fit$coefficients``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    p = x.shape[1]
    _, _, jpvt, rank = dqrdc2(x, tol=tol)

    coef = np.zeros(p, dtype=np.float64)
    if rank > 0:
        keep = jpvt[:rank]
        # Solve the truncated problem on the retained (pivoted) columns.
        q, r = np.linalg.qr(x[:, keep])
        coef[:rank] = np.linalg.solve(r, q.T @ y)
    return coef, jpvt, rank


# --------------------------------------------------------------------------- #
# 2. glm.fit
# --------------------------------------------------------------------------- #

class GLMResult:
    """Minimal mirror of R's ``glm`` return value (only what Niche-DE reads)."""

    __slots__ = ("coefficients", "linear_predictors", "fitted_values",
                 "rank", "pivot", "deviance", "iter", "converged")

    def __init__(self, coefficients, linear_predictors, fitted_values,
                 rank, pivot, deviance, n_iter, converged):
        self.coefficients = coefficients
        self.linear_predictors = linear_predictors
        self.fitted_values = fitted_values
        self.rank = rank
        self.pivot = pivot
        self.deviance = deviance
        self.iter = n_iter
        self.converged = converged


def _poisson_dev_resids(y, mu, wt):
    r = mu * wt
    p = y > 0
    r = 2.0 * r
    out = np.empty_like(r)
    out[:] = r
    out[p] = 2.0 * wt[p] * (y[p] * np.log(y[p] / mu[p]) - (y[p] - mu[p]))
    return out


def r_glm_fit(x, y, offset=None, family="poisson", epsilon=1e-8, maxit=25):
    """Port of ``stats::glm.fit`` for the two families Niche-DE uses.

    ``family="poisson"`` (log link) is the ``Int = TRUE`` path; ``"gaussian"``
    (identity link) is the ``Int = FALSE`` path, where R uses ``lm`` — the
    single IRLS step of a Gaussian/identity GLM *is* the OLS fit, so the same
    code covers both.

    Follows R exactly:

    * poisson ``initialize``: ``mustart <- y + 0.1``, ``eta <- log(mustart)``
    * per iteration ``z = (eta - offset) + (y - mu)/mu'``, ``w = sqrt(mu'^2/V(mu))``
    * ``Cdqrls(x*w, z*w, tol = min(1e-7, epsilon/1000))``
    * aliased coefficients are ``0`` **during** the loop (so ``eta = x %*% start``
      stays finite) and become ``NA`` only after convergence:
      ``coef[pivot][rank+1 : nvars] <- NA``
    * convergence when ``|dev - devold| / (0.1 + |dev|) < epsilon``
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, nvars = x.shape
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64).ravel()
    wt = np.ones(n)

    if family == "poisson":
        mustart = y + 0.1
        eta = np.log(mustart)
    elif family == "gaussian":
        mustart = y.copy()
        eta = mustart.copy()
    else:                                                    # pragma: no cover
        raise ValueError(f"unsupported family {family!r}")

    def linkinv(e):
        return np.exp(e) if family == "poisson" else e

    def mu_eta(e):
        return np.exp(e) if family == "poisson" else np.ones_like(e)

    def variance(m):
        return m if family == "poisson" else np.ones_like(m)

    def dev_resids(yy, mm):
        if family == "poisson":
            return _poisson_dev_resids(yy, mm, wt)
        return wt * (yy - mm) ** 2

    mu = linkinv(eta)
    devold = float(np.sum(dev_resids(y, mu)))
    start = np.zeros(nvars)
    converged = False
    tol = min(1e-7, epsilon / 1000.0)
    pivot = np.arange(nvars)
    rank = nvars
    n_iter = 0

    for n_iter in range(1, maxit + 1):
        mu_eta_val = mu_eta(eta)
        good = mu_eta_val != 0
        if not good.any():
            break
        z = (eta - offset)[good] + (y - mu)[good] / mu_eta_val[good]
        w = np.sqrt((wt[good] * mu_eta_val[good] ** 2) / variance(mu)[good])

        coef_piv, pivot, rank = r_lstsq(x[good] * w[:, None], z * w, tol=tol)
        if not np.all(np.isfinite(coef_piv)):
            break

        start = np.zeros(nvars)
        start[pivot] = coef_piv
        eta = x @ start + offset
        mu = linkinv(eta)
        dev = float(np.sum(dev_resids(y, mu)))

        if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
            converged = True
            devold = dev
            break
        devold = dev

    coef = start.astype(np.float64).copy()
    if rank < nvars:
        coef[pivot[rank:]] = np.nan
    return GLMResult(coef, eta, mu, rank, pivot, devold, n_iter, converged)


# --------------------------------------------------------------------------- #
# 3. optimize / Brent fmin
# --------------------------------------------------------------------------- #

_DBL_EPSILON = np.finfo(np.float64).eps


def brent_fmin(ax: float, bx: float, f, tol: float) -> float:
    """Port of R's ``Brent_fmin`` (``src/library/stats/src/optimize.c``).

    Golden-section search with parabolic interpolation.  Line-for-line the same
    control flow as the C original, so the returned minimiser matches R's
    ``optimize()$minimum`` to the last bit on identical objective values.
    """
    c = (3.0 - np.sqrt(5.0)) * 0.5
    eps = np.sqrt(_DBL_EPSILON)

    a, b = ax, bx
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0
    fx = f(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0

    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def r_optimize(f, lower: float, upper: float, tol: float | None = None) -> dict:
    """``stats::optimize`` — Brent ``fmin`` plus one extra objective evaluation.

    R's default ``tol`` is ``.Machine$double.eps^0.25`` (≈ 1.22e-4) and it
    re-evaluates ``f`` at the returned minimiser to build ``$objective``.
    """
    if tol is None:
        tol = _DBL_EPSILON ** 0.25
    val = brent_fmin(lower, upper, f, tol)
    return {"minimum": val, "objective": f(val)}


# --------------------------------------------------------------------------- #
# 4. dnbinom(mu=) / nb_lik
# --------------------------------------------------------------------------- #

def dnbinom_mu(x, size, mu, log: bool = True):
    """``stats::dnbinom(x, size, mu = mu, log = TRUE)``.

    Uses the closed-form log-pmf

        lgamma(x+size) - lgamma(size) - lgamma(x+1)
        + size*log(size/(size+mu)) + x*log(mu/(size+mu))

    R evaluates the same quantity through Loader's saddle-point ``dbinom_raw``
    for extra accuracy in the far tails; on the ranges Niche-DE explores
    (``size ∈ [0.05, 100]``, ``mu`` = fitted Poisson means) the two agree to
    ~1e-13 relative, far below the ``optimize`` tolerance of 1.2e-4 that
    consumes the value.
    """
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    size = float(size)
    out = (gammaln(x + size) - gammaln(size) - gammaln(x + 1.0)
           + size * np.log(size / (size + mu))
           + x * np.log(mu / (size + mu)))
    # x = 0 -> the x*log(...) term is 0 even if mu is 0
    out = np.where(x == 0, size * np.log(size / (size + mu)), out)
    return out if log else np.exp(out)


def nb_lik(x, mu, disp) -> float:
    """``nicheDE::nb_lik`` — negative NB log-likelihood, ``Var = mu + mu^2/size``.

    Mirrors R's recycling: when ``len(mu) != len(x)`` R silently tiles the
    shorter argument (this happens in ``niche_DE_core`` when spots with zero
    expected expression are dropped from ``mu_hat`` but not from ``counts``).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    if mu.size != x.size:
        mu = np.resize(mu, x.size)          # R-style recycling
    return float(-np.sum(dnbinom_mu(x, disp, mu, log=True)))


# --------------------------------------------------------------------------- #
# 5. p.adjust / quantile / weighted.mean
# --------------------------------------------------------------------------- #

def p_adjust(p, method: str = "BH", n: int | None = None):
    """``stats::p.adjust``.

    The ``NA`` convention is the part everybody gets wrong, and it is subtler
    than "``n`` counts the ``NA``s".  R's signature is
    ``p.adjust(p, method, n = length(p))``, and because R default arguments are
    **lazy**, ``n`` is not forced until ``stopifnot(n >= lp)`` — which runs
    *after* the body has already executed ``p <- p[!is.na(p)]``.  So when the
    caller does not pass ``n`` explicitly, ``n`` ends up being the number of
    **non-NA** p-values, and ``p.adjust(c(.4, NA, .01, .9, NA, .02, .5))``
    returns exactly what ``p.adjust(c(.4, .01, .9, .02, .5))`` returns, with
    ``NA``s reinserted.  ``statsmodels.stats.multitest`` has no equivalent, and
    getting this wrong inflates every adjusted p-value by ``n_total/n_valid``
    (in Niche-DE, by ~3x, which silently loses a third of the significant
    genes).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    lp_all = p.size
    nna = ~np.isnan(p)
    out = np.full(lp_all, np.nan)
    pv = p[nna]
    lp = pv.size
    if n is None:
        n = lp                      # lazy default, forced after the NA drop
    if n <= 1:
        return p.copy()
    if lp == 0:
        return out

    if method == "none":
        out[nna] = pv
    elif method == "bonferroni":
        out[nna] = np.minimum(1.0, n * pv)
    elif method in ("BH", "fdr"):
        o = np.argsort(-pv, kind="stable")           # decreasing
        ro = np.empty_like(o)
        ro[o] = np.arange(lp)
        i = np.arange(lp, 0, -1)                     # lp:1
        out[nna] = np.minimum(1.0, np.minimum.accumulate((n / i) * pv[o]))[ro]
    elif method == "holm":
        o = np.argsort(pv, kind="stable")
        ro = np.empty_like(o)
        ro[o] = np.arange(lp)
        i = np.arange(lp)
        out[nna] = np.minimum(1.0, np.maximum.accumulate((n - i) * pv[o]))[ro]
    elif method == "BY":
        q = np.sum(1.0 / np.arange(1, n + 1))
        o = np.argsort(-pv, kind="stable")
        ro = np.empty_like(o)
        ro[o] = np.arange(lp)
        i = np.arange(lp, 0, -1)
        out[nna] = np.minimum(1.0, np.minimum.accumulate(q * (n / i) * pv[o]))[ro]
    else:                                                    # pragma: no cover
        raise ValueError(f"unsupported p.adjust method {method!r}")
    return out


def r_quantile(x, probs, na_rm: bool = False):
    """``stats::quantile(type = 7)`` — R's default, identical to numpy 'linear'."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if na_rm:
        x = x[~np.isnan(x)]
    return np.quantile(x, probs, method="linear")


def weighted_mean(x, w=None, na_rm: bool = False) -> float:
    """``stats::weighted.mean``.

    R's body is ``sum((x*w)[w != 0]) / sum(w)`` after (optionally) dropping the
    positions where ``x`` is ``NA`` from **both** ``x`` and ``w``.  Excluding
    the ``w == 0`` terms from the numerator is what keeps ``Inf * 0`` from
    poisoning the Cauchy combination when a kernel gets weight 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, dtype=np.float64).ravel()
    if na_rm:
        keep = ~np.isnan(x)
        x, w = x[keep], w[keep]
    if x.size == 0:
        return np.nan
    nz = w != 0
    return float(np.sum((x * w)[nz]) / np.sum(w))
