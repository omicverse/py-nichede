"""Clean-room reimplementation of the two ``poolr`` entry points Niche-DE uses.

Niche-DE pools the up-to-``n_celltype^2`` correlated per-interaction p-values of
one gene into a gene-level (and cell-type-level) p-value with **Brown's method**
— ``poolr::fisher(p, side = 1, R = ..., adjust = "generalized")`` — where ``R``
is produced by ``poolr::mvnconv(cor_matrix, side = 1, target = "m2lp")``.

``poolr`` is GPL-2+; ``py-nichede`` is MIT (matching upstream ``nicheDE``), so
poolr's shipped ``mvnlookup`` data table is **not** vendored.  It is instead
re-derived from its own definition:

    mvnlookup[rho, target_side] = Cov( g(Z1), g(Z2) ),   (Z1, Z2) ~ BVN(rho)

with, for one-sided p-values ``p = 1 - Phi(Z)`` and two-sided
``p = 2 (1 - Phi(|Z|))``,

    target "p"       g = p
    target "z"       g = Phi^-1(1 - p)
    target "chisq1"  g = qchisq(1 - p, df = 1)
    target "m2lp"    g = -2 log p

The covariance is evaluated with **Mehler's formula**.  Expanding ``g`` in the
orthonormal Hermite basis ``psi_n = He_n / sqrt(n!)`` with coefficients
``alpha_n = E[g(Z) psi_n(Z)]``, Mehler's kernel expansion of the bivariate
normal density gives the closed power series

    Cov( g(Z1), g(Z2) ) = sum_{n >= 1} alpha_n^2 rho^n .

One 1-D quadrature therefore produces the covariance at *every* rho at once,
and the series converges to machine precision within ~50 terms (verified
against the exact identities ``E[-2 log p] = 2`` and ``Var[-2 log p] = 4``).

.. note:: **This module is more accurate than poolr, and that is a measured
   divergence, not a bug.**  poolr's shipped ``mvnlookup`` is itself a
   numerically-integrated approximation stored to 4 decimals: its ``chisq1_2``
   column, which has the exact closed form ``2 rho^2``, deviates from it by up
   to ``6.28e-4``.  The re-derived table here agrees with poolr's to
   ``<= ~1e-3`` everywhere and is exact where a closed form exists
   (``z_1 = rho``, ``chisq1_2 = 2 rho^2``).  ``tests/test_poolr_table.py``
   pins both facts.  The downstream effect on Niche-DE's pooled p-values is
   quantified in ``RECONSTRUCTION_REPORT.md`` Section 6.
"""

from __future__ import annotations

import functools

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtri_exp
from scipy.stats import chi2, norm

__all__ = ["mvnconv", "fisher_generalized", "build_mvnlookup", "MVN_RHOS"]

# poolr's grid: rho = 1.000 down to -0.990 in steps of 0.001 (1991 rows).
MVN_RHOS = np.round(np.arange(1000, -991, -1) / 1000.0, 3)

_TARGETS = ("m2lp", "z", "chisq1", "p")


# --------------------------------------------------------------------------- #
# Table derivation
# --------------------------------------------------------------------------- #

def _transform(z: np.ndarray, target: str, side: int) -> np.ndarray:
    """Map standard-normal draws to the requested pooling statistic.

    Everything is routed through ``log p`` so the deep tails (where ``p``
    underflows to 0) stay finite: ``Phi^-1(1-p)`` is evaluated as
    ``-ndtri_exp(log p)`` and ``qchisq(1-p, 1)`` as ``ndtri_exp(log p - log 2)^2``.
    """
    if side == 1:
        logp = norm.logsf(z)                     # log(1 - Phi(z)), stable
    elif side == 2:
        logp = np.log(2.0) + norm.logsf(np.abs(z))
    else:                                                    # pragma: no cover
        raise ValueError("side must be 1 or 2")
    logp = np.minimum(logp, 0.0)                 # side=2 at z=0 gives p=1 exactly

    if target == "m2lp":
        return -2.0 * logp
    if target == "p":
        return np.exp(logp)
    if target == "z":
        # Phi^-1(1 - p) = -Phi^-1(p)
        return -ndtri_exp(logp)
    if target == "chisq1":
        # F_chisq1^-1(1 - p) = (Phi^-1(1 - p/2))^2
        return ndtri_exp(logp - np.log(2.0)) ** 2
    raise ValueError(f"unknown target {target!r}")           # pragma: no cover


@functools.lru_cache(maxsize=1)
def _quad_nodes():
    """Composite Gauss–Legendre panels over z, refined geometrically at 0.

    The ``side = 2`` ``z`` target has a mild ``sqrt(log(1/|z|))`` singularity at
    the origin, so the panel edges are packed geometrically towards 0 (and
    mirrored) instead of being uniform.
    """
    x, w = leggauss(40)
    right = np.concatenate([[0.0], np.geomspace(1e-10, 14.0, 400)])
    edges = np.unique(np.concatenate([-right[::-1], right]))
    zs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        zs.append(0.5 * (b - a) * x + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w)
    z = np.concatenate(zs)
    wq = np.concatenate(ws) * norm.pdf(np.concatenate(zs))
    return z, wq


def _hermite_coefficients(target: str, side: int, n_terms: int = 2000) -> np.ndarray:
    """``alpha_n = E[g(Z) psi_n(Z)]`` for ``n = 1 .. n_terms``."""
    z, wq = _quad_nodes()
    g = _transform(z, target, side)
    gw = g * wq

    alpha = np.empty(n_terms + 1)
    psi_prev = np.zeros_like(z)
    psi = np.ones_like(z)
    alpha[0] = float(np.sum(gw * psi))
    for n in range(n_terms):
        psi_next = (z * psi - np.sqrt(n) * psi_prev) / np.sqrt(n + 1)
        psi_prev, psi = psi, psi_next
        alpha[n + 1] = float(np.sum(gw * psi))
    return alpha[1:]


def build_mvnlookup(n_terms: int = 2000) -> np.ndarray:
    """Re-derive poolr's ``mvnlookup`` table via Mehler's formula.

    Returns
    -------
    (1991, 9) array with columns
    ``rhos, m2lp_1, m2lp_2, z_1, z_2, chisq1_1, chisq1_2, p_1, p_2``.
    """
    out = np.empty((MVN_RHOS.size, 9), dtype=np.float64)
    out[:, 0] = MVN_RHOS
    col = 1
    for target in _TARGETS:
        for side in (1, 2):
            a = _hermite_coefficients(target, side, n_terms)
            # Cov(rho) = sum_{n>=1} alpha_n^2 rho^n   (no constant term)
            powers = np.vander(MVN_RHOS, n_terms + 1, increasing=True)[:, 1:]
            out[:, col] = powers @ (a ** 2)
            col += 1
    return out


@functools.lru_cache(maxsize=1)
def _lookup() -> np.ndarray:
    """The table as poolr exposes it: rounded to 4 decimals."""
    tbl = build_mvnlookup()
    tbl[:, 1:] = np.round(tbl[:, 1:], 4)
    return tbl


# --------------------------------------------------------------------------- #
# mvnconv
# --------------------------------------------------------------------------- #

def mvnconv(R, side: int = 2, target: str = "m2lp", cov2cor: bool = False):
    """``poolr::mvnconv``.

    Rounds ``R`` to 3 decimals, floors it at ``-0.99`` (poolr's table stops
    there), and looks the covariance up per element.
    """
    if target not in _TARGETS:                               # pragma: no cover
        raise ValueError(f"unknown target {target!r}")
    if side not in (1, 2):                                   # pragma: no cover
        raise ValueError("side must be 1 or 2")

    column = _TARGETS.index(target) * 2 + 1 + (1 if side == 2 else 0)

    R = np.asarray(R, dtype=np.float64)
    Rr = np.round(R, 3)
    Rr = np.where(Rr < -0.99, -0.99, Rr)

    tbl = _lookup()
    # index into the descending rho grid: rho = 1.000 - idx/1000
    idx = np.rint((1.0 - Rr) * 1000.0).astype(np.int64)
    idx = np.clip(idx, 0, tbl.shape[0] - 1)
    covs = tbl[idx, column]

    if cov2cor:
        covs = covs / tbl[0, column]
    return covs


# --------------------------------------------------------------------------- #
# Brown's method
# --------------------------------------------------------------------------- #

def _near_pd(A: np.ndarray, eig_tol: float = 1e-6, conv_tol: float = 1e-7,
             posd_tol: float = 1e-8, maxit: int = 100) -> np.ndarray:
    """``Matrix::nearPD(corr = FALSE, keepDiag = FALSE)`` — Higham's algorithm.

    poolr calls this through ``.check.R(..., nearpd = TRUE)`` whenever the
    converted covariance matrix is not positive definite.
    """
    n = A.shape[0]
    X = A.copy()
    D_S = np.zeros_like(X)
    for _ in range(maxit):
        Y = X.copy()
        R = Y - D_S
        vals, vecs = np.linalg.eigh(R)
        keep = vals > eig_tol * vals[-1]
        if not keep.any():                                   # pragma: no cover
            raise np.linalg.LinAlgError("nearPD: matrix seems negative semi-definite")
        X = (vecs[:, keep] * vals[keep]) @ vecs[:, keep].T
        D_S = X - R
        X = (X + X.T) / 2.0
        conv = np.linalg.norm(Y - X, np.inf) / np.linalg.norm(Y, np.inf)
        if conv <= conv_tol:
            break
    vals = np.linalg.eigvalsh(X)
    if vals[0] < posd_tol:
        X = X + np.eye(n) * (posd_tol - vals[0])
    return X


def fisher_generalized(p, R, nearpd: bool = True) -> float:
    """``poolr::fisher(p, R = R, adjust = "generalized")$p`` — Brown's method.

    ``R`` must already be the ``mvnconv``-converted covariance matrix of the
    ``-2 log p`` statistics.  The Fisher statistic ``-2 Σ log p`` is scaled by
    ``c = Var/(2 E)`` and referred to ``chi^2`` with ``f = 2 E^2 / Var`` degrees
    of freedom, where ``E = 2k`` and ``Var = Σ_ij Cov_ij``.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    k = p.size
    if np.any(np.isnan(p)):                                  # pragma: no cover
        raise ValueError("Values in 'p' vector must not contain NAs.")
    if np.any(p < 0) or np.any(p > 1):                       # pragma: no cover
        raise ValueError("Values in 'p' must be between 0 and 1.")

    covs = np.asarray(R, dtype=np.float64)
    if covs.ndim != 2 or covs.shape[0] != k:                 # pragma: no cover
        raise ValueError("R does not match length of p")

    if nearpd:
        w = np.linalg.eigvalsh((covs + covs.T) / 2.0)
        if np.any(w <= 0):
            covs = _near_pd(covs)

    statistic = -2.0 * np.sum(np.log(p))
    expx2 = 2.0 * k
    varx2 = float(np.sum(covs))
    fval = 2.0 * expx2 ** 2 / varx2
    cval = varx2 / (2.0 * expx2)
    return float(chi2.sf(statistic / cval, df=fval))
