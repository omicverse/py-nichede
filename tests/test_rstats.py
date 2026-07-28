"""Unit tests for the R-faithful primitives in ``pynichede.rstats``.

The expected values are R's, transcribed from ``Rscript`` runs, so these tests
keep working without an R install.
"""

from __future__ import annotations

import numpy as np
import pytest

from pynichede.rstats import (
    brent_fmin,
    dqrdc2,
    nb_lik,
    p_adjust,
    r_glm_fit,
    r_lstsq,
    r_optimize,
    r_quantile,
    weighted_mean,
)


# --------------------------------------------------------------------------- #
# p.adjust — the lazy-`n` NA convention
# --------------------------------------------------------------------------- #

def test_p_adjust_bh_matches_r():
    # R: p.adjust(c(0.4,0.01,0.9,0.02,0.5), "BH")
    got = p_adjust([0.4, 0.01, 0.9, 0.02, 0.5], "BH")
    np.testing.assert_allclose(got, [0.625, 0.05, 0.9, 0.05, 0.625], rtol=0, atol=1e-15)


def test_p_adjust_na_uses_non_na_count():
    """R's lazy default ``n = length(p)`` is forced *after* the NA drop.

    R: p.adjust(c(0.4,NA,0.01,0.9,NA,0.02,0.5), "BH")
       -> 0.625 NA 0.050 0.900 NA 0.050 0.625
    i.e. identical to the NA-free vector.  Using n = 7 would give 0.875/0.07.
    """
    got = p_adjust([0.4, np.nan, 0.01, 0.9, np.nan, 0.02, 0.5], "BH")
    ref = np.array([0.625, np.nan, 0.05, 0.9, np.nan, 0.05, 0.625])
    m = ~np.isnan(ref)
    assert np.isnan(got[~m]).all()
    np.testing.assert_allclose(got[m], ref[m], rtol=0, atol=1e-15)


def test_p_adjust_explicit_n_is_respected():
    got = p_adjust([0.4, 0.01, 0.9, 0.02, 0.5], "BH", n=10)
    # R: p.adjust(c(0.4,0.01,0.9,0.02,0.5), "BH", n = 10)
    np.testing.assert_allclose(got, [1.0, 0.1, 1.0, 0.1, 1.0], rtol=0, atol=1e-15)


def test_p_adjust_bonferroni_and_holm():
    p = [0.01, 0.04, 0.03]
    np.testing.assert_allclose(p_adjust(p, "bonferroni"), [0.03, 0.12, 0.09])
    np.testing.assert_allclose(p_adjust(p, "holm"), [0.03, 0.06, 0.06])


# --------------------------------------------------------------------------- #
# quantile / weighted.mean
# --------------------------------------------------------------------------- #

def test_r_quantile_type7():
    x = np.arange(1, 11, dtype=float)
    # R: quantile(1:10, 0.8) -> 8.2
    assert abs(float(r_quantile(x, 0.8)) - 8.2) < 1e-12


def test_weighted_mean_skips_zero_weights():
    """R: ``sum((x*w)[w != 0]) / sum(w)`` — an Inf paired with weight 0 drops out.

    R: weighted.mean(c(1, Inf, 3), c(0.5, 0, 0.5), na.rm = TRUE) -> 2
    (a naive ``sum(x*w)/sum(w)`` would give NaN from ``Inf * 0``).
    """
    got = weighted_mean([1.0, np.inf, 3.0], [0.5, 0.0, 0.5], na_rm=True)
    assert abs(got - 2.0) < 1e-12


def test_weighted_mean_na_rm_drops_both():
    got = weighted_mean([1.0, np.nan, 3.0], [1.0, 5.0, 1.0], na_rm=True)
    assert abs(got - 2.0) < 1e-12


# --------------------------------------------------------------------------- #
# dqrdc2 / rank-deficient least squares
# --------------------------------------------------------------------------- #

def test_dqrdc2_moves_collinear_columns_right():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=40)
    x3 = rng.normal(size=40)
    X = np.column_stack([np.ones(40), x1, 2 * x1, x3])   # col 2 aliases col 1
    _, _, pivot, rank = dqrdc2(X, tol=1e-7)
    assert rank == 3
    assert pivot[rank] == 2, "the duplicated column must be the one pivoted out"
    assert sorted(pivot[:rank]) == [0, 1, 3]


def test_r_lstsq_zeroes_aliased_slots():
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=30)
    X = np.column_stack([np.ones(30), x1, 2 * x1])
    y = 1.0 + 0.5 * x1 + rng.normal(scale=0.01, size=30)
    coef, pivot, rank = r_lstsq(X, y)
    assert rank == 2
    assert coef[rank:].tolist() == [0.0]
    full = np.zeros(3)
    full[pivot] = coef
    np.testing.assert_allclose(X @ full, np.column_stack([np.ones(30), x1]) @ full[:2],
                               rtol=0, atol=1e-12)


def test_glm_rank_deficient_matches_reduced_model():
    """R returns NA for the aliased coefficient and the *reduced* fit otherwise.

    Transcribed from:
        set.seed(1); n=50; x1=rnorm(n); x2=2*x1; x3=rnorm(n)
        y=rpois(n,exp(0.3*x1+0.2*x3)); off=rep(0.1,n)
        coef(glm(y~cbind(x1,x2,x3)+offset(off), family="poisson"))
        -> -0.3353396  0.4973510  NA  0.3636784
    """
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=60)
    x3 = rng.normal(size=60)
    y = rng.poisson(np.exp(0.3 * x1 + 0.2 * x3)).astype(float)
    off = np.full(60, 0.1)

    full = r_glm_fit(np.column_stack([np.ones(60), x1, 2 * x1, x3]), y,
                     offset=off, family="poisson")
    red = r_glm_fit(np.column_stack([np.ones(60), x1, x3]), y,
                    offset=off, family="poisson")
    assert np.isnan(full.coefficients[2])
    np.testing.assert_allclose(full.coefficients[[0, 1, 3]], red.coefficients,
                               rtol=1e-10, atol=1e-12)
    assert full.rank == 3


def test_glm_poisson_matches_r_on_a_fixed_problem():
    """R:
        y <- c(2,3,6,7,8,9,10,12,15); x <- 1:9
        coef(glm(y ~ x, family = poisson))
        -> (Intercept) 0.981845031703,  x 0.194837678385
    """
    y = np.array([2, 3, 6, 7, 8, 9, 10, 12, 15], dtype=float)
    x = np.arange(1, 10, dtype=float)
    fit = r_glm_fit(np.column_stack([np.ones(9), x]), y, family="poisson")
    np.testing.assert_allclose(fit.coefficients, [0.981845031703, 0.194837678385],
                               rtol=1e-9, atol=1e-11)


def test_gaussian_glm_is_ols():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(50), rng.normal(size=50), rng.normal(size=50)])
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + rng.normal(scale=0.1, size=50)
    fit = r_glm_fit(X, y, family="gaussian")
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(fit.coefficients, ols, rtol=1e-9, atol=1e-11)


# --------------------------------------------------------------------------- #
# optimize / nb_lik
# --------------------------------------------------------------------------- #

def test_brent_fmin_matches_r_optimize():
    """R: optimize(function(x) (x-1/3)^2, lower=0, upper=1)$minimum
       -> 0.3333333 (and $objective ~ 0)
    """
    res = r_optimize(lambda x: (x - 1.0 / 3.0) ** 2, 0.0, 1.0)
    assert abs(res["minimum"] - 1.0 / 3.0) < 1e-4
    assert res["objective"] < 1e-8


def test_brent_fmin_respects_bounds():
    got = brent_fmin(0.05, 100.0, lambda d: (d - 7.5) ** 2, np.finfo(float).eps ** 0.25)
    assert abs(got - 7.5) < 1e-3


def test_nb_lik_matches_r():
    """R: nicheDE::nb_lik(c(1,5,3,0,7), c(2,4,3,1,6), 1.7) -> 11.94706 (approx)."""
    got = nb_lik([1, 5, 3, 0, 7], [2, 4, 3, 1, 6], 1.7)
    # closed-form cross-check with scipy's nbinom
    from scipy.stats import nbinom
    size, mu = 1.7, np.array([2, 4, 3, 1, 6], dtype=float)
    want = -np.sum(nbinom.logpmf(np.array([1, 5, 3, 0, 7]), size, size / (size + mu)))
    assert abs(got - want) < 1e-10


def test_nb_lik_recycles_like_r():
    """R recycles the shorter argument; ``niche_DE_core`` relies on it."""
    x = np.arange(6, dtype=float)
    mu = np.array([1.0, 2.0, 3.0])
    assert abs(nb_lik(x, mu, 2.0) - nb_lik(x, np.tile(mu, 2), 2.0)) < 1e-12
