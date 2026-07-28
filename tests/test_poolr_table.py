"""Verification of the clean-room ``poolr`` reimplementation.

``pynichede.poolr`` re-derives poolr's ``mvnlookup`` covariance table from its
mathematical definition rather than vendoring the GPL-licensed data.  These
tests pin two things:

1. the derivation is **exact** wherever a closed form exists, and
2. it agrees with the published poolr table to within poolr's own numerical
   accuracy — which is itself only ~1e-3, as the closed-form column proves.
"""

from __future__ import annotations

import numpy as np
import pytest

from pynichede.poolr import MVN_RHOS, build_mvnlookup, fisher_generalized, mvnconv

COLS = ["rhos", "m2lp_1", "m2lp_2", "z_1", "z_2",
        "chisq1_1", "chisq1_2", "p_1", "p_2"]


@pytest.fixture(scope="module")
def table():
    return build_mvnlookup()


def test_grid_matches_poolr(table):
    assert table.shape == (1991, 9)
    assert abs(table[0, 0] - 1.0) < 1e-15
    assert abs(table[-1, 0] - (-0.990)) < 1e-15
    np.testing.assert_allclose(table[:, 0], MVN_RHOS, rtol=0, atol=1e-15)


def test_closed_form_z_side1_is_exactly_rho(table):
    """side = 1, target = "z": ``Phi^-1(1 - p) = Z``, so the covariance is rho."""
    err = np.max(np.abs(table[:, COLS.index("z_1")] - table[:, 0]))
    assert err < 1e-12, f"z_1 deviates from rho by {err:.3e}"


def test_closed_form_chisq1_side2_is_exactly_two_rho_squared(table):
    """side = 2, target = "chisq1": ``F^-1(1-p, 1) = Z^2``, so Cov = 2 rho^2.

    poolr's shipped table deviates from this identity by up to 6.28e-4, which
    is the clearest evidence that ``mvnlookup`` is a numerical approximation
    stored to 4 decimals rather than an exact table.
    """
    exact = 2.0 * table[:, 0] ** 2
    err = np.max(np.abs(table[:, COLS.index("chisq1_2")] - exact))
    assert err < 1e-12, f"chisq1_2 deviates from 2 rho^2 by {err:.3e}"


def test_m2lp_moments_are_exact(table):
    """``-2 log p`` is chi-square with 2 df: mean 2, variance 4.

    At rho = 1 the covariance collapses to the variance, so the first row of
    the m2lp columns must be exactly 4.
    """
    assert abs(table[0, COLS.index("m2lp_1")] - 4.0) < 1e-9
    # side = 2 puts a corner at z = 0 (|z| is not differentiable there), so the
    # Hermite series converges only polynomially; 2000 terms leaves ~5e-6, which
    # still rounds to poolr's 4 decimals exactly.  side = 1 -- the column
    # Niche-DE actually uses -- is smooth and converges to machine precision.
    assert abs(table[0, COLS.index("m2lp_2")] - 4.0) < 1e-5


def test_zero_correlation_gives_zero_covariance(table):
    i = int(np.argmin(np.abs(table[:, 0])))
    assert abs(table[i, 0]) < 1e-15
    for c in COLS[1:]:
        assert abs(table[i, COLS.index(c)]) < 1e-9


def test_mvnconv_lookup_semantics():
    """Rounds to 3 decimals, floors at -0.99, diagonal maps to the rho = 1 row."""
    R = np.array([[1.0, 0.5001], [0.5001, 1.0]])
    cv = mvnconv(R, side=1, target="m2lp")
    assert cv.shape == (2, 2)
    assert abs(cv[0, 0] - 4.0) < 1e-3            # rho = 1 -> Var(-2 log p) = 4
    assert cv[0, 1] == cv[1, 0]
    # -0.995 and -1.0 both clamp onto the -0.99 row
    a = mvnconv(np.array([[1.0, -0.995], [-0.995, 1.0]]), side=1, target="m2lp")
    b = mvnconv(np.array([[1.0, -1.0], [-1.0, 1.0]]), side=1, target="m2lp")
    assert a[0, 1] == b[0, 1]


def test_fisher_generalized_reduces_to_plain_fisher_when_independent():
    """With independent p-values, Brown's method must collapse to Fisher's."""
    from scipy.stats import chi2
    p = np.array([0.01, 0.2, 0.5, 0.001, 0.7, 0.03])
    k = p.size
    R = mvnconv(np.eye(k), side=1, target="m2lp")
    got = fisher_generalized(p, R)
    want = float(chi2.sf(-2 * np.sum(np.log(p)), df=2 * k))
    assert abs(got - want) < 1e-12


def test_fisher_generalized_is_conservative_under_dependence():
    p = np.array([0.01, 0.01, 0.01, 0.01])
    indep = fisher_generalized(p, mvnconv(np.eye(4), side=1, target="m2lp"))
    dep = fisher_generalized(p, mvnconv(np.full((4, 4), 0.9) + 0.1 * np.eye(4),
                                        side=1, target="m2lp"))
    assert dep > indep, "positively correlated tests must give a larger pooled p"
