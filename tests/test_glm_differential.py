"""Differential test of ``r_glm_fit`` against R's ``glm.fit``, off-fixture.

The parity gate in ``test_exact_match.py`` proves equivalence on **one** input.
This module widens that for the single most load-bearing primitive in the port:
R's IRLS + LINPACK ``dqrdc2`` limited-pivot rank detection, whose behaviour on
rank-deficient designs decides *which interactions Niche-DE reports at all*.

``tests/data_glm_reference.npz`` holds 42 randomly generated Poisson-GLM
problems (n between 30 and 150, p between 2 and 12, offsets, and deliberately
injected collinearity or constant columns) together with R's answer, produced by
``glm.fit(X, y, offset = off, family = poisson())``. 22 of the 42 are
rank-deficient.

The full generated batch was 200 problems, 97 of them rank-deficient, with
**zero** rank or ``NA``-pattern mismatches and a max relative coefficient
deviation of 1.55e-13; this committed subset is the regression guard.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pynichede.rstats import r_glm_fit

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data_glm_reference.npz")


@pytest.fixture(scope="module")
def cases():
    if not os.path.exists(REF):                              # pragma: no cover
        pytest.skip("GLM reference cases not present")
    d = np.load(REF)
    return d, int(d["n_cases"][0])


def test_rank_matches_r_on_every_case(cases):
    """`dqrdc2` must find the same numerical rank R does."""
    d, n = cases
    bad = [k for k in range(n)
           if r_glm_fit(d[f"X{k}"], d[f"y{k}"], offset=d[f"off{k}"],
                        family="poisson").rank != int(d[f"rank{k}"][0])]
    assert not bad, f"rank mismatch on cases {bad}"


def test_na_pattern_matches_r_on_every_case(cases):
    """The *same* columns must be aliased away — not merely the same count.

    This is the property `statsmodels.GLM` does not have: with `pinv` it
    spreads the fit across collinear columns instead of dropping a specific
    one, so Niche-DE's `null` / `new_null` bookkeeping would diverge.
    """
    d, n = cases
    bad = []
    for k in range(n):
        fit = r_glm_fit(d[f"X{k}"], d[f"y{k}"], offset=d[f"off{k}"], family="poisson")
        if not np.array_equal(np.isnan(d[f"coef{k}"]), np.isnan(fit.coefficients)):
            bad.append(k)
    assert not bad, f"NA pattern mismatch on cases {bad}"


def test_coefficients_match_r_to_1e_10(cases):
    d, n = cases
    worst = 0.0
    for k in range(n):
        ref = d[f"coef{k}"]
        fit = r_glm_fit(d[f"X{k}"], d[f"y{k}"], offset=d[f"off{k}"], family="poisson")
        m = ~np.isnan(ref)
        if m.any():
            worst = max(worst, float(np.max(
                np.abs(ref[m] - fit.coefficients[m]) / np.maximum(np.abs(ref[m]), 1e-8))))
    assert worst < 1e-10, f"max relative coefficient deviation {worst:.3e}"


def test_rank_deficient_cases_are_actually_exercised(cases):
    """Guard against the fixture silently losing its rank-deficient cases."""
    d, n = cases
    deficient = sum(1 for k in range(n)
                    if int(d[f"rank{k}"][0]) < d[f"X{k}"].shape[1])
    assert deficient >= 15, f"only {deficient} rank-deficient cases in the fixture"
