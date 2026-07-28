"""pytest fixtures for py-nichede."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Where the R reference driver wrote its dump.  Override with NICHEDE_REF_DIR.
REF_DIR = os.environ.get(
    "NICHEDE_REF_DIR",
    os.path.join(REPO, "data", "reference"),
)


def manifest():
    import yaml
    with open(os.path.join(REPO, "data", "manifest.yaml")) as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def gate():
    """The pre-registered parity gate, keyed by output name."""
    m = manifest()
    return {o["name"]: o for o in m["outputs"]}


@pytest.fixture(scope="session")
def ref():
    """The R reference dump; skips the test session if it is not present."""
    from refload import RefDump
    if not os.path.exists(os.path.join(REF_DIR, "meta.json")):
        pytest.skip(
            f"No R reference dump at {REF_DIR}. Generate it with:\n"
            f"  Rscript tests/r_reference_driver.R {REF_DIR} 0 8\n"
            f"or point NICHEDE_REF_DIR at an existing dump."
        )
    return RefDump(REF_DIR)


@pytest.fixture(scope="session")
def cand(ref):
    """The candidate output produced by ``tests/_run_candidate.py``."""
    p = os.path.join(REF_DIR, "candidate.npz")
    if not os.path.exists(p):
        pytest.skip(
            f"No candidate dump at {p}. Generate it with:\n"
            f"  python tests/_run_candidate.py {REF_DIR} 8"
        )
    return np.load(p, allow_pickle=True)


@pytest.fixture(scope="session")
def toy():
    """A tiny synthetic spatial dataset for smoke tests (no R needed)."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n_spot, n_gene, n_ct = 60, 40, 3
    xs, ys = np.meshgrid(np.arange(10), np.arange(6))
    coord = pd.DataFrame({"x": xs.ravel() * 100.0, "y": ys.ravel() * 100.0},
                         index=[f"s{i}" for i in range(n_spot)])
    genes = [f"g{j}" for j in range(n_gene)]
    cts = [f"ct{c}" for c in range(n_ct)]
    lib = pd.DataFrame(rng.gamma(2.0, 1.0, size=(n_ct, n_gene)), index=cts, columns=genes)
    w = rng.dirichlet(np.ones(n_ct), size=n_spot)
    deconv = pd.DataFrame(w, index=coord.index, columns=cts)
    lam = (w @ lib.to_numpy()) * 6.0
    counts = pd.DataFrame(rng.poisson(lam).astype(float), index=coord.index, columns=genes)
    return counts, coord, lib, deconv
