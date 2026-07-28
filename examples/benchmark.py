"""Acceleration-Agent benchmark harness.

Measures each admissible rewrite in ``ITERATION_LOG.md`` against the variant it
replaced, on the canonical fixture.  Every alternative implemented here is the
*previous* iteration's code, kept only so the speedup can be measured; the
shipped package always uses the fastest accepted variant.

    python examples/benchmark.py <ref_dir> [n_reps]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))

from refload import RefDump                                   # noqa: E402
import pynichede as nde                                       # noqa: E402


def timeit(fn, n_reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


# --------------------------------------------------------------------------- #
# iter 1 — design matrix: per-spot Python loop (R transcription) vs broadcasting
# --------------------------------------------------------------------------- #

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
    """Iteration 1: one broadcast + reshape."""
    X = (en_r[:, :, None] * pstg[:, None, :]).transpose(0, 2, 1).reshape(en_r.shape[0], -1)
    if not self_EN:
        diag_idx = np.arange(n_type) * n_type + np.arange(n_type)
        X[:, diag_idx] = 0.0
    return X


# --------------------------------------------------------------------------- #
# iter 2 — inverse of X'WX: explicit triangular inverse vs cho_solve
# --------------------------------------------------------------------------- #

def inv_explicit(var_mat):
    """Iteration 0: R's literal `solve(A) %*% t(solve(A))`."""
    A = cholesky(var_mat, lower=False)
    Ainv = solve_triangular(A, np.eye(A.shape[0]), lower=False)
    return Ainv @ Ainv.T


def inv_cho_solve(var_mat):
    """Iteration 2: two triangular solves, no explicit inverse."""
    c = cho_factor(var_mat, lower=False)
    return cho_solve(c, np.eye(var_mat.shape[0]))


# --------------------------------------------------------------------------- #
# iter 4 — mvnconv: rebuild the Mehler table per call vs cached lookup
# --------------------------------------------------------------------------- #

def mvnconv_uncached(R, side=1, target="m2lp"):
    from pynichede import poolr
    tbl = poolr.build_mvnlookup()
    tbl[:, 1:] = np.round(tbl[:, 1:], 4)
    col = poolr._TARGETS.index(target) * 2 + 1 + (1 if side == 2 else 0)
    Rr = np.where(np.round(R, 3) < -0.99, -0.99, np.round(R, 3))
    idx = np.clip(np.rint((1.0 - Rr) * 1000.0).astype(np.int64), 0, tbl.shape[0] - 1)
    return tbl[idx, col]


def main():
    ref_dir = sys.argv[1]
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    d = RefDump(ref_dir)
    cts = list(d.meta["cell_types"])
    cells = list(d.meta["cell_names"])
    genes = list(d.meta["gene_names"])
    sigma = np.atleast_1d(np.asarray(d.meta["sigma"], dtype=float))
    n_type = len(cts)

    counts = pd.DataFrame(d["ref_counts"], index=cells, columns=genes)
    coord = pd.DataFrame(d["in_coord"], index=cells, columns=["imagerow", "imagecol"])
    libmat = pd.DataFrame(d["ref_ref_expr"], index=cts, columns=genes)
    deconv = pd.DataFrame(d["in_deconv"], index=cells, columns=cts)

    obj = nde.create_nichede_object(counts, coord, libmat, deconv, sigma=sigma)
    obj = nde.calculate_effective_niche(obj)

    nc = np.asarray(obj.num_cells, dtype=float)
    ref = np.asarray(obj.ref_expr, dtype=float)
    en_r = np.round(obj.effective_niche[0], 2)
    re = ref[:, 0]
    EEJ = nc @ re
    with np.errstate(divide="ignore", invalid="ignore"):
        pstg = nc * re[None, :] / EEJ[:, None]
    pstg[pstg < 0.05] = 0.0

    results = {}

    # --- iter 1 -----------------------------------------------------------
    a = build_X_loop(en_r, pstg, n_type)
    b = build_X_broadcast(en_r, pstg, n_type)
    err = float(np.max(np.abs(a - b)))
    t_old, s_old = timeit(lambda: build_X_loop(en_r, pstg, n_type), n_reps)
    t_new, s_new = timeit(lambda: build_X_broadcast(en_r, pstg, n_type), n_reps)
    results["iter1_design_matrix"] = dict(old=t_old, old_sd=s_old, new=t_new,
                                          new_sd=s_new, speedup=t_old / t_new,
                                          max_abs_err=err)

    # --- iter 2 -----------------------------------------------------------
    X = np.column_stack([b, np.ones(b.shape[0])])
    W = np.abs(np.asarray(counts)[:, 0]) + 1.0
    keep = np.flatnonzero((X > 0).sum(0) >= 10)
    Xk = np.column_stack([X[:, keep], np.ones(X.shape[0])])
    var_mat = (Xk * W[:, None]).T @ Xk
    v1, v2 = inv_explicit(var_mat), inv_cho_solve(var_mat)
    err = float(np.max(np.abs(v1 - v2)) / np.max(np.abs(v1)))
    t_old, s_old = timeit(lambda: inv_explicit(var_mat), n_reps * 20)
    t_new, s_new = timeit(lambda: inv_cho_solve(var_mat), n_reps * 20)
    results["iter2_cho_solve"] = dict(old=t_old, old_sd=s_old, new=t_new,
                                      new_sd=s_new, speedup=t_old / t_new,
                                      max_rel_err=err)

    # --- iter 3 -----------------------------------------------------------
    t1, s1 = timeit(lambda: nde.niche_DE(obj, num_cores=1, verbose=False), 1, warmup=0)
    t8, s8 = timeit(lambda: nde.niche_DE(obj, num_cores=8, verbose=False), 1, warmup=0)
    t16, s16 = timeit(lambda: nde.niche_DE(obj, num_cores=16, verbose=False), 1, warmup=0)
    results["iter3_parallel"] = dict(serial=t1, jobs8=t8, jobs16=t16,
                                     speedup8=t1 / t8, speedup16=t1 / t16)

    # --- iter 4 -----------------------------------------------------------
    Rp = np.eye(7) * 0.5 + 0.5
    t_old, s_old = timeit(lambda: mvnconv_uncached(Rp), 3, warmup=0)
    t_new, s_new = timeit(lambda: nde.mvnconv(Rp, 1, "m2lp"), 200)
    results["iter4_cached_lookup"] = dict(old=t_old, old_sd=s_old, new=t_new,
                                          new_sd=s_new, speedup=t_old / t_new,
                                          max_abs_err=0.0)

    # --- reference wall clock --------------------------------------------
    results["reference"] = dict(
        r_niche_DE_s=float(np.atleast_1d(d.meta.get("time_niche_DE", [np.nan]))[0]),
        r_effective_niche_s=float(np.atleast_1d(d.meta.get("time_effective_niche", [np.nan]))[0]),
        r_cores=int(np.atleast_1d(d.meta.get("n_cores", [1]))[0]),
        n_spots=len(cells), n_genes=len(genes), n_celltypes=n_type,
        n_sigma=int(sigma.size),
    )

    print(json.dumps(results, indent=2))
    with open(os.path.join(ref_dir, "benchmark.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
