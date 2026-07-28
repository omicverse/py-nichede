"""Pipeline-level ablation: swap ONE accepted rewrite back to its predecessor.

Each run measures the *whole* `niche_DE` call, so the numbers in
`ITERATION_LOG.md::wall_clock_mean_s` are comparable across iterations.

    python examples/ablation.py <ref_dir> [n_jobs]
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
from refload import RefDump
import pynichede as nde
from pynichede import niche_de as ND, poolr as PL
from scipy.linalg import cholesky, solve_triangular
warnings.simplefilter("ignore")

ref_dir = sys.argv[1]; n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
d = RefDump(ref_dir)
cts = list(d.meta["cell_types"]); cells = list(d.meta["cell_names"])
genes = list(d.meta["gene_names"]); sigma = np.atleast_1d(np.asarray(d.meta["sigma"], float))
counts = pd.DataFrame(d["ref_counts"], index=cells, columns=genes)
coord = pd.DataFrame(d["in_coord"], index=cells, columns=["a", "b"])
lib = pd.DataFrame(d["ref_ref_expr"], index=cts, columns=genes)
dec = pd.DataFrame(d["in_deconv"], index=cells, columns=cts)
obj = nde.calculate_effective_niche(nde.create_nichede_object(counts, coord, lib, dec, sigma=sigma))

def run(jobs=1, reps=3, warmup=1):
    """Warmup-excluded mean of `reps` runs (the first call also builds the
    1.6 s mvnlookup table, so a warmup is mandatory for comparable numbers)."""
    for _ in range(warmup):
        o = nde.niche_DE(obj, num_cores=jobs, verbose=False)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); o = nde.niche_DE(obj, num_cores=jobs, verbose=False)
        ts.append(time.perf_counter() - t)
    return (float(np.mean(ts)), float(np.std(ts)), ts), o

def tstat(o):
    return np.array([r["T_stat"] if r["T_stat"] is not None else np.full((len(cts),) * 2, np.nan)
                     for r in o.niche_DE[0]])

res = {}
t, base = run(1); res["current_serial"] = t
t2, _ = run(n_jobs); res["current_parallel"] = t2
ref_T = tstat(base)

def parity(o):
    a, b = ref_T.ravel(), tstat(o).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m])))

# --- A0: per-gene joblib dispatch instead of chunked -----------------------
_orig_chunk = ND._run_sigma
def per_gene_sigma(counts_m, en, nc_, ctf_, C, M, Int, batch, bid, refx, nt, sen, jobs, verbose):
    from joblib import Parallel, delayed
    ng = counts_m.shape[1]
    def one(j):
        return ND._niche_de_core(counts_m[:, j], j, en, nc_, ctf_, C, M, Int,
                                 batch, bid, refx, nt, sen)
    if jobs in (0, 1, None):
        return [one(j) for j in range(ng)]
    return Parallel(n_jobs=jobs, batch_size=64, verbose=0)(delayed(one)(j) for j in range(ng))
ND._run_sigma = per_gene_sigma
t, o = run(n_jobs); res["ablate_per_gene_dispatch_parallel"] = t
res["ablate_per_gene_dispatch_err"] = parity(o)
ND._run_sigma = _orig_chunk

# --- A: no BLAS pinning ----------------------------------------------------
orig = ND._single_threaded_blas
import contextlib
ND._single_threaded_blas = lambda: contextlib.nullcontext()
t, o = run(1); res["ablate_blas_pinning_serial"] = t; res["ablate_blas_pinning_err"] = parity(o)
ND._single_threaded_blas = orig

# --- B: explicit triangular inverse instead of cho_solve -------------------
src = ND._niche_de_core.__code__
import types
def inv_explicit(a, b):                       # signature-compatible with cho_solve
    A = a[0]; Ainv = solve_triangular(A, np.eye(A.shape[0]), lower=False)
    return Ainv @ Ainv.T
orig_cs = ND.cho_solve; ND.cho_solve = inv_explicit
t, o = run(1); res["ablate_cho_solve_serial"] = t; res["ablate_cho_solve_err"] = parity(o)
ND.cho_solve = orig_cs

# --- C: per-spot Python loop for the design matrix -------------------------
orig_core = ND._niche_de_core
def loop_core(counts_v, iter_, en, nc, ctf, C, M, Int, batch, bid, refx, nt, sen):
    class _P:                                  # force the loop path via a proxy
        pass
    return orig_core(counts_v, iter_, en, nc, ctf, C, M, Int, batch, bid, refx, nt, sen)
# the loop variant is measured directly on the design step for every runnable gene
tot = np.asarray(counts).sum(0)
ctf = np.array([np.quantile(np.asarray(lib)[i], 0.8) for i in range(len(cts))])
passes = (np.asarray(lib) < ctf[:, None]).mean(0) != 1
runnable = np.flatnonzero((tot > 150) & passes)
nc = np.asarray(dec) * 0 + np.asarray(obj.num_cells)
def design_loop_total():
    t0 = time.perf_counter()
    for g in runnable[:200]:
        re = np.asarray(lib)[:, g]; EEJ = nc @ re
        with np.errstate(divide="ignore", invalid="ignore"):
            p = nc * re[None, :] / EEJ[:, None]
        p[p < 0.05] = 0
        e = np.round(obj.effective_niche[0], 2)
        X = np.empty((e.shape[0], len(cts) ** 2))
        for k in range(e.shape[0]):
            X[k] = np.outer(e[k], p[k]).ravel(order="F")
    return time.perf_counter() - t0
def design_vec_total():
    t0 = time.perf_counter()
    for g in runnable[:200]:
        re = np.asarray(lib)[:, g]; EEJ = nc @ re
        with np.errstate(divide="ignore", invalid="ignore"):
            p = nc * re[None, :] / EEJ[:, None]
        p[p < 0.05] = 0
        e = np.round(obj.effective_niche[0], 2)
        X = (e[:, :, None] * p[:, None, :]).transpose(0, 2, 1).reshape(e.shape[0], -1)
    return time.perf_counter() - t0
res["design_loop_200genes"] = design_loop_total()
res["design_vec_200genes"] = design_vec_total()

# --- D: uncached mvnconv --------------------------------------------------
t0 = time.perf_counter(); PL.build_mvnlookup(); res["mvnlookup_build_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(200): PL.mvnconv(np.eye(7) * 0.5 + 0.5, 1, "m2lp")
res["mvnconv_cached_per_call_s"] = (time.perf_counter() - t0) / 200

res["n_genes"] = len(genes); res["n_runnable"] = int(runnable.size)
res["r_niche_DE_s"] = float(np.atleast_1d(d.meta.get("time_niche_DE", [np.nan]))[0])
res["r_cores"] = int(np.atleast_1d(d.meta.get("n_cores", [1]))[0])
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(ref_dir, "ablation.json"), "w"), indent=2)

# --- E: cho_solve AND BLAS pinning ablated together ------------------------
# These two rewrites attack the same bottleneck (a small, BLAS-threaded
# triangular inverse), so their gains are NOT additive.  Measuring the
# combination is the only honest way to attribute them.
ND._single_threaded_blas = lambda: contextlib.nullcontext()
ND.cho_solve = inv_explicit
t, o = run(1)
res["ablate_both_inverse_and_blas_serial"] = t
res["ablate_both_err"] = parity(o)
ND._single_threaded_blas = orig
ND.cho_solve = orig_cs
print(json.dumps({k: v for k, v in res.items() if "both" in k}, indent=2))
json.dump(res, open(os.path.join(ref_dir, "ablation.json"), "w"), indent=2)
