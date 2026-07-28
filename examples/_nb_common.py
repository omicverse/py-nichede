"""Shared pieces for ``examples/_build_notebooks.py``.

Keeping the boilerplate here (notebook metadata, the preamble code cell, the
small plotting helpers that every notebook injects) makes the four notebook
builders readable.
"""

from __future__ import annotations

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

KERNEL_META = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def write_notebook(cells, path: str):
    nb = new_notebook(cells=cells, metadata=dict(KERNEL_META))
    nbf.write(nb, path)
    return path


# --------------------------------------------------------------------------- #
# The preamble every notebook starts with
# --------------------------------------------------------------------------- #

PREAMBLE = r'''
%matplotlib inline
import os, sys, json, re, time, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

PKG_ROOT = os.environ.get("PYNICHEDE_ROOT",
                          "/scratch/users/steorra/analysis/omicverse_dev/py-nichede")
REF_DIR  = os.environ.get("NICHEDE_REF_DIR",     "/scratch/users/steorra/nichede_ref_out/full")
DEV_DIR  = os.environ.get("NICHEDE_DEV_REF_DIR", "/scratch/users/steorra/nichede_ref_out/dev300")
PERFUNC  = os.environ.get("NICHEDE_PERFUNC_DIR", "/scratch/users/steorra/nichede_ref_out/perfunc")
RSCRIPT  = os.environ.get("NICHEDE_RSCRIPT",     "/scratch/users/steorra/env/CMAP/bin/Rscript")
R_LIBS   = os.environ.get("NICHEDE_R_LIBS",      "/scratch/users/steorra/Rlibs_nichede")
N_JOBS   = int(os.environ.get("NICHEDE_N_JOBS", "16"))

os.environ.setdefault("TMPDIR", "/scratch/users/steorra/tmp")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.environ["TMPDIR"])
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
for _p in (PKG_ROOT, os.path.join(PKG_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from refload import RefDump                      # tests/refload.py
from parity_report import det, corr, infer       # tests/parity_report.py
import pynichede as nde

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})
C_R, C_PY, C_BAD = "#1f4e79", "#c1440e", "#8b0000"

def sub(a, n=20000, seed=0):
    """Deterministic subsample of a flat array, for readable scatter plots."""
    a = np.asarray(a).ravel()
    if a.size <= n:
        return np.arange(a.size)
    return np.sort(np.random.default_rng(seed).choice(a.size, n, replace=False))

def overlay_det(ref, cand, title, unit="", ax=None):
    """`deterministic` visual: sorted R vs Py overlay + the max abs error."""
    r = np.asarray(ref, dtype=float).ravel()
    c = np.asarray(cand, dtype=float).ravel()
    m = np.isfinite(r) & np.isfinite(c)
    o = np.argsort(r[m], kind="stable")
    err = float(np.max(np.abs(r[m] - c[m]))) if m.any() else 0.0
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.1))
    i = sub(r[m][o], 8000)
    ax[0].plot(np.arange(i.size), r[m][o][i], color=C_R, lw=2.2, label="R (nicheDE)")
    ax[0].plot(np.arange(i.size), c[m][o][i], color=C_PY, lw=0.9, ls="--",
               label="Python (pynichede)")
    ax[0].set_xlabel("entry (sorted by the R value)")
    ax[0].set_ylabel(unit or "value")
    ax[0].set_title(f"{title} — overlay")
    ax[0].legend(loc="upper left")
    d = np.abs(r[m] - c[m])
    d = d[d > 0]
    if d.size:
        ax[1].hist(np.log10(d), bins=50, color=C_PY, alpha=0.85)
        ax[1].set_xlabel("log10 |R - Python|")
    else:
        ax[1].text(0.5, 0.5, "bit-identical\n(all differences exactly 0)",
                   ha="center", va="center", transform=ax[1].transAxes)
        ax[1].set_xticks([])
    ax[1].set_ylabel("count")
    ax[1].set_title(f"max |err| = {err:.3e}   (n = {int(m.sum())})")
    return err

def scatter_corr(ref, cand, title, ax=None, label="T_stat"):
    """`pearson` / ordinal visual: R vs Py scatter with Pearson + Spearman."""
    st = corr(ref, cand)
    r = np.asarray(ref, dtype=float).ravel()
    c = np.asarray(cand, dtype=float).ravel()
    m = np.isfinite(r) & np.isfinite(c)
    i = sub(r[m], 25000)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.scatter(r[m][i], c[m][i], s=3, alpha=0.25, color=C_PY, edgecolors="none")
    lo, hi = float(np.min(r[m])), float(np.max(r[m]))
    ax.plot([lo, hi], [lo, hi], color=C_R, lw=1.0, ls="--", label="y = x")
    ax.set_xlabel(f"R {label}")
    ax.set_ylabel(f"Python {label}")
    ax.set_title(f"{title}\nPearson {st['pearson']:.6f} | Spearman {st['spearman']:.6f}\n"
                 f"n = {st['n']}, max|err| = {st['max_abs_err']:.2e}", fontsize=8)
    ax.legend(loc="upper left", fontsize=7)
    return st

def scatter_pval(ref, cand, title, axes=None, ks=(10, 25, 50, 100, 250, 500)):
    """`inference` visual: -log10(p) scatter + top-K overlap curve."""
    st = infer(ref, cand)
    r = np.clip(np.asarray(ref, dtype=float).ravel(), 1e-300, 1.0)
    c = np.clip(np.asarray(cand, dtype=float).ravel(), 1e-300, 1.0)
    m = np.isfinite(r) & np.isfinite(c)
    r, c = r[m], c[m]
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    i = sub(r, 25000)
    axes[0].scatter(-np.log10(r[i]), -np.log10(c[i]), s=3, alpha=0.25,
                    color=C_PY, edgecolors="none")
    hi = float(np.max(-np.log10(r))) if r.size else 1.0
    axes[0].plot([0, hi], [0, hi], color=C_R, lw=1.0, ls="--")
    axes[0].set_xlabel("R  -log10(p)")
    axes[0].set_ylabel("Python  -log10(p)")
    axes[0].set_title(f"{title}\nSpearman(-log10 p) = {st['spearman_neglog10p']:.6f}"
                      f"  (n = {st['n']})", fontsize=8)
    jac, kk = [], []
    for k in ks:
        k = min(k, r.size)
        a = set(np.argsort(r, kind="stable")[:k].tolist())
        b = set(np.argsort(c, kind="stable")[:k].tolist())
        jac.append(len(a & b) / len(a | b))
        kk.append(k)
    axes[1].plot(kk, jac, "o-", color=C_PY)
    axes[1].axhline(0.70, color=C_BAD, ls="--", lw=1, label="gate 0.70")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("K (most significant genes)")
    axes[1].set_ylabel("top-K Jaccard(R, Python)")
    axes[1].set_title(f"top-50 Jaccard = {st['top50_jaccard']:.3f}", fontsize=8)
    axes[1].legend(fontsize=7)
    return st

print("pynichede", nde.__version__, "| python", sys.version.split()[0],
      "| numpy", np.__version__, "| scipy", __import__("scipy").__version__)
print("REF_DIR    ", REF_DIR)
print("PERFUNC_DIR", PERFUNC)
'''


FIXTURE_LOAD = r'''
d = RefDump(REF_DIR)
cts    = list(d.meta["cell_types"])
cells  = list(d.meta["cell_names"])
genes  = list(d.meta["gene_names"])
sigma  = np.atleast_1d(np.asarray(d.meta["sigma"], dtype=float))

counts = pd.DataFrame(d["ref_counts"],   index=cells, columns=genes)
coord  = pd.DataFrame(d["in_coord"],     index=cells, columns=["imagerow", "imagecol"])
libmat = pd.DataFrame(d["ref_ref_expr"], index=cts,   columns=genes)
deconv = pd.DataFrame(d["in_deconv"],    index=cells, columns=cts)

print(f"counts  {counts.shape}  (spots x genes)")
print(f"coord   {coord.shape}")
print(f"libmat  {libmat.shape}  (cell types x genes)")
print(f"deconv  {deconv.shape}")
print("cell types:", cts)
print("kernel bandwidths (sigma):", sigma)
'''
