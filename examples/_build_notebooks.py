"""Regenerate the four mandatory notebooks under ``examples/``.

    python examples/_build_notebooks.py [nb1 nb2 nb3 nb4]

Builds the notebook JSON with ``nbformat`` (so the notebooks are reproducible
from source rather than hand-edited), then execute them with::

    python -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3 examples/<nb>.ipynb

The four deliverables (see ``omicverse-rebuildr/NOTEBOOKS.md``):

======  ==========================================  ==========================
n       file                                        audience
======  ==========================================  ==========================
1       ``compare_R_vs_Python.ipynb``               reviewer / scientist
2       ``tutorial_liver_met_visium.ipynb``         new Python user
3       ``function_by_function_R_parity.ipynb``     R user porting code
4       ``evolution.ipynb``                         process auditor
======  ==========================================  ==========================
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _nb_common import write_notebook   # noqa: E402

TARGETS = {
    "nb1": ("_nb1", "compare_R_vs_Python.ipynb"),
    "nb2": ("_nb2", "tutorial_liver_met_visium.ipynb"),
    "nb3": ("_nb3", "function_by_function_R_parity.ipynb"),
    "nb4": ("_nb4", "evolution.ipynb"),
}


def main(which=None):
    which = which or list(TARGETS)
    for key in which:
        modname, fname = TARGETS[key]
        mod = __import__(modname)
        path = write_notebook(mod.cells(), os.path.join(HERE, fname))
        import nbformat
        nb = nbformat.read(path, as_version=4)
        print(f"{key}: wrote {fname}  ({len(nb.cells)} cells, "
              f"{sum(c.cell_type == 'code' for c in nb.cells)} code)")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
