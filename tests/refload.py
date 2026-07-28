"""Loader for the binary dumps produced by ``tests/r_reference_driver.R``."""

from __future__ import annotations

import json
import os

import numpy as np


class RefDump:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "meta.json")) as fh:
            m = json.load(fh)
        # jsonlite wraps scalars in length-1 lists unless auto_unbox; normalise
        self.meta = {k: (v[0] if isinstance(v, list) and len(v) == 1
                         and k in ("ref_scale", "ref_spot_distance",
                                   "contrast_index", "contrast_niche1",
                                   "contrast_niche2", "time_niche_DE",
                                   "time_effective_niche", "n_cores")
                         else v) for k, v in m.items()}

    def has(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.path, name + ".f64"))

    def __getitem__(self, name: str) -> np.ndarray:
        info = self.meta[name]
        shape = info["shape"]
        if not isinstance(shape, list):
            shape = [shape]
        arr = np.fromfile(os.path.join(self.path, name + ".f64"), dtype="<f8")
        if len(shape) == 1:
            return arr
        return arr.reshape(shape, order="F")

    def csv(self, name: str):
        import pandas as pd
        p = os.path.join(self.path, name + ".csv")
        return pd.read_csv(p) if os.path.exists(p) else None
