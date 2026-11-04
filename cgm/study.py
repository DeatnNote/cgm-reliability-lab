"""Run every analysis: results/results.json (about 1 minute).   python -m cgm.study"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

from . import analysis as A
from .sim import simulate

warnings.filterwarnings("ignore")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(seed: int = 0, B: int = 500) -> dict:
    t = time.time()
    d = simulate(seed)
    out = {"data": {"lots": len(d["lots"]), "bench_sensors": len(d["bench"]),
                    "clinical_participants": int(d["clinical"]["participants"].shape[0]),
                    "clinical_reference_pairs": int(len(d["clinical"]["pairs"])),
                    "real_world_sessions": len(d["real_world"]), "overnight_windows": len(d["night"])},
           "accuracy_v1": A.accuracy(d, "cgm_v1", B), "firmware": A.firmware(d, B), "bench_to_body": A.bench_to_body(d),
           "reliability": A.reliability(d), "compression": A.compression(d, seed)}
    out["seconds"] = round(time.time() - t, 1)
    out = _clean(out)
    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("data", "seconds")}, indent=1))
