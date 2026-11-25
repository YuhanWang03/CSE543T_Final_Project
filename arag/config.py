"""Config loading, paths and small shared helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("data_dir", "cache_dir", "results_dir"):
        d = (ROOT / cfg["paths"][k]).resolve()
        d.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = d
    (cfg["paths"]["results_dir"] / "figures").mkdir(exist_ok=True)
    return cfg


def alpha_grid(step: float) -> np.ndarray:
    n = int(round(1.0 / step))
    return np.round(np.arange(n + 1) * step, 10)


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def record_decision(cfg: dict, key: str, value, force: bool = False) -> dict:
    """Append a pre-registered decision to results/decisions.json.

    A decision, once written, is locked: later runs reuse it instead of overwriting,
    unless force=True. This enforces "decide once, before seeing downstream results".
    """
    path = cfg["paths"]["results_dir"] / "decisions.json"
    dec = read_json(path) if path.exists() else {}
    if key in dec and not force:
        new = json.loads(json.dumps(value, default=_default))
        if new != dec[key]:
            print(f"[decisions] WARNING: '{key}' is locked as {dec[key]!r}; "
                  f"recomputed value {new!r} ignored (use --force-decisions to overwrite)")
        return dec[key]
    dec[key] = value
    write_json(path, dec)
    return value
