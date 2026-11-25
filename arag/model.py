"""Shared training context: feature matrices, objectives, BFGS fitting, dev evaluation, run registry."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from .config import read_json, write_json
from .evaluate import Evaluator
from .features import BASE_FEATURES, EXT_FEATURES
from .fusion import standardize
from .objective import Objective, anchor_weights
from .pipeline import get_dataset, load_scores

FEATURE_SETS = {
    "F_text": ["query_len", "mean_idf"],
    "F5": BASE_FEATURES,                       # g is replaced by the pre-registered divergence
    "F8": BASE_FEATURES + EXT_FEATURES,
}
GTOL = 1e-5        # common acceptance: ||∇J||_inf <= 1e-5
MAXITER = 2000


class Context:
    """Everything a fit needs, loaded once from the phase-1 caches."""

    def __init__(self, cfg):
        self.cfg = cfg
        res, cache = cfg["paths"]["results_dir"], cfg["paths"]["cache_dir"]
        self.ds = get_dataset(cfg)
        B, D = load_scores(cfg)
        eps = cfg["fusion"]["std_eps"]
        self.Bz, _ = standardize(B, eps)
        self.Dz, _ = standardize(D, eps)
        self.feats = dict(np.load(cache / "features.npz"))
        self.top_b = np.load(cache / "top_bm25.npy")
        self.top_d = np.load(cache / "top_dense.npy")
        self.dec = read_json(res / "decisions.json")
        self.g_name = self.dec["divergence"]["chosen"]
        self.alpha0 = float(self.dec["alpha0_train"])
        self.train_rows = np.array(self.ds.split_rows["train"])
        self.ev_dev = Evaluator(self.ds.qrels["dev"], cfg["eval"]["k"])
        A = np.load(cache / "e1_alpha_per_query.npz")
        assert np.array_equal(A["dev_rows"], self.ev_dev.rows)
        ia = int(np.argmin(np.abs(A["grid"] - self.dec["e1_alpha_fixed_dev"])))
        self.fixed_dev_ndcg = A["dev_ndcg_at_10"][ia]
        self._pairs = {}

    # ---------------------------------------------------------------- data pieces
    def feature_names(self, fset: str) -> list[str]:
        return [self.g_name if n == "g" else n for n in FEATURE_SETS[fset]]

    def design(self, fset: str):
        """[n_queries_total, d] design matrix: bias + features standardised with train mean/std."""
        names = self.feature_names(fset)
        raw = np.column_stack([self.feats[n] for n in names]).astype(np.float64)
        mu = raw[self.train_rows].mean(0)
        sd = raw[self.train_rows].std(0)
        zero = sd < 1e-12
        Z = np.where(zero, 0.0, (raw - mu) / np.where(zero, 1.0, sd))
        X = np.column_stack([np.ones(len(raw)), Z])
        return X, {"names": names, "mean": mu, "std": sd, "zero_variance": [n for n, z in zip(names, zero) if z]}

    def design_spec(self, spec: dict):
        """Design matrix for a run spec (handles the E5 'drop g feature' variant)."""
        X_all, finfo = self.design(spec["features"])
        if spec.get("drop_g_feature", False):
            keep = [0] + [i + 1 for i, n in enumerate(finfo["names"]) if n != self.g_name]
            X_all = X_all[:, keep]
        return X_all

    def pairs(self, variant: str = "main", seed: int | None = None):
        seed = self.cfg["runtime"]["seed"] if seed is None else seed
        key = (variant, seed)
        if key not in self._pairs:
            path = self.cfg["paths"]["cache_dir"] / f"pairs_{variant}_s{seed}.npz"
            self._pairs[key] = dict(np.load(path))
        return self._pairs[key]

    def objective(self, fset: str, model: str, lam: float, gamma: float, variant="main", seed=None,
                  drop_g_feature: bool = False):
        pr = self.pairs(variant, seed)
        X_all, finfo = self.design(fset)
        if drop_g_feature:
            keep = [0] + [i + 1 for i, n in enumerate(finfo["names"]) if n != self.g_name]
            X_all = X_all[:, keep]
            finfo = {**finfo, "names": [n for n in finfo["names"] if n != self.g_name]}
        q_rows = np.unique(pr["q_row"])
        q = np.searchsorted(q_rows, pr["q_row"])
        r, p, n = pr["q_row"], pr["pos"], pr["neg"]
        dB = self.Bz[r, p] - self.Bz[r, n]
        dD = self.Dz[r, p] - self.Dz[r, n]
        g = self.feats[self.g_name][q_rows]
        h, degenerate = anchor_weights(model, g)
        obj = Objective(X_all[q_rows], q, dB, dD, self.alpha0, lam, 0.0 if model == "R1" else gamma, h)
        return obj, X_all, finfo, degenerate

    # ---------------------------------------------------------------- evaluation
    def dev_eval(self, X_all, w) -> dict:
        rows = self.ev_dev.rows
        a = expit(X_all[rows] @ w)
        S = a[:, None] * self.Bz[rows] + (1.0 - a)[:, None] * self.Dz[rows]
        m = self.ev_dev.per_query_rows(S)
        delta = m["ndcg@10"] - self.fixed_dev_ndcg
        return {
            "dev": {k: float(v.mean()) for k, v in m.items()},
            "dev_delta_vs_fixed": {"mean": float(delta.mean()), "severe_degradation_frac": float(np.mean(delta < -0.05)),
                                   "p10": float(np.percentile(delta, 10)), "improved_frac": float(np.mean(delta > 0))},
            "dev_alpha": {"mean": float(a.mean()), "std": float(a.std()),
                          **{f"p{p}": float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)}},
            "_per_query_ndcg": m["ndcg@10"],
        }


def anchor_start(d: int, alpha0: float) -> np.ndarray:
    w = np.zeros(d)
    w[0] = logit(np.clip(alpha0, 0.001, 0.999))
    return w


def fit_bfgs(obj: Objective, w0: np.ndarray) -> dict:
    t0 = time.perf_counter()
    r = minimize(obj.value_and_grad, w0, jac=True, method="BFGS",
                 options={"gtol": GTOL, "norm": np.inf, "maxiter": MAXITER})
    elapsed = time.perf_counter() - t0
    g = obj.grad(r.x)
    H = obj.hess(r.x)
    ev = np.linalg.eigvalsh(H)
    return {
        "w": r.x, "time_s": elapsed, "nit": int(r.nit), "nfev": int(r.nfev), "njev": int(getattr(r, "njev", r.nfev)),
        "scipy_success": bool(r.success), "message": str(r.message),
        "grad_inf": float(np.abs(g).max()), "converged": bool(np.abs(g).max() <= GTOL),
        **obj.parts(r.x),
        "hess_eig_min": float(ev.min()), "hess_eig_max": float(ev.max()),
        "cond": float(ev.max() / ev.min()) if ev.min() > 0 else float("inf"),
    }


# ---------------------------------------------------------------- run registry
def run_key(spec: dict) -> str:
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def fit_or_load(ctx: Context, spec: dict) -> dict:
    """Fit once per unique spec; results/runs/<key>.json is reused afterwards (and counts once in the budget)."""
    run_dir = ctx.cfg["paths"]["results_dir"] / "runs"
    run_dir.mkdir(exist_ok=True)
    key = run_key(spec)
    path = run_dir / f"{key}.json"
    if path.exists():
        rec = read_json(path)
        rec["reused"] = True
        return rec
    obj, X_all, finfo, degenerate = ctx.objective(spec["features"], spec["model"], spec["lam"], spec["gamma"],
                                                  spec.get("pairs", "main"), spec.get("seed"),
                                                  spec.get("drop_g_feature", False))
    fit = fit_bfgs(obj, anchor_start(obj.d, ctx.alpha0))
    ev = ctx.dev_eval(X_all, fit["w"])
    rec = {"key": key, "spec": spec, "feature_names": finfo["names"], "zero_variance": finfo["zero_variance"],
           "anchor_degenerated_to_R1": degenerate, "n_queries": obj.nQ, "n_pairs": int(len(obj.q)),
           **{k: v for k, v in fit.items()}, **{k: v for k, v in ev.items() if not k.startswith("_")}}
    rec["w"] = [float(x) for x in fit["w"]]
    write_json(path, rec)
    np.save(run_dir / f"{key}_dev_ndcg.npy", ev["_per_query_ndcg"])
    rec["reused"] = False
    return rec
