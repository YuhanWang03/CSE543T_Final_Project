"""Phase-2 stages: pairs, gradcheck, e2 (feature set), e3 (R1–R4 tuning), lock."""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import read_json, record_decision, write_json
from .fusion import select_with_ties
from .model import Context, anchor_start, fit_or_load
from .objective import Objective, anchor_weights, check_derivatives
from .pairs import VARIANTS, pairs_md5, sample_pairs, validate_pairs
from .pipeline import get_dataset

MODELS = ("R1", "R2", "R3", "R4")


# ---------------------------------------------------------------- pairs
def build_pairs(cfg, variant: str, seed: int, ds=None, quiet=False):
    ds = ds or get_dataset(cfg)
    cache = cfg["paths"]["cache_dir"]
    tr = cfg["train"]
    top_b, top_d = np.load(cache / "top_bm25.npy"), np.load(cache / "top_dense.npy")
    arr, stats = sample_pairs(ds.split_rows["train"], ds.qrels["train"], top_b, top_d, ds.n_docs, variant, seed,
                              tr["pairs_per_query"], tr["hard_per_query"], tr["skip_head"], tr["hard_depth"])
    validate_pairs(arr, ds.qrels["train"])
    np.savez(cache / f"pairs_{variant}_s{seed}.npz", **arr)
    if not quiet:
        print(f"[pairs] {variant} seed={seed}: {stats['n_pairs']} pairs / {stats['n_queries']} queries; "
              f"hard {stats['hard_fraction']:.1%}, fallback {stats['fallback_fraction']:.2%}, "
              f"empty hard sets {stats['queries_with_empty_hard_set']}, dup {stats['duplicate_fraction']:.3%}, "
              f"min pairs/query {stats['pairs_per_query_min']}")
    return stats


def stage_pairs(cfg):
    ds = get_dataset(cfg)
    seed = cfg["runtime"]["seed"]
    stats = {v: build_pairs(cfg, v, seed, ds) for v in cfg["train"]["pair_variants"]}
    write_json(cfg["paths"]["results_dir"] / "pairs_stats.json", stats)
    record_decision(cfg, "pairs_main_md5", stats["main"]["md5"])
    return {"pairs_main_md5": stats["main"]["md5"]}


# ---------------------------------------------------------------- gradient / Hessian checks
def _ok(r):
    g_ok = all((r[f"grad_rel_err_h{s:g}"] <= 1e-5) or (r["grad_inf"] < 1e-6 and r[f"grad_abs_err_h{s:g}"] <= 1e-8)
               for s in (1e-5, 1e-6))
    h_ok = r["hess_rel_err_fro"] <= 1e-4 or r["hess_abs_err_fro"] <= 1e-9
    return g_ok and h_ok and r["hess_asym"] <= 1e-12


def _check_set(label, make_obj, d, rng, alpha0):
    rows = []
    for model in MODELS:
        obj = make_obj(model)
        for wname, w in (("anchor", anchor_start(d, alpha0)), ("random", rng.normal(0, 1, d)),
                         ("extreme", rng.normal(0, 1, d) * 15)):
            r = check_derivatives(obj, w)
            r.update({"set": label, "model": model, "w": wname, "pass": _ok(r)})
            rows.append(r)
    return rows


def stage_gradcheck(cfg):
    rng = np.random.default_rng(cfg["runtime"]["seed"])
    ctx = Context(cfg)
    rows = []

    # synthetic data with uneven pairs per query
    nQ, d = 30, 4
    X = np.column_stack([np.ones(nQ), rng.normal(size=(nQ, d - 1))])
    n_q = rng.integers(1, 20, nQ)
    q = np.repeat(np.arange(nQ), n_q)
    dB, dD = rng.normal(size=len(q)) * 2, rng.normal(size=len(q)) * 2
    g = rng.uniform(0, 1, nQ)
    syn = lambda m: Objective(X, q, dB, dD, 0.3, 1e-2, 0.0 if m == "R1" else 0.5, anchor_weights(m, g)[0])
    rows += _check_set("synthetic", syn, d, rng, 0.3)

    # real subset: first 200 train queries of the main pairs, F5 features
    full, _, _, _ = ctx.objective("F5", "R2", 1e-3, 0.1)
    keep = full.q < 200
    gq = ctx.feats[ctx.g_name][np.unique(ctx.pairs()["q_row"])][:200]
    real = lambda m: Objective(full.X[:200], full.q[keep], full.c[keep] + full.dD[keep], full.dD[keep], ctx.alpha0,
                               1e-3, 0.0 if m == "R1" else 0.1, anchor_weights(m, gq)[0])
    rows += _check_set("real_subset", real, full.d, rng, ctx.alpha0)

    # boundary cases
    one = Objective(X[:1], np.zeros(5, dtype=int), dB[:5], dD[:5], 0.3, 1e-2, 0.5, np.ones(1))
    r = check_derivatives(one, rng.normal(size=d))
    rows.append({**r, "set": "single_query", "model": "R2", "w": "random", "pass": _ok(r)})
    h_const, deg_c = anchor_weights("R3", np.full(nQ, 0.4))
    h_zero, deg_z = anchor_weights("R3", np.zeros(nQ))
    edge = {"R3_constant_g_equals_R2": bool(np.allclose(h_const, 1.0)) and not deg_c,
            "R3_all_zero_g_degenerates_to_R1": bool(deg_z and np.all(h_zero == 0)),
            "R4_all_one_g_degenerates_to_R1": bool(anchor_weights("R4", np.ones(nQ))[1])}
    obj_c = Objective(X, q, dB, dD, 0.3, 1e-2, 0.5, h_const)
    r = check_derivatives(obj_c, rng.normal(size=d))
    rows.append({**r, "set": "constant_g", "model": "R3", "w": "random", "pass": _ok(r)})

    worst = {k: max(r[k] for r in rows) for k in ("grad_rel_err_h1e-05", "grad_rel_err_h1e-06", "hess_rel_err_fro", "hess_asym")}
    out = {"n_checks": len(rows), "all_pass": all(r["pass"] for r in rows) and all(edge.values()),
           "worst": worst, "edge_cases": edge, "checks": rows}
    write_json(cfg["paths"]["results_dir"] / "gradcheck.json", out)
    print(f"[gradcheck] {len(rows)} checks, all pass={out['all_pass']}; worst: " +
          ", ".join(f"{k}={v:.2e}" for k, v in worst.items()) + f"; edge cases: {edge}")
    failed = [(r["set"], r["model"], r["w"]) for r in rows if not r["pass"]]
    if failed or not all(edge.values()):
        raise RuntimeError(f"derivative check failed: {failed} {edge}")
    return {}


# ---------------------------------------------------------------- E2
def _spec(fset, model, lam, gamma, seed, pairs="main", **extra):
    return {"features": fset, "model": model, "lam": float(lam), "gamma": float(gamma if model != "R1" else 0.0),
            "pairs": pairs, "seed": int(seed), "init": "anchor", "method": "BFGS", **extra}


def _fmt_run(r):
    return (f"dev NDCG@10={r['dev']['ndcg@10']:.4f}  Δfixed={r['dev_delta_vs_fixed']['mean']:+.4f}  "
            f"severe={r['dev_delta_vs_fixed']['severe_degradation_frac']:.3f}  |∇J|∞={r['grad_inf']:.1e}  "
            f"conv={r['converged']}  nit={r['nit']}  t={r['time_s']:.2f}s  κ={r['cond']:.3g}")


def stage_e2(cfg):
    ctx = Context(cfg)
    seed, tr = cfg["runtime"]["seed"], cfg["train"]
    best, runs = {}, {}
    for fset in ("F_text", "F5", "F8"):
        recs = [fit_or_load(ctx, _spec(fset, "R1", lam, 0.0, seed)) for lam in tr["lam_grid"]]
        runs[fset] = recs
        for r in recs:
            print(f"[e2] {fset:6s} λ={r['spec']['lam']:.0e}  {_fmt_run(r)}{'  (reused)' if r['reused'] else ''}")
        lam = select_with_ties(tr["lam_grid"], [r["dev"]["ndcg@10"] for r in recs], lambda l: -l)
        best[fset] = next(r for r in recs if r["spec"]["lam"] == lam)
    s5, s8 = best["F5"]["dev"]["ndcg@10"], best["F8"]["dev"]["ndcg@10"]
    chosen = "F8" if s8 >= s5 + tr["feature_margin"] else "F5"
    dec = record_decision(cfg, "feature_set", {"chosen": chosen, "dev_ndcg10_F5": s5, "dev_ndcg10_F8": s8,
                                                "dev_ndcg10_F_text": best["F_text"]["dev"]["ndcg@10"],
                                                "margin": tr["feature_margin"]})
    L = ["| features | d | best λ | dev NDCG@10 | Δ vs fixed | severe degr. | dev α mean (sd) |", "|---|---|---|---|---|---|---|"]
    for fset, r in best.items():
        L.append(f"| {fset} | {len(r['w'])} | {r['spec']['lam']:.0e} | {r['dev']['ndcg@10']:.4f} | "
                 f"{r['dev_delta_vs_fixed']['mean']:+.4f} | {r['dev_delta_vs_fixed']['severe_degradation_frac']:.3f} | "
                 f"{r['dev_alpha']['mean']:.3f} ({r['dev_alpha']['std']:.3f}) |")
    L += ["", f"**Decision:** feature set = `{dec['chosen']}` (F8 needs ≥ F5 + {tr['feature_margin']})"]
    (cfg["paths"]["results_dir"] / "e2_features.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return {}


# ---------------------------------------------------------------- E3
def stage_e3(cfg):
    ctx = Context(cfg)
    res = cfg["paths"]["results_dir"]
    seed, tr = cfg["runtime"]["seed"], cfg["train"]
    fset = read_json(res / "decisions.json")["feature_set"]["chosen"]
    allruns, selected = {}, {}
    for model in MODELS:
        grid = [(lam, 0.0) for lam in tr["lam_grid"]] if model == "R1" else \
               [(lam, gam) for lam in tr["lam_grid"] for gam in tr["gamma_grid"]]
        recs = [fit_or_load(ctx, _spec(fset, model, lam, gam, seed)) for lam, gam in grid]
        allruns[model] = recs
        pick = select_with_ties([(r["spec"]["lam"], r["spec"]["gamma"]) for r in recs],
                                [r["dev"]["ndcg@10"] for r in recs], lambda v: (v[1], -v[0]))
        selected[model] = next(r for r in recs if (r["spec"]["lam"], r["spec"]["gamma"]) == pick)
        print(f"[e3] {model}: selected λ={pick[0]:.0e} γ={pick[1]:g}  {_fmt_run(selected[model])}")
    n_fits = len({r["key"] for recs in allruns.values() for r in recs})
    nonconv = [(m, r["spec"]["lam"], r["spec"]["gamma"]) for m, recs in allruns.items() for r in recs if not r["converged"]]

    write_json(res / "e3_runs.json", {"feature_set": fset, "n_unique_fits": n_fits, "not_converged": nonconv,
                                      "runs": {m: [{k: r[k] for k in ("key", "spec", "dev", "dev_delta_vs_fixed", "dev_alpha",
                                                                      "J", "L_rank", "l2", "anchor", "grad_inf", "converged",
                                                                      "nit", "nfev", "time_s", "message", "hess_eig_min",
                                                                      "hess_eig_max", "cond", "w")} for r in recs]
                                               for m, recs in allruns.items()},
                                      "selected": {m: r["key"] for m, r in selected.items()}})
    L = [f"Feature set: `{fset}`; unique fits: {n_fits}; not converged (‖∇J‖∞ > 1e-5): {len(nonconv)}", "",
         "| model | λ | γ | dev NDCG@10 | R@10 | MAP@10 | Δ vs fixed | improved | severe (<−0.05) | Δ p10 | α mean (sd) | ‖∇J‖∞ | nit | time s | κ(∇²J) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in selected.items():
        dd = r["dev_delta_vs_fixed"]
        L.append(f"| {m} | {r['spec']['lam']:.0e} | {r['spec']['gamma']:g} | {r['dev']['ndcg@10']:.4f} | {r['dev']['recall@10']:.4f} | "
                 f"{r['dev']['map@10']:.4f} | {dd['mean']:+.4f} | {dd['improved_frac']:.3f} | {dd['severe_degradation_frac']:.3f} | "
                 f"{dd['p10']:+.3f} | {r['dev_alpha']['mean']:.3f} ({r['dev_alpha']['std']:.3f}) | {r['grad_inf']:.1e} | "
                 f"{r['nit']} | {r['time_s']:.2f} | {r['cond']:.3g} |")
    fixed = float(ctx.fixed_dev_ndcg.mean())
    L += ["", f"Reference: dev-best fixed fusion α={ctx.dec['e1_alpha_fixed_dev']:.2f}, dev NDCG@10 = {fixed:.4f}", "",
          "All runs (dev NDCG@10):", ""]
    for m, recs in allruns.items():
        for r in recs:
            L.append(f"- {m} λ={r['spec']['lam']:.0e} γ={r['spec']['gamma']:g}: {_fmt_run(r)}")
    (res / "e3_models.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, m in zip(axes, ("R2", "R3", "R4")):
        for lam in tr["lam_grid"]:
            rr = sorted([r for r in allruns[m] if r["spec"]["lam"] == lam], key=lambda r: r["spec"]["gamma"])
            ax.plot([r["spec"]["gamma"] for r in rr], [r["dev"]["ndcg@10"] for r in rr], marker="o", label=f"λ={lam:.0e}")
        r1 = max(allruns["R1"], key=lambda r: r["dev"]["ndcg@10"])["dev"]["ndcg@10"]
        ax.axhline(r1, c="gray", ls=":", lw=1, label="best R1")
        ax.axhline(fixed, c="k", ls="--", lw=1, label="fixed fusion")
        ax.set_xscale("log")
        ax.set_title(m)
        ax.set_xlabel("γ")
    axes[0].set_ylabel("dev NDCG@10")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(res / "figures" / "e3_sensitivity.png", dpi=150)
    plt.close(fig)
    print("\n".join(L[:4 + len(selected)]))
    print(f"Reference fixed fusion dev NDCG@10 = {fixed:.4f}")
    return {}


# ---------------------------------------------------------------- lock
def stage_lock(cfg):
    res = cfg["paths"]["results_dir"]
    path = res / "lock.json"
    if path.exists():
        print(f"[lock] {path.name} already exists — configuration is locked; not overwritten")
        return {}
    dec = read_json(res / "decisions.json")
    e3 = read_json(res / "e3_runs.json")
    runs = {m: read_json(res / "runs" / f"{k}.json") for m, k in e3["selected"].items()}
    lock = {
        "locked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Test metrics may be read only after this file exists. Nothing below may change afterwards.",
        "decisions": {k: v for k, v in dec.items() if k != "dedup_test_rows"},
        "dedup_test_rows_count": len(dec["dedup_test_rows"]),
        "selected_models": {m: {"key": r["key"], "spec": r["spec"], "w": r["w"], "feature_names": r["feature_names"],
                                "dev_ndcg10": r["dev"]["ndcg@10"]} for m, r in runs.items()},
        "test_models": ["BM25", "Dense", "Fixed α=0.5", "Fixed α=dev-best", "RRF k=60", "RRF k=dev-selected",
                        "R1", "R2", "R3", "R4"],
        "primary_comparison": "R3 (main seed) vs dev-best fixed fusion, paired bootstrap on test NDCG@10",
        "e4_targets": ["R1", "R3"],
    }
    write_json(path, lock)
    print(f"[lock] wrote {path}; selected: " + ", ".join(f"{m}(λ={r['spec']['lam']:.0e}, γ={r['spec']['gamma']:g})" for m, r in runs.items()))
    return {}
