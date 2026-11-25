"""E5 ablations, E6 seed stability, and the locked test evaluation (plan v3 §10).

Everything here runs after results/lock.json exists and uses only locked hyper-parameters.
"""
from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from .config import read_json, write_json
from .evaluate import Evaluator
from .fusion import fixed_fusion, rank, rank_positions, rrf
from .model import Context, fit_or_load
from .pipeline import load_scores
from .training import build_pairs

E6_SEEDS = (17, 23, 42, 71, 101)
METRICS = ("ndcg@10", "recall@10", "map@10")


def _lock(cfg):
    path = cfg["paths"]["results_dir"] / "lock.json"
    if not path.exists():
        raise SystemExit("results/lock.json is missing: run `--stages lock` first (test metrics are locked until then)")
    return read_json(path)


def _sel(lock, m):
    return dict(lock["selected_models"][m]["spec"])


# ---------------------------------------------------------------- E5
def e5_variants(lock):
    r1, r2, r3 = _sel(lock, "R1"), _sel(lock, "R2"), _sel(lock, "R3")
    return [
        ("R3 − g feature (anchor kept)", "R3", {**r3, "drop_g_feature": True}),
        ("R3 − anchor term (g feature kept)", "R3", {**r3, "model": "R1", "gamma": 0.0}),
        ("R1 uniform negatives", "R1", {**r1, "pairs": "uniform"}),
        ("R2 uniform negatives", "R2", {**r2, "pairs": "uniform"}),
        ("R3 uniform negatives", "R3", {**r3, "pairs": "uniform"}),
        ("R3 no head skipping (v2 hard set)", "R3", {**r3, "pairs": "noskip"}),
    ]


def stage_e5(cfg):
    lock = _lock(cfg)
    ctx = Context(cfg)
    res = cfg["paths"]["results_dir"]
    rows = []
    for name, base, spec in e5_variants(lock):
        r = fit_or_load(ctx, spec)
        rows.append({"name": name, "base": base, "key": r["key"], "spec": spec, "reused": r["reused"],
                     "dev": r["dev"], "dev_delta_vs_fixed": r["dev_delta_vs_fixed"], "dev_alpha": r["dev_alpha"],
                     "converged": r["converged"], "base_dev_ndcg10": lock["selected_models"][base]["dev_ndcg10"]})
        print(f"[e5] {name:38s} dev NDCG@10={r['dev']['ndcg@10']:.4f} (locked {base}: {rows[-1]['base_dev_ndcg10']:.4f})  "
              f"α mean={r['dev_alpha']['mean']:.3f}  conv={r['converged']}{'  (reused: identical spec)' if r['reused'] else ''}")
    write_json(res / "e5.json", {"variants": rows})
    return {}


# ---------------------------------------------------------------- E6
def stage_e6(cfg):
    lock = _lock(cfg)
    res, cache = cfg["paths"]["results_dir"], cfg["paths"]["cache_dir"]
    pair_stats = {}
    for s in E6_SEEDS:
        if not (cache / f"pairs_main_s{s}.npz").exists():
            pair_stats[s] = build_pairs(cfg, "main", s)
    ctx = Context(cfg)
    rows = []
    for m in ("R1", "R2", "R3"):
        for s in E6_SEEDS:
            r = fit_or_load(ctx, {**_sel(lock, m), "seed": s})
            rows.append({"model": m, "seed": s, "key": r["key"], "dev": r["dev"], "converged": r["converged"],
                         "dev_alpha_mean": r["dev_alpha"]["mean"], "reused": r["reused"]})
        v = [r["dev"]["ndcg@10"] for r in rows if r["model"] == m]
        print(f"[e6] {m}: dev NDCG@10 over seeds {E6_SEEDS} = " + ", ".join(f"{x:.4f}" for x in v)
              + f"  mean={np.mean(v):.4f} sd={np.std(v, ddof=1):.4f}")
    write_json(res / "e6.json", {"seeds": list(E6_SEEDS), "runs": rows, "new_pair_stats": pair_stats})
    return {}


# ---------------------------------------------------------------- test
def paired_bootstrap(x, y, n_boot=10_000, seed=17):
    d = np.asarray(x) - np.asarray(y)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)], "frac_boot_gt0": float(np.mean(means > 0)),
            "n_queries": int(len(d)), "_boot": means}


def stage_test(cfg):
    lock = _lock(cfg)
    res = cfg["paths"]["results_dir"]
    ctx = Context(cfg)
    ds, k = ctx.ds, cfg["eval"]["k"]
    ev = Evaluator(ds.qrels["test"], k)
    rows = ev.rows
    dedup = np.isin(rows, np.array(read_json(res / "decisions.json")["dedup_test_rows"]))
    B, D = load_scores(cfg)
    dec = lock["decisions"]
    a_fix, k_rrf, k_def = dec["e1_alpha_fixed_dev"], dec["e1_rrf_k_dev"], cfg["fusion"]["rrf_k_default"]
    Bz, Dz = ctx.Bz[rows], ctx.Dz[rows]
    pos_b, pos_d = rank_positions(rank(B[rows])), rank_positions(rank(D[rows]))

    def learned(spec, w):
        a = expit(ctx.design_spec(spec)[rows] @ np.asarray(w))
        return a[:, None] * Bz + (1 - a)[:, None] * Dz, a

    systems, alphas = {}, {}
    systems["BM25"] = B[rows]
    systems["Dense"] = D[rows]
    systems["Fixed α=0.50"] = fixed_fusion(Bz, Dz, 0.5)
    systems[f"Fixed α={a_fix:.2f} (dev-best)"] = fixed_fusion(Bz, Dz, a_fix)
    systems[f"RRF k={k_def}"] = rrf(pos_b, pos_d, k_def)
    systems[f"RRF k={k_rrf} (dev-selected)"] = rrf(pos_b, pos_d, k_rrf)
    fixed_name = f"Fixed α={a_fix:.2f} (dev-best)"
    for m, sm in lock["selected_models"].items():
        systems[m], alphas[m] = learned(sm["spec"], sm["w"])

    def run_of(key):
        return read_json(res / "runs" / f"{key}.json")

    extra = {}
    if (res / "e5.json").exists():
        for v in read_json(res / "e5.json")["variants"]:
            r = run_of(v["key"])
            extra[f"E5: {v['name']}"] = learned(r["spec"], r["w"])[0]
    if (res / "e6.json").exists():
        for v in read_json(res / "e6.json")["runs"]:
            r = run_of(v["key"])
            extra[f"E6: {v['model']} seed {v['seed']}"] = learned(r["spec"], r["w"])[0]

    per = {name: ev.per_query_rows(S) for name, S in {**systems, **extra}.items()}
    fixed = per[fixed_name]["ndcg@10"]
    table = []
    for name in systems:
        d = per[name]["ndcg@10"] - fixed
        table.append({"system": name, **{m: float(per[name][m].mean()) for m in METRICS},
                      "dedup_ndcg@10": float(per[name]["ndcg@10"][dedup].mean()),
                      "delta_vs_fixed": float(d.mean()), "improved_frac": float(np.mean(d > 0)),
                      "severe_degradation_frac": float(np.mean(d < -0.05)), "delta_p10": float(np.percentile(d, 10))})

    boots = {}
    for a, b in (("R3", fixed_name), ("R1", fixed_name), ("R2", fixed_name), ("R4", fixed_name), ("R3", "R2"), ("R3", "R1")):
        boots[f"{a} − {b}"] = paired_bootstrap(per[a]["ndcg@10"], per[b]["ndcg@10"], seed=cfg["runtime"]["seed"])
    primary = boots[f"R3 − {fixed_name}"]

    seeds = {}
    for m in ("R1", "R2", "R3"):
        vals = {s: float(per[f"E6: {m} seed {s}"]["ndcg@10"].mean()) for s in E6_SEEDS if f"E6: {m} seed {s}" in per}
        if vals:
            v = np.array(list(vals.values()))
            seeds[m] = {"per_seed": vals, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    e5 = {name[4:]: float(per[name]["ndcg@10"].mean()) for name in per if name.startswith("E5: ")}

    # cases: R3 vs fixed fusion
    d = per["R3"]["ndcg@10"] - fixed
    order = np.argsort(d, kind="stable")
    g = ctx.feats[ctx.g_name][rows]

    def case(i):
        return {"query_id": ds.query_ids[rows[i]], "query": ds.query_texts[rows[i]][:160], "delta": float(d[i]),
                "ndcg_R3": float(per["R3"]["ndcg@10"][i]), "ndcg_fixed": float(fixed[i]),
                "ndcg_BM25": float(per["BM25"]["ndcg@10"][i]), "ndcg_Dense": float(per["Dense"]["ndcg@10"][i]),
                "alpha_R3": float(alphas["R3"][i]), "g": float(g[i]), "n_relevant": int(ev.n_rel[i])}
    cases = {"best": [case(i) for i in order[::-1][:3]], "worst": [case(i) for i in order[:3]]}

    # online cost of the fusion step (features already cached): α, fused scores, ranking
    sm = lock["selected_models"]["R3"]
    X_all = ctx.design_spec(sm["spec"])
    t0 = time.perf_counter()
    for _ in range(5):
        a = expit(X_all[rows] @ np.asarray(sm["w"]))
        rank(a[:, None] * Bz + (1 - a)[:, None] * Dz, k)
    fuse_ms = (time.perf_counter() - t0) / 5 / len(rows) * 1000
    rc = read_json(res / "retrieval_costs.json")
    cost = {"fusion_and_ranking_ms_per_query": fuse_ms,
            "bm25_ms_per_query": rc["bm25_index_and_score_all_queries_s"] / rc["n_queries"] * 1000,
            "dense_query_encoding_and_scoring_ms_per_query": rc["dense_query_encoding_and_scoring_s"] / rc["n_queries"] * 1000,
            "dense_doc_encoding_total_s": rc["dense_doc_encoding_s"],
            "training_time_s": {m: run_of(v["key"])["time_s"] for m, v in lock["selected_models"].items()}}

    out = {"n_test_queries": int(len(rows)), "n_dedup_queries": int(dedup.sum()), "table": table,
           "bootstrap": {k_: {kk: vv for kk, vv in v.items() if kk != "_boot"} for k_, v in boots.items()},
           "primary_comparison": f"R3 − {fixed_name}", "seed_stability": seeds, "e5_test_ndcg10": e5,
           "cases": cases, "cost": cost,
           "alpha_test": {m: {"mean": float(a.mean()), "sd": float(a.std()), "p5": float(np.percentile(a, 5)),
                              "p50": float(np.median(a)), "p95": float(np.percentile(a, 95))} for m, a in alphas.items()}}
    write_json(res / "test_results.json", out)
    np.savez(res / "test_per_query_ndcg10.npz", rows=rows, **{n: v["ndcg@10"] for n, v in per.items()})
    _test_figures(res, d, primary["_boot"], alphas, a_fix)
    _test_report(res, out, fixed_name)
    return {}


def _test_figures(res, d, boot, alphas, a_fix):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    ax[0].hist(d, bins=41, color="#3b6ea5", edgecolor="white")
    ax[0].axvline(-0.05, ls="--", c="#c0392b", lw=1, label="severe (−0.05)")
    ax[0].set_title("Test: per-query ΔNDCG@10 (R3 − fixed)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].hist(boot, bins=60, color="#7f8c8d", edgecolor="white")
    ax[1].axvline(0, c="k", lw=1)
    for q in np.percentile(boot, [2.5, 97.5]):
        ax[1].axvline(q, ls="--", c="#c0392b", lw=1)
    ax[1].set_title("Paired bootstrap of mean Δ (10,000)", fontsize=10)
    ax[2].boxplot([alphas[m] for m in alphas], tick_labels=list(alphas), showfliers=False)
    ax[2].axhline(a_fix, ls="--", c="gray", lw=1, label=f"fixed α={a_fix:.2f}")
    ax[2].set_title("Test: predicted α_q", fontsize=10)
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "figures" / "test_overview.png", dpi=150)
    plt.close(fig)


def _test_report(res, out, fixed_name):
    L = [f"# Test evaluation (locked configuration; {out['n_test_queries']} queries, dedup subset {out['n_dedup_queries']})", "",
         "| system | NDCG@10 | R@10 | MAP@10 | NDCG@10 (dedup) | Δ vs fixed | improved | severe (<−0.05) | Δ p10 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in out["table"]:
        L.append(f"| {r['system']} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | {r['map@10']:.4f} | {r['dedup_ndcg@10']:.4f} | "
                 f"{r['delta_vs_fixed']:+.4f} | {r['improved_frac']:.3f} | {r['severe_degradation_frac']:.3f} | {r['delta_p10']:+.3f} |")
    L += ["", "Paired bootstrap on test NDCG@10 (10,000 resamples of queries; 95% percentile CI):", ""]
    for name, b in out["bootstrap"].items():
        tag = " **(primary, pre-registered)**" if name == out["primary_comparison"] else ""
        L.append(f"- {name}: mean Δ = {b['mean_diff']:+.4f}, 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]{tag}")
    if out["seed_stability"]:
        L += ["", "Weak-negative resampling (E6), test NDCG@10:", ""]
        for m, s in out["seed_stability"].items():
            L.append(f"- {m}: mean {s['mean']:.4f}, sd {s['sd']:.4f}; " + ", ".join(f"s{k}={v:.4f}" for k, v in s["per_seed"].items()))
    if out["e5_test_ndcg10"]:
        L += ["", "E5 variants, test NDCG@10:", ""] + [f"- {k}: {v:.4f}" for k, v in out["e5_test_ndcg10"].items()]
    L += ["", "Cases (R3 vs fixed fusion):", ""]
    for kind in ("best", "worst"):
        for c in out["cases"][kind]:
            L.append(f"- [{kind}] {c['query_id']} Δ={c['delta']:+.3f} (R3 {c['ndcg_R3']:.3f}, fixed {c['ndcg_fixed']:.3f}, "
                     f"BM25 {c['ndcg_BM25']:.3f}, Dense {c['ndcg_Dense']:.3f}); α_R3={c['alpha_R3']:.2f}, g={c['g']:.2f}, "
                     f"|rel|={c['n_relevant']}: “{c['query'][:90]}”")
    c = out["cost"]
    L += ["", f"Cost per query: BM25 {c['bm25_ms_per_query']:.2f} ms, dense query encoding + scoring "
          f"{c['dense_query_encoding_and_scoring_ms_per_query']:.2f} ms, fusion + ranking {c['fusion_and_ranking_ms_per_query']:.3f} ms; "
          f"one-off document encoding {c['dense_doc_encoding_total_s']:.1f} s; training (BFGS) "
          + ", ".join(f"{m} {t:.2f}s" for m, t in c["training_time_s"].items()) + "."]
    (res / "test_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
