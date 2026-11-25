"""E1 retrieval baselines and E0 pre-diagnostics (plan v3 §4, §9).

Test queries are NOT evaluated here: the plan only allows test evaluation after the
configuration is locked. Everything below uses train and dev only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .config import alpha_grid, record_decision, write_json
from .evaluate import Evaluator
from .features import BASE_FEATURES, EXT_FEATURES
from .fusion import fixed_fusion, rank, rank_positions, rrf, select_with_ties, standardize
from .pipeline import get_dataset, load_scores

SPLITS = ("train", "dev")
METRICS = ("ndcg@10", "recall@10", "map@10")


def _prefer_alpha(a):  # closer to 0.5, then smaller alpha
    return (round(abs(a - 0.5), 10), a)


# ---------------------------------------------------------------- E1
def stage_e1(cfg):
    ds = get_dataset(cfg)
    B, D = load_scores(cfg)
    eps, k = cfg["fusion"]["std_eps"], cfg["eval"]["k"]
    Bz, _ = standardize(B, eps)
    Dz, _ = standardize(D, eps)
    grid = alpha_grid(cfg["fusion"]["alpha_step"])
    res, cache = cfg["paths"]["results_dir"], cfg["paths"]["cache_dir"]

    evs = {s: Evaluator(ds.qrels[s], k) for s in SPLITS}
    alpha_pq = {}  # split -> metric -> [n_alpha, n_queries]
    for s, ev in evs.items():
        per = [ev.per_query_rows(fixed_fusion(Bz[ev.rows], Dz[ev.rows], a)) for a in grid]
        alpha_pq[s] = {m: np.stack([p[m] for p in per]) for m in METRICS}
    np.savez(cache / "e1_alpha_per_query.npz", grid=grid,
             **{f"{s}_{m.replace('@', '_at_')}": alpha_pq[s][m] for s in SPLITS for m in METRICS},
             **{f"{s}_rows": evs[s].rows for s in SPLITS})

    alpha_fixed = select_with_ties(grid, alpha_pq["dev"]["ndcg@10"].mean(1), _prefer_alpha)
    alpha0 = select_with_ties(grid, alpha_pq["train"]["ndcg@10"].mean(1), _prefer_alpha)

    pos = {}
    for s, ev in evs.items():
        pos[s] = (rank_positions(rank(B[ev.rows])), rank_positions(rank(D[ev.rows])))
    rrf_dev = {kk: evs["dev"].per_query_rows(rrf(*pos["dev"], kk))["ndcg@10"].mean() for kk in cfg["fusion"]["rrf_k_grid"]}
    k_default = cfg["fusion"]["rrf_k_default"]
    rrf_k = select_with_ties(list(rrf_dev), list(rrf_dev.values()), lambda kk: (kk != k_default, kk))

    alpha_fixed = record_decision(cfg, "e1_alpha_fixed_dev", float(alpha_fixed))
    alpha0 = record_decision(cfg, "alpha0_train", float(alpha0))
    rrf_k = record_decision(cfg, "e1_rrf_k_dev", int(rrf_k))

    systems = {
        "BM25": lambda ev, s: ev.per_query_rows(B[ev.rows]),
        "Dense (E5-small-v2)": lambda ev, s: ev.per_query_rows(D[ev.rows]),
        "Fixed fusion α=0.50": lambda ev, s: ev.per_query_rows(fixed_fusion(Bz[ev.rows], Dz[ev.rows], 0.5)),
        f"Fixed fusion α={alpha_fixed:.2f} (dev-best)": lambda ev, s: ev.per_query_rows(fixed_fusion(Bz[ev.rows], Dz[ev.rows], alpha_fixed)),
        f"RRF k={k_default}": lambda ev, s: ev.per_query_rows(rrf(*pos[s], k_default)),
        f"RRF k={rrf_k} (dev-selected)": lambda ev, s: ev.per_query_rows(rrf(*pos[s], rrf_k)),
    }
    table, per_query = [], {}
    for name, fn in systems.items():
        row = {"system": name}
        for s, ev in evs.items():
            m = fn(ev, s)
            per_query[f"{s}::{name}"] = m["ndcg@10"]
            for met in METRICS:
                row[f"{s}_{met}"] = float(m[met].mean())
        table.append(row)
    np.savez(res / "e1_per_query_ndcg10.npz", **per_query)
    write_json(res / "e1_baselines.json", {"table": table, "alpha_fixed_dev": alpha_fixed, "alpha0_train": alpha0,
                                           "rrf_k_dev": rrf_k, "rrf_dev_ndcg10": rrf_dev,
                                           "alpha_curve": {s: alpha_pq[s]["ndcg@10"].mean(1) for s in SPLITS}})
    lines = ["| System | train NDCG@10 | train R@10 | train MAP@10 | dev NDCG@10 | dev R@10 | dev MAP@10 |",
             "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['system']} | " + " | ".join(f"{r[f'{s}_{m}']:.4f}" for s in SPLITS for m in METRICS) + " |")
    (res / "e1_baselines.md").write_text("\n".join(lines) + "\n\nTest split intentionally not evaluated (locked until configuration freeze).\n", encoding="utf-8")

    plt.figure(figsize=(6, 3.6))
    for s in SPLITS:
        plt.plot(grid, alpha_pq[s]["ndcg@10"].mean(1), marker="o", ms=3, label=s)
    plt.axvline(alpha_fixed, ls="--", c="gray", lw=1, label=f"dev-best α={alpha_fixed:.2f}")
    plt.xlabel("α (weight on BM25)")
    plt.ylabel("mean NDCG@10")
    plt.title("Fixed fusion: NDCG@10 vs α")
    plt.legend()
    plt.tight_layout()
    plt.savefig(res / "figures" / "e1_alpha_curve.png", dpi=150)
    plt.close()
    print("[e1]\n" + "\n".join(lines))
    print(f"[e1] dev-best fixed α={alpha_fixed}, α0(train)={alpha0}, RRF k(dev)={rrf_k}")
    return {}


# ---------------------------------------------------------------- E0
def _oracle(M, grid, alpha_fixed):
    """Per-query best α on the grid; exact ties -> closest to alpha_fixed, then smaller α."""
    best = M.max(0)
    key = np.round(np.abs(grid - alpha_fixed), 10)[:, None] * 100 + grid[:, None]  # distance dominates, α breaks ties
    key = np.where(np.isclose(M, best[None, :], rtol=0, atol=1e-12), key, np.inf)
    return best, grid[key.argmin(0)]


def _dist_stats(x):
    vals, counts = np.unique(np.round(x, 9), return_counts=True)
    q = np.percentile(x, [0, 10, 25, 50, 75, 90, 100])
    return {"percentiles": dict(zip(["p0", "p10", "p25", "p50", "p75", "p90", "p100"], q.tolist())),
            "iqr": float(q[4] - q[2]), "mode_value": float(vals[counts.argmax()]),
            "mode_share": float(counts.max() / len(x)), "n_distinct": int(len(vals)), "mean": float(x.mean())}


def stage_e0(cfg):
    ds = get_dataset(cfg)
    res, cache = cfg["paths"]["results_dir"], cfg["paths"]["cache_dir"]
    A = np.load(cache / "e1_alpha_per_query.npz")
    grid = A["grid"]
    from .config import read_json
    dec = read_json(res / "decisions.json")
    alpha_fixed = dec["e1_alpha_fixed_dev"]
    ia = int(np.argmin(np.abs(grid - alpha_fixed)))
    feats = dict(np.load(cache / "features.npz"))
    e0cfg, dv = cfg["e0"], cfg["divergence"]
    out = {"alpha_fixed_dev": alpha_fixed}

    # ---- 4.1 oracle per-query weights
    oracle_alpha = {}
    for s in SPLITS:
        M = A[f"{s}_ndcg_at_10"]
        best, a_star = _oracle(M, grid, alpha_fixed)
        oracle_alpha[s] = a_star
        out[f"oracle_{s}"] = {
            "mean_ndcg10_oracle": float(best.mean()), "mean_ndcg10_fixed": float(M[ia].mean()),
            "gap": float(best.mean() - M[ia].mean()),
            "frac_alpha_far_from_fixed": float(np.mean(np.abs(a_star - alpha_fixed) > e0cfg["alpha_diff"] + 1e-9)),
            "frac_queries_any_gain": float(np.mean(best > M[ia] + 1e-12)),
            "frac_queries_all_alpha_equal": float(np.mean(np.isclose(M.max(0), M.min(0)))),
        }
    gap = out["oracle_dev"]["gap"]
    if gap >= e0cfg["gap_high"]:
        band = "high: adaptive fusion is the main line (E3 retrieval comparison is the core result)"
    elif gap >= e0cfg["gap_low"]:
        band = "medium: adaptive fusion stays the main line; emphasise limited headroom, degradation and stability"
    else:
        band = "low: switch report main line to optimisation analysis + negative result; still run all core experiments"
    out["e0_band"] = record_decision(cfg, "e0_band", {"gap_dev": gap, "band": band})

    train_rows = A["train_rows"]
    spear = {}
    for name in BASE_FEATURES[:-1] + EXT_FEATURES + ["g_jaccard", "g_rbo"]:
        x = feats[name][train_rows]
        rho = spearmanr(x, oracle_alpha["train"]).statistic if np.ptp(x) > 0 else float("nan")
        spear[name] = float(rho)
    out["spearman_feature_vs_oracle_alpha_train"] = spear

    # ---- 4.2 divergence distribution (label-free) and pre-registered switch
    gstats = {}
    for gname in ("g_jaccard", "g_rbo"):
        gstats[gname] = {s: _dist_stats(feats[gname][A[f"{s}_rows"]]) for s in SPLITS}
    tj = gstats["g_jaccard"]["train"]
    trig_mode = tj["mode_share"] > dv["switch_mode_share"]
    trig_iqr = tj["iqr"] < dv["switch_iqr"]
    choice = "g_rbo" if (trig_mode or trig_iqr) else "g_jaccard"
    out["divergence_stats"] = gstats
    out["divergence_decision"] = record_decision(cfg, "divergence", {
        "chosen": choice, "train_mode_share": tj["mode_share"], "train_iqr": tj["iqr"],
        "triggered_by_mode_share": bool(trig_mode), "triggered_by_iqr": bool(trig_iqr)})
    chosen = out["divergence_decision"]["chosen"]

    # ---- 4.3 descriptive: divergence vs |NDCG_BM25 - NDCG_Dense|
    desc = {}
    for s in SPLITS:
        M = A[f"{s}_ndcg_at_10"]
        absdiff = np.abs(M[-1] - M[0])  # α=1 is the BM25 ranking, α=0 the Dense ranking
        g = feats[chosen][A[f"{s}_rows"]]
        qs = np.quantile(g, [0.25, 0.5, 0.75])
        bins = np.digitize(g, qs)
        desc[s] = {"spearman": float(spearmanr(g, absdiff).statistic),
                   "mean_absdiff_by_g_quartile": [float(absdiff[bins == i].mean()) if np.any(bins == i) else None for i in range(4)],
                   "n_by_quartile": [int(np.sum(bins == i)) for i in range(4)]}
    out["divergence_vs_retriever_disagreement"] = desc
    write_json(res / "e0_summary.json", out)

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, gname, title in ((ax[0], "g_jaccard", "g_J (Top-10 Jaccard distance)"), (ax[1], "g_rbo", "g_RBO (1 − RBO_ext@20, p=0.9)")):
        a.hist(feats[gname][train_rows], bins=22 if gname == "g_jaccard" else 30, color="#3b6ea5", edgecolor="white")
        a.set_title(title, fontsize=10)
        a.set_xlabel("value (train queries)")
    ax[0].set_ylabel("queries")
    fig.tight_layout()
    fig.savefig(res / "figures" / "e0_divergence_hist.png", dpi=150)
    plt.close(fig)
    plt.figure(figsize=(5, 3.2))
    plt.hist(oracle_alpha["train"], bins=np.arange(-0.025, 1.05, 0.05), color="#3b6ea5", edgecolor="white")
    plt.axvline(alpha_fixed, ls="--", c="gray")
    plt.xlabel("per-query oracle α (train)")
    plt.ylabel("queries")
    plt.tight_layout()
    plt.savefig(res / "figures" / "e0_oracle_alpha_hist.png", dpi=150)
    plt.close()

    # ---- readable report
    L = ["# E0 pre-diagnostics", "", f"Dev-best fixed α = {alpha_fixed:.2f}", "",
         "| split | oracle NDCG@10 | fixed NDCG@10 | Gap | frac |α*−α_fixed|>0.2 | frac queries with any gain | frac flat (all α equal) |",
         "|---|---|---|---|---|---|---|"]
    for s in SPLITS:
        o = out[f"oracle_{s}"]
        L.append(f"| {s} | {o['mean_ndcg10_oracle']:.4f} | {o['mean_ndcg10_fixed']:.4f} | {o['gap']:.4f} | "
                 f"{o['frac_alpha_far_from_fixed']:.3f} | {o['frac_queries_any_gain']:.3f} | {o['frac_queries_all_alpha_equal']:.3f} |")
    L += ["", f"**Decision (§4.1):** {out['e0_band']['band']}", "", "Spearman(feature, oracle α) on train:", ""]
    L += [f"- {k}: {v:+.3f}" for k, v in spear.items()]
    L += ["", "Divergence distribution (train):", ""]
    for gname in ("g_jaccard", "g_rbo"):
        st = gstats[gname]["train"]
        L.append(f"- {gname}: IQR={st['iqr']:.3f}, mode={st['mode_value']:.3f} (share {st['mode_share']:.1%}), "
                 f"distinct={st['n_distinct']}, median={st['percentiles']['p50']:.3f}")
    d = out["divergence_decision"]
    L += ["", f"**Decision (§4.2):** use `{d['chosen']}` (mode-share trigger={d['triggered_by_mode_share']}, IQR trigger={d['triggered_by_iqr']})", "",
          f"Divergence `{chosen}` vs |NDCG_BM25 − NDCG_Dense| (§4.3):", ""]
    for s in SPLITS:
        L.append(f"- {s}: Spearman={desc[s]['spearman']:+.3f}; mean |Δ| by g quartile = "
                 + ", ".join(f"{v:.3f}" for v in desc[s]["mean_absdiff_by_g_quartile"] if v is not None))
    (res / "e0_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return {}
