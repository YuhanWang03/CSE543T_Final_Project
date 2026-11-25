"""E4: solver comparison and curvature analysis on the locked R1 and R3 objectives (plan v3 §8)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import read_json, write_json
from .model import Context, anchor_start
from .solvers import GTOL, METHODS, run_solver

START_SEEDS = (11, 29)
TIMING_REPEATS = 3
COLORS = {"GD": "#c0392b", "BFGS": "#2471a3", "L-BFGS-B": "#17a589", "Newton-CG": "#8e44ad", "trust-exact": "#d68910"}


def _objective_from_spec(ctx, spec):
    obj, X_all, _, _ = ctx.objective(spec["features"], spec["model"], spec["lam"], spec["gamma"],
                                     spec.get("pairs", "main"), spec.get("seed"), spec.get("drop_g_feature", False))
    return obj, X_all


def stage_e4(cfg):
    res = cfg["paths"]["results_dir"]
    lock = read_json(res / "lock.json")
    ctx = Context(cfg)
    runs, curvature = [], {}
    for target in lock["e4_targets"]:
        spec = lock["selected_models"][target]["spec"]
        obj, X_all = _objective_from_spec(ctx, spec)
        w_anchor = anchor_start(obj.d, ctx.alpha0)
        starts = {"anchor": w_anchor}
        for s in START_SEEDS:
            starts[f"perturbed_s{s}"] = w_anchor + np.random.default_rng(s).normal(0.0, 0.1, obj.d)
        for sname, w0 in starts.items():
            for method in METHODS:
                rec = run_solver(obj, w0, method)
                times = [rec["time_s"]]
                if sname == "anchor":
                    times += [run_solver(obj, w0, method)["time_s"] for _ in range(TIMING_REPEATS - 1)]
                rec.update({"target": target, "start": sname, "times_s": times, "time_median_s": float(np.median(times)),
                            "dev_ndcg10": ctx.dev_eval(X_all, np.array(rec["w"]))["dev"]["ndcg@10"]})
                runs.append(rec)
                print(f"[e4] {target} {sname:13s} {method:11s} conv={rec['converged']!s:5s} J={rec['J']:.10f} "
                      f"|g|={rec['grad_inf']:.1e} nit={rec['nit']:4d} f/g/H={rec['n_f']}/{rec['n_g']}/{rec['n_h']} "
                      f"t={rec['time_median_s']:.3f}s  {rec['message'][:45]}")
        curvature[target] = _curvature(obj, w_anchor, [r for r in runs if r["target"] == target])

    # multi-start agreement among converged runs
    agreement = {}
    for target in lock["e4_targets"]:
        tr = [r for r in runs if r["target"] == target and r["converged"]]
        ref = next(r for r in runs if r["target"] == target and r["start"] == "anchor" and r["method"] == "BFGS")
        Js = np.array([r["J"] for r in tr])
        agreement[target] = {"n_converged": len(tr), "n_runs": sum(r["target"] == target for r in runs),
                             "J_spread": float(Js.max() - Js.min()) if len(Js) else None,
                             "max_w_distance_to_BFGS_anchor": float(max(np.linalg.norm(np.array(r["w"]) - np.array(ref["w"])) for r in tr)) if tr else None,
                             "all_solutions_positive_definite": bool(all(r["hess_eig_min"] > 0 for r in tr)),
                             "dev_ndcg10_range": [float(min(r["dev_ndcg10"] for r in tr)), float(max(r["dev_ndcg10"] for r in tr))] if tr else None}
    out = {"runs": runs, "curvature": curvature, "multi_start": agreement, "kappa_vs_gamma": _kappa_vs_gamma(res)}
    write_json(res / "e4_runs.json", out)
    _figures(res, runs, lock["e4_targets"], out["kappa_vs_gamma"])
    _report(res, runs, curvature, agreement, out["kappa_vs_gamma"], lock)
    return {}


def _curvature(obj, w_anchor, target_runs):
    """GD rate theory vs observation, Hessian at the start, and a random-point nonconvexity probe."""
    ref = next(r for r in target_runs if r["start"] == "anchor" and r["method"] == "BFGS")
    gd = next(r for r in target_runs if r["start"] == "anchor" and r["method"] == "GD")
    lmin, lmax = ref["hess_eig_min"], ref["hess_eig_max"]
    kappa = lmax / lmin
    g0 = float(np.abs(obj.grad(w_anchor)).max())
    # Armijo accepted t = 1 throughout (t < 2/λ_max), so GD behaves as fixed-step GD with t = 1;
    # its local linear rate is max|1 − t λ_i| = 1 − λ_min, while the optimally scaled rate is (κ−1)/(κ+1).
    rate_fixed = max(abs(1 - lmin), abs(1 - lmax))
    rate_opt = (kappa - 1) / (kappa + 1)
    pred_fixed = float(np.log(GTOL / g0) / np.log(rate_fixed))
    pred_opt = float(np.log(GTOL / g0) / np.log(rate_opt))
    ev0 = np.linalg.eigvalsh(obj.hess(w_anchor))
    rng = np.random.default_rng(0)
    probe = {}
    for scale in (0.5, 1.0, 3.0):
        neg = [np.linalg.eigvalsh(obj.hess(w_anchor + rng.normal(0, scale, obj.d))).min() < 0 for _ in range(200)]
        probe[f"N(0,{scale}^2)"] = float(np.mean(neg))
    return {"lambda_min": lmin, "lambda_max": lmax, "kappa": kappa, "grad_inf_at_anchor": g0,
            "gd_rate_fixed_step_1": rate_fixed, "gd_rate_optimal_step": rate_opt,
            "gd_iterations_predicted_fixed_step": pred_fixed, "gd_iterations_predicted_optimal_step": pred_opt,
            "gd_iterations_observed": gd["nit"], "gd_converged": gd["converged"],
            "hessian_at_anchor_eig": [float(x) for x in ev0], "hessian_at_anchor_indefinite": bool(ev0.min() < 0),
            "frac_random_points_with_negative_curvature": probe}


def _kappa_vs_gamma(res):
    e3 = read_json(res / "e3_runs.json")
    rows = []
    for m, recs in e3["runs"].items():
        for r in recs:
            rows.append({"model": m, "lam": r["spec"]["lam"], "gamma": r["spec"]["gamma"], "kappa": r["cond"],
                         "lambda_min": r["hess_eig_min"], "lambda_max": r["hess_eig_max"], "bfgs_nit": r["nit"]})
    return rows


def _figures(res, runs, targets, kg):
    fig, axes = plt.subplots(len(targets), 2, figsize=(11, 3.6 * len(targets)))
    for i, target in enumerate(targets):
        tr = [r for r in runs if r["target"] == target]
        Jstar = min(r["J"] for r in tr)
        for r in tr:
            if r["start"] != "anchor":
                continue
            h = np.array(r["history"])
            gap = np.maximum(h[:, 1] - Jstar, 1e-16)
            axes[i, 0].semilogy(h[:, 0], gap, color=COLORS[r["method"]], label=r["method"])
            axes[i, 1].semilogy(np.maximum(h[:, 3], 1e-4), gap, color=COLORS[r["method"]], label=r["method"])
        axes[i, 0].set_title(f"{target}: J − J* vs iteration (anchor start)", fontsize=10)
        axes[i, 1].set_title(f"{target}: J − J* vs wall time", fontsize=10)
        axes[i, 0].set_xscale("symlog", linthresh=10)
        axes[i, 1].set_xscale("log")
        axes[i, 0].set_xlabel("iteration")
        axes[i, 1].set_xlabel("seconds")
        axes[i, 0].set_ylabel("J − J*")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "figures" / "e4_convergence.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    lams = sorted({r["lam"] for r in kg})
    for ax, m in zip(axes, ("R2", "R3", "R4")):
        for lam in lams:
            rr = sorted([r for r in kg if r["model"] == m and r["lam"] == lam], key=lambda r: r["gamma"])
            ax.loglog([r["gamma"] for r in rr], [r["kappa"] for r in rr], marker="o", label=f"λ={lam:.0e}")
            r1 = next(r for r in kg if r["model"] == "R1" and r["lam"] == lam)
            ax.axhline(r1["kappa"], ls=":", lw=1, color=ax.lines[-1].get_color())
        ax.set_title(f"{m}: κ(∇²J) at the solution", fontsize=10)
        ax.set_xlabel("γ (dotted: R1, γ=0)")
    axes[0].set_ylabel("condition number κ")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "figures" / "e4_kappa_vs_gamma.png", dpi=150)
    plt.close(fig)


def _report(res, runs, curvature, agreement, kg, lock):
    L = ["# E4 solver comparison and curvature", "",
         "Common acceptance: ‖∇J‖∞ ≤ 1e-5 at the returned point; ≤ 2,000 iterations; ≤ 10,000 f/g calls. "
         "Time = median of 3 runs for the anchor start (single run otherwise), history evaluation excluded.", ""]
    for target in lock["e4_targets"]:
        spec = lock["selected_models"][target]["spec"]
        L += [f"## {target} (λ={spec['lam']:g}, γ={spec['gamma']:g})", "",
              "| start | method | converged | J | ‖∇J‖∞ | iters | f / g / H calls | time s | λ_min(∇²J) | κ | dev NDCG@10 | stop reason |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in runs if r["target"] == target]:
            L.append(f"| {r['start']} | {r['method']} | {'yes' if r['converged'] else '**no**'} | {r['J']:.9f} | {r['grad_inf']:.1e} | "
                     f"{r['nit']} | {r['n_f']} / {r['n_g']} / {r['n_h']} | {r['time_median_s']:.3f} | {r['hess_eig_min']:.2e} | "
                     f"{r['cond']:.3g} | {r['dev_ndcg10']:.4f} | {r['message'][:60]} |")
        c, a = curvature[target], agreement[target]
        L += ["", f"- Hessian at the solution: λ_min={c['lambda_min']:.3e}, λ_max={c['lambda_max']:.3e}, κ={c['kappa']:.3g}.",
              f"- GD (Armijo accepted t=1, i.e. fixed step 1): predicted local rate 1−λ_min = {c['gd_rate_fixed_step_1']:.5f} "
              f"→ ≈{c['gd_iterations_predicted_fixed_step']:.0f} iterations to reach the tolerance; observed {c['gd_iterations_observed']} "
              f"(converged={c['gd_converged']}). With an ideally scaled step the rate would be (κ−1)/(κ+1) = {c['gd_rate_optimal_step']:.3f} "
              f"→ ≈{c['gd_iterations_predicted_optimal_step']:.0f} iterations.",
              f"- Hessian at the anchor start indefinite: {c['hessian_at_anchor_indefinite']}; fraction of random points with negative curvature: "
              + ", ".join(f"{k}: {v:.0%}" for k, v in c["frac_random_points_with_negative_curvature"].items()) + ".",
              f"- Multi-start: {a['n_converged']}/{a['n_runs']} runs converged; spread of J among them {a['J_spread']:.2e}; "
              f"max ‖w − w_BFGS,anchor‖ = {a['max_w_distance_to_BFGS_anchor']:.2e}; all positive definite: {a['all_solutions_positive_definite']}.", ""]
    L += ["## κ at the E3 solutions vs γ", "", "| model | λ | γ | λ_min | λ_max | κ | BFGS iterations |", "|---|---|---|---|---|---|---|"]
    for r in kg:
        L.append(f"| {r['model']} | {r['lam']:.0e} | {r['gamma']:g} | {r['lambda_min']:.2e} | {r['lambda_max']:.2e} | {r['kappa']:.3g} | {r['bfgs_nit']} |")
    (res / "e4_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
