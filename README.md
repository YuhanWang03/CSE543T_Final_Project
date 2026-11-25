# Nonlinear Optimization for Adaptive Hybrid Retrieval (CSE543T)

Query-adaptive fusion of BM25 and E5 dense retrieval on BEIR NFCorpus.

**Status: all experiments done.** Configuration locked on 2026-11-19 (`results/lock.json`) before any
test metric was read; E4 solvers and curvature, E5 ablations, E6 seed stability and the test evaluation
followed. 62 unique fits in total. **Start with `results/SUMMARY.md`.** The written report and slides are not
included in this repository.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

(On the original machine the venv was created with `--system-site-packages` to reuse an existing
CUDA torch; `results/run_info.json` records the versions actually imported.)

## Run

```bash
.venv/Scripts/python.exe run_all.py --stages phase1
.venv/Scripts/python.exe run_all.py --stages phase2
.venv/Scripts/python.exe run_all.py --stages lock      # only after reviewing E3
.venv/Scripts/python.exe run_all.py --stages phase3    # e4, e5, e6, test (requires lock.json)
```

Timing-sensitive stage `e4` takes about 20 minutes (gradient descent runs up to 2,000 iterations);
run nothing else on the machine meanwhile. Everything else finishes in a few minutes.

Stages run in order and each reads the previous stage's cache; any subset can be re-run, e.g. `--stages e1,e0`.

| Stage | Does | Main outputs |
|---|---|---|
| `prepare` | download `nfcorpus.zip`, check MD5, verify counts and labels, duplicate-text check, build the shared 512-token visible text | `results/data_check.json`, `results/text_stats.json`, `cache/visible_texts.jsonl` |
| `retrieve` | BM25Okapi (k1=1.5, b=0.75, ε=0.25) and E5-small-v2 scores against all 3,633 docs | `cache/scores_bm25.npy`, `cache/scores_dense.npy`, `results/retrieval_costs.json` |
| `features` | z-scored scores, top-100 lists, F5+F8 features, g_J and g_RBO | `cache/features.npz`, `cache/top_*.npy` |
| `e1` | BM25, Dense, fixed fusion (α grid), RRF; dev-selected α and k; α0 on train | `results/e1_baselines.md`, `results/figures/e1_alpha_curve.png` |
| `e0` | per-query oracle α and Gap, feature learnability, divergence distribution and switch rule | `results/e0_report.md`, `results/figures/e0_*.png` |
| `pairs` | 100 pairs per train query (main: 50 hard from ranks 21–100 + 50 uniform; plus `noskip` and `uniform` variants), validated | `cache/pairs_*.npz`, `results/pairs_stats.json` |
| `gradcheck` | finite-difference checks of ∇J and ∇²J for R1–R4 (synthetic, real subset, edge cases) | `results/gradcheck.json` |
| `e2` | R1 with F_text / F5 / F8 × 3 λ; picks the feature set | `results/e2_features.md` |
| `e3` | R1 (3 λ) and R2–R4 (3 λ × 4 γ), BFGS from the anchor start; dev selection | `results/e3_models.md`, `results/e3_runs.json`, `results/figures/e3_sensitivity.png` |
| `lock` | writes `results/lock.json` (selected models, all decisions); refuses to overwrite | `results/lock.json` |
| `e4` | GD (Armijo), BFGS, L-BFGS-B, Newton-CG, trust-exact on locked R1 and R3 from 3 starts; Hessian at solutions, GD rate theory vs observed, κ vs γ, nonconvexity probe | `results/e4_report.md`, `results/e4_runs.json`, `results/figures/e4_*.png` |
| `e5` | R3 without g feature, R3 without anchor, uniform negatives (R1–R3), no head skipping (R3) | `results/e5.json` |
| `e6` | R1–R3 with weak negatives resampled under seeds 17, 23, 42, 71, 101 | `results/e6.json`, `cache/pairs_main_s*.npz` |
| `test` | all locked systems on test (and dedup subset); paired bootstrap; degradation; E5/E6 on test; cases; cost | `results/test_report.md`, `results/test_results.json`, `results/figures/test_overview.png` |

Every fit is stored once in `results/runs/<key>.json` (spec, weights, J parts, ‖∇J‖∞, iterations, time,
Hessian eigenvalues at the solution, dev metrics) and reused if the same spec is requested again.

Tests (including an exact cross-check of the metric code against `pytrec_eval`):

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

## Protocol safeguards

* **Test split is evaluated only by the `test` stage**, which refuses to run without `results/lock.json`.
* **`results/decisions.json`** stores every pre-registered choice (dev-best α, α0, RRF k, E0 band,
  divergence measure, dedup test subset). Once written, an entry is locked: re-runs reuse it and warn if
  a recomputed value differs. `--force-decisions` overwrites and should only be used to fix a bug.
* **Ties** are broken by document ID everywhere (documents are stored in ID order and sorted stably).
* **Dense model revision** is resolved at first run and recorded (`results/text_stats.json`,
  `results/run_info.json`). Pin it in `config.yaml` (`dense.revision`) before the final runs.

## Layout

```
config.yaml        all fixed settings
run_all.py         entry point
arag/              data, text, retrievers, fusion, evaluate, features, pipeline, experiments (E0/E1),
                   pairs, objective (J, ∇J, ∇²J), model (fitting, registry), training (phase-2 stages),
                   solvers + e4 (E4), analysis (E5, E6, test)
tests/             unit tests and pytrec_eval cross-check
data/ cache/       downloaded data and score caches (git-ignored)
results/           tables, figures, decisions, run logs
```
