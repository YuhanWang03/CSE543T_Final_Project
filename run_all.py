"""Single entry point.

    python run_all.py --stages phase1       # prepare, retrieve, features, e1, e0
    python run_all.py --stages phase2       # pairs, gradcheck, e2, e3
    python run_all.py --stages lock         # freeze the configuration (only after reviewing E3)
    python run_all.py --stages e2,e3        # any subset, re-using caches
"""
from __future__ import annotations

import argparse
import os
import sys

PHASE1 = ["prepare", "retrieve", "features", "e1", "e0"]
PHASE2 = ["pairs", "gradcheck", "e2", "e3"]
PHASE3 = ["e4", "e5", "e6", "test"]   # all after `lock`


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", default=",".join(PHASE1))
    ap.add_argument("--force-decisions", action="store_true",
                    help="overwrite locked entries in results/decisions.json (use only to fix a bug, and say so)")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to GBK; reports contain α, −, etc.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from arag.config import load_config
    cfg = load_config(args.config)
    threads = str(cfg["runtime"]["threads"])
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, threads)

    if args.force_decisions:
        import arag.config as c
        _orig = c.record_decision
        c.record_decision = lambda cfg, k, v, force=False: _orig(cfg, k, v, force=True)

    from arag import analysis, e4, experiments, pipeline, training
    stages = {"prepare": pipeline.stage_prepare, "retrieve": pipeline.stage_retrieve,
              "features": pipeline.stage_features, "e1": experiments.stage_e1, "e0": experiments.stage_e0,
              "pairs": training.stage_pairs, "gradcheck": training.stage_gradcheck,
              "e2": training.stage_e2, "e3": training.stage_e3, "lock": training.stage_lock,
              "e4": e4.stage_e4, "e5": analysis.stage_e5, "e6": analysis.stage_e6, "test": analysis.stage_test}
    aliases = {"phase1": PHASE1, "phase2": PHASE2, "phase3": PHASE3}
    todo = [t for s in args.stages.split(",") if s.strip() for t in aliases.get(s.strip(), [s.strip()])]
    unknown = [s for s in todo if s not in stages]
    if unknown:
        sys.exit(f"unknown stage(s): {unknown}; available: {list(stages)}")
    info = {}
    for s in todo:
        print(f"\n===== stage: {s} =====")
        info.update(stages[s](cfg) or {})
    pipeline.write_run_info(cfg, info, todo)


if __name__ == "__main__":
    main()
