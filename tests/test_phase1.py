"""Phase-1 checks. Run:  .venv/Scripts/python.exe -m pytest tests -q   (or run this file directly)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arag.config import ROOT, load_config  # noqa: E402
from arag.evaluate import Evaluator  # noqa: E402
from arag.features import jaccard_distance, rbo_ext  # noqa: E402
from arag.fusion import rank, rank_positions, standardize  # noqa: E402


def test_rbo_properties():
    a = list(range(20))
    assert abs(rbo_ext(a, a, 0.9) - 1.0) < 1e-12
    assert abs(rbo_ext(a, list(range(100, 120)), 0.9)) < 1e-12
    b = a[:]
    b[0], b[1] = b[1], b[0]          # swap at the head ...
    c = a[:]
    c[18], c[19] = c[19], c[18]      # ... hurts more than a swap at the tail
    assert rbo_ext(a, b, 0.9) < rbo_ext(a, c, 0.9) < 1.0


def test_jaccard():
    t = np.array([[0, 1, 2, 3]])
    assert jaccard_distance(t, t)[0] == 0.0
    assert jaccard_distance(t, np.array([[4, 5, 6, 7]]))[0] == 1.0
    assert abs(jaccard_distance(t, np.array([[0, 1, 8, 9]]))[0] - (1 - 2 / 6)) < 1e-12


def test_ties_by_doc_index_and_positions():
    S = np.array([[1.0, 3.0, 3.0, 0.0]])
    order = rank(S)
    assert order.tolist() == [[1, 2, 0, 3]]
    assert rank_positions(order).tolist() == [[3, 1, 2, 4]]


def test_standardize_degenerate():
    Z, deg = standardize(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]), 1e-8)
    assert deg.tolist() == [True, False] and np.all(Z[0] == 0)
    assert abs(Z[1].mean()) < 1e-12


def test_evaluator_matches_pytrec_eval():
    """Our NDCG/Recall/MAP@10 on the cached BM25 and Dense dev runs must equal pytrec_eval."""
    import pytrec_eval

    from arag.pipeline import get_dataset, load_scores

    cfg = load_config()
    if not (cfg["paths"]["cache_dir"] / "scores_bm25.npy").exists():
        import pytest
        pytest.skip("run `python run_all.py` first")
    ds = get_dataset(cfg)
    B, D = load_scores(cfg)
    for split in ("dev", "train"):
        ev = Evaluator(ds.qrels[split], 10)
        qrels = {ds.query_ids[r]: {ds.doc_ids[d]: v for d, v in ds.qrels[split][r].items()} for r in ev.rows}
        trec = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10", "map_cut.10"})
        for S in (B, D):
            top = rank(S[ev.rows], 100)
            # strictly decreasing pseudo-scores make trec_eval use exactly our order
            run = {ds.query_ids[r]: {ds.doc_ids[d]: float(100 - j) for j, d in enumerate(row)} for r, row in zip(ev.rows, top)}
            ref = trec.evaluate(run)
            ours = ev.per_query_from_topk(top[:, :10])
            for key, name in (("ndcg@10", "ndcg_cut_10"), ("recall@10", "recall_10"), ("map@10", "map_cut_10")):
                theirs = np.array([ref[ds.query_ids[r]][name] for r in ev.rows])
                assert np.allclose(ours[key], theirs, atol=1e-9), (split, key, np.abs(ours[key] - theirs).max())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
