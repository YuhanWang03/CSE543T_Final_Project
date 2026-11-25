"""Phase-2 checks: objective derivatives, anchor weights, pair sampler. Fast, synthetic-only (plus cached pairs if present)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arag.objective import Objective, anchor_weights, check_derivatives  # noqa: E402
from arag.pairs import SRC_FALLBACK, SRC_HARD, hard_candidates, sample_pairs, validate_pairs  # noqa: E402


def _toy(seed=0, nQ=25, d=5):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(nQ), rng.normal(size=(nQ, d - 1))])
    q = np.repeat(np.arange(nQ), rng.integers(1, 15, nQ))
    return rng, X, q, rng.normal(size=len(q)) * 2, rng.normal(size=len(q)) * 2, rng.uniform(0, 1, nQ)


def test_derivatives_all_models():
    rng, X, q, dB, dD, g = _toy()
    for model in ("R1", "R2", "R3", "R4"):
        obj = Objective(X, q, dB, dD, 0.2, 1e-2, 0.0 if model == "R1" else 0.7, anchor_weights(model, g)[0])
        for w in (rng.normal(size=X.shape[1]), rng.normal(size=X.shape[1]) * 10):
            r = check_derivatives(obj, w)
            assert r["grad_rel_err_h1e-06"] < 1e-5 or r["grad_abs_err_h1e-06"] < 1e-9, (model, r)
            assert r["hess_rel_err_fro"] < 1e-4, (model, r)


def test_value_and_grad_consistent_and_bias_unpenalised():
    rng, X, q, dB, dD, g = _toy(1)
    obj = Objective(X, q, dB, dD, 0.2, 0.5, 0.3, np.ones(X.shape[0]))
    w = rng.normal(size=X.shape[1])
    J, gr = obj.value_and_grad(w)
    assert np.isclose(J, obj.value(w)) and np.allclose(gr, obj.grad(w))
    w2 = w.copy()
    w2[0] += 1.0
    assert np.isclose(obj.parts(w)["l2"], obj.parts(w2)["l2"])


def test_anchor_weights():
    g = np.array([0.0, 0.5, 1.0])
    assert np.allclose(anchor_weights("R2", g)[0], 1)
    assert np.isclose(anchor_weights("R3", g)[0].mean(), 1) and np.isclose(anchor_weights("R4", g)[0].mean(), 1)
    assert anchor_weights("R3", np.zeros(3))[1] and anchor_weights("R4", np.ones(3))[1]


def test_hard_candidates_skip_head():
    tb, td = np.arange(100), np.arange(100)[::-1].copy()
    H = hard_candidates(tb, td, P={25}, variant="main")
    assert 25 not in H and not (set(H) & (set(range(20)) | set(range(80, 100))))
    assert set(hard_candidates(tb, td, P=set(), variant="noskip")) == set(range(100))


def test_sampler_counts_and_validity():
    n_docs = 300
    rng = np.random.default_rng(0)
    top = np.stack([rng.permutation(n_docs)[:100] for _ in range(4)])
    # q2: every doc except 7 is relevant, so U = {7}
    qrels = {0: {1: 1, 2: 1}, 1: {5: 1}, 2: {i: 1 for i in range(300) if i != 7}, 3: {9: 1}}
    arr, st = sample_pairs([0, 1, 2, 3], qrels, top, top.copy(), n_docs, "main", 17)
    validate_pairs(arr, qrels)
    assert np.bincount(arr["q_row"], minlength=4).tolist() == [100, 100, 100, 100]
    assert set(arr["neg"][arr["q_row"] == 2].tolist()) == {7}
    assert st["duplicate_fraction"] == 0.0
    arr2, _ = sample_pairs([0, 1, 2, 3], qrels, top, top.copy(), n_docs, "main", 17)
    assert all(np.array_equal(arr[k], arr2[k]) for k in arr)  # deterministic


def test_sampler_empty_hard_set_and_short_query():
    # 12 docs: skipping the top-20 leaves no hard candidates; only 11 unique pairs exist
    top = np.arange(12)[None, :]
    arr, st = sample_pairs([0], {0: {0: 1}}, top, top.copy(), 12, "main", 17)
    validate_pairs(arr, {0: {0: 1}})
    assert len(arr["q_row"]) == 11
    assert st["queries_with_empty_hard_set"] == 1 and st["queries_short_of_pairs"] == 1
    assert not np.any(arr["src"] == SRC_HARD)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
