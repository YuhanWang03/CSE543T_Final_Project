"""Weak-supervision training pairs (plan v3 §5).

For a train query q: P_q = labelled relevant docs (train qrels), U_q = all other docs (unlabelled,
treated as weak negatives), H_q = hard candidates from the frozen retrievers.

Variants
  main     : 50 hard (ranks 21-100 of either retriever, minus both top-20 lists and P_q) + 50 uniform from U_q
  noskip   : like main, but H_q = union of both top-100 lists minus P_q (v2 rule, no head skipping)
  uniform  : 100 negatives uniform from U_q
Shortfalls of hard pairs (H_q empty or too few unique pairs) are filled from U_q and recorded as fallback.
"""
from __future__ import annotations

import hashlib

import numpy as np

SRC_UNIFORM, SRC_HARD, SRC_FALLBACK = 0, 1, 2
VARIANTS = ("main", "noskip", "uniform")


def hard_candidates(top_b, top_d, P: set, variant: str, skip: int = 20, depth: int = 100) -> list[int]:
    if variant == "uniform":
        return []
    if variant == "noskip":
        cand = set(top_b[:depth].tolist()) | set(top_d[:depth].tolist())
    elif variant == "main":
        head = set(top_b[:skip].tolist()) | set(top_d[:skip].tolist())
        cand = (set(top_b[skip:depth].tolist()) | set(top_d[skip:depth].tolist())) - head
    else:
        raise ValueError(variant)
    return sorted(cand - P)


def _draw_unique(rng, P: np.ndarray, N: np.ndarray, k: int, chosen: set, max_tries: int = 200_000):
    """Up to k (p, n) pairs from P x N not already in `chosen` (uniform; exhaustive when small)."""
    if k <= 0 or len(P) == 0 or len(N) == 0:
        return []
    out = []
    if len(P) * len(N) <= 4 * k:  # small pool: enumerate, then sample without replacement
        pool = [(int(p), int(n)) for p in P for n in N if (int(p), int(n)) not in chosen]
        if len(pool) <= k:
            out = pool
        else:
            out = [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    else:
        tries = 0
        while len(out) < k and tries < max_tries:
            tries += 1
            pair = (int(P[rng.integers(len(P))]), int(N[rng.integers(len(N))]))
            if pair not in chosen:
                chosen.add(pair)
                out.append(pair)
        return out
    chosen.update(out)
    return out


def sample_pairs(train_rows, qrels_train, top_b, top_d, n_docs: int, variant: str, seed: int,
                 n_pairs: int = 100, n_hard: int = 50, skip: int = 20, depth: int = 100) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    q_rows, pos, neg, src = [], [], [], []
    stats = {"variant": variant, "seed": seed, "skipped_no_positive": 0, "skipped_no_unlabelled": 0,
             "queries_short_of_pairs": 0, "fallback_pairs": 0, "queries_with_empty_hard_set": 0,
             "hard_set_size": []}
    all_docs = np.arange(n_docs)
    for r in train_rows:  # fixed order: sorted train query rows
        P_list = sorted(d for d, v in qrels_train[r].items() if v > 0)
        if not P_list:
            stats["skipped_no_positive"] += 1
            continue
        P = np.array(P_list)
        U = np.setdiff1d(all_docs, P, assume_unique=True)
        if len(U) == 0:
            stats["skipped_no_unlabelled"] += 1
            continue
        H = np.array(hard_candidates(top_b[r], top_d[r], set(P_list), variant, skip, depth), dtype=np.int64)
        stats["hard_set_size"].append(len(H))
        if variant != "uniform" and len(H) == 0:
            stats["queries_with_empty_hard_set"] += 1
        chosen: set = set()
        want_hard = 0 if variant == "uniform" else n_hard
        hard = _draw_unique(rng, P, H, want_hard, chosen)
        short = want_hard - len(hard)
        uni = _draw_unique(rng, P, U, n_pairs - want_hard, chosen)
        fb = _draw_unique(rng, P, U, short, chosen)
        stats["fallback_pairs"] += len(fb)
        got = hard + uni + fb
        if len(got) < n_pairs:
            stats["queries_short_of_pairs"] += 1
        for (p, n), s in zip(got, [SRC_HARD] * len(hard) + [SRC_UNIFORM] * len(uni) + [SRC_FALLBACK] * len(fb)):
            q_rows.append(r)
            pos.append(p)
            neg.append(n)
            src.append(s)
    arr = {"q_row": np.array(q_rows, dtype=np.int32), "pos": np.array(pos, dtype=np.int32),
           "neg": np.array(neg, dtype=np.int32), "src": np.array(src, dtype=np.int8)}
    hs = np.array(stats.pop("hard_set_size") or [0])
    n_q = len(np.unique(arr["q_row"]))
    keys = arr["q_row"].astype(np.int64) * n_docs * n_docs + arr["pos"].astype(np.int64) * n_docs + arr["neg"]
    stats.update({
        "n_pairs": int(len(arr["q_row"])), "n_queries": int(n_q),
        "pairs_per_query_min": int(np.bincount(arr["q_row"])[np.unique(arr["q_row"])].min()),
        "hard_fraction": float(np.mean(arr["src"] == SRC_HARD)),
        "fallback_fraction": float(np.mean(arr["src"] == SRC_FALLBACK)),
        "duplicate_fraction": float(1 - len(np.unique(keys)) / len(keys)),
        "hard_set_size_percentiles": {p: float(np.percentile(hs, p)) for p in (0, 10, 50, 90, 100)},
        "md5": pairs_md5(arr),
    })
    return arr, stats


def validate_pairs(arr: dict, qrels_train) -> None:
    """Every pair: pos != neg, pos in P_q, neg not in P_q (train labels only)."""
    for r, p, n in zip(arr["q_row"], arr["pos"], arr["neg"]):
        rel = qrels_train[int(r)]
        if p == n or rel.get(int(p), 0) <= 0 or rel.get(int(n), 0) > 0:
            raise AssertionError(f"invalid pair q={r} pos={p} neg={n}")


def pairs_md5(arr: dict) -> str:
    h = hashlib.md5()
    for k in ("q_row", "pos", "neg", "src"):
        h.update(np.ascontiguousarray(arr[k]).tobytes())
    return h.hexdigest()
