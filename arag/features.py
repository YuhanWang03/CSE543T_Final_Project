"""Label-free query features and divergence measures (plan v3 §6)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BASE_FEATURES = ["query_len", "mean_idf", "bm25_gap", "dense_gap", "g"]
EXT_FEATURES = ["bm25_top1_raw_log", "query_coverage", "dense_top10_std"]


def jaccard_distance(top_b: np.ndarray, top_d: np.ndarray) -> np.ndarray:
    """g_J = 1 - |T_B ∩ T_D| / |T_B ∪ T_D| row-wise for top-k index arrays."""
    out = np.empty(len(top_b))
    for i, (a, b) in enumerate(zip(top_b, top_d)):
        sa, sb = set(a.tolist()), set(b.tolist())
        out[i] = 1.0 - len(sa & sb) / len(sa | sb)
    return out


def rbo_ext(a, b, p: float) -> float:
    """Extrapolated RBO (Webber, Moffat & Zobel 2010, eq. 32) for two equal-length rankings.

    RBO_ext = (X_k / k) p^k + (1 - p)/p * sum_{d=1..k} (X_d / d) p^d, X_d = |a[:d] ∩ b[:d]|.
    Lists are tie-free (ties already broken by document ID).
    """
    k = len(a)
    assert len(b) == k
    seen_a, seen_b, x, acc = set(), set(), 0, 0.0
    for d in range(1, k + 1):
        u, v = a[d - 1], b[d - 1]
        if u == v:
            x += 1
        else:
            x += (u in seen_b) + (v in seen_a)
        seen_a.add(u)
        seen_b.add(v)
        acc += (x / d) * p ** d
    return (x / k) * p ** k + (1 - p) / p * acc


def rbo_distance(top_b: np.ndarray, top_d: np.ndarray, p: float) -> np.ndarray:
    return np.array([1.0 - rbo_ext(a.tolist(), b.tolist(), p) for a, b in zip(top_b, top_d)])


def document_frequencies(doc_tokens) -> Counter:
    df = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    return df


def query_features(query_tokens, df: Counter, n_docs: int, B_raw, Bz, Dz, top_b, top_d, g_j, g_rbo) -> dict[str, np.ndarray]:
    """Raw (unstandardised) features for every query row. g is reported as both variants;
    the pre-registered decision (results/decisions.json) says which one plays the role of g."""
    nq = len(query_tokens)
    feats = {name: np.zeros(nq) for name in ["query_len", "mean_idf", "bm25_gap", "dense_gap",
                                             "bm25_top1_raw_log", "query_coverage", "dense_top10_std"]}
    rows = np.arange(nq)
    for i, toks in enumerate(query_tokens):
        feats["query_len"][i] = len(toks)
        if toks:
            idf = [math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in toks]
            feats["mean_idf"][i] = float(np.mean(idf))
            feats["query_coverage"][i] = float(np.mean([df.get(t, 0) > 0 for t in toks]))
    feats["bm25_gap"] = Bz[rows, top_b[:, 0]] - Bz[rows, top_b[:, 1]]
    feats["dense_gap"] = Dz[rows, top_d[:, 0]] - Dz[rows, top_d[:, 1]]
    feats["bm25_top1_raw_log"] = np.log1p(B_raw.max(axis=1).astype(np.float64))
    feats["dense_top10_std"] = np.take_along_axis(Dz, top_d[:, :10], axis=1).std(axis=1)
    feats["g_jaccard"] = g_j
    feats["g_rbo"] = g_rbo
    return feats
