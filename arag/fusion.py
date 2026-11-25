"""Score standardisation, ranking with deterministic tie-breaks, fixed fusion and RRF."""
from __future__ import annotations

import numpy as np


def standardize(S: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-query z-score over the whole corpus: (S - mean) / (std + eps).

    Rows whose (population) std is below eps are set to 0 and flagged as degenerate.
    """
    S = S.astype(np.float64)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    Z = (S - mu) / (sd + eps)
    degenerate = sd[:, 0] < eps
    Z[degenerate] = 0.0
    return Z, degenerate


def rank(S: np.ndarray, k: int | None = None) -> np.ndarray:
    """Doc indices sorted by descending score; ties by ascending doc index.

    Documents are stored in lexicographic ID order, so index order == doc-ID order,
    and a stable sort of -S implements "ties broken by document ID".
    """
    idx = np.argsort(-S, axis=1, kind="stable")
    return idx if k is None else idx[:, :k]


def fixed_fusion(Bz: np.ndarray, Dz: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * Bz + (1.0 - alpha) * Dz


def rank_positions(order: np.ndarray) -> np.ndarray:
    """Inverse of `rank`: 1-based position of every document."""
    pos = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    pos[rows, order] = np.arange(1, order.shape[1] + 1)
    return pos


def rrf(pos_b: np.ndarray, pos_d: np.ndarray, k: int) -> np.ndarray:
    return 1.0 / (k + pos_b) + 1.0 / (k + pos_d)


def select_with_ties(values, scores, prefer) -> float:
    """argmax of `scores`; exact ties resolved by the `prefer` key (smaller is better)."""
    scores = np.asarray(scores, dtype=np.float64)
    best = scores.max()
    cands = [v for v, s in zip(values, scores) if np.isclose(s, best, rtol=0, atol=1e-12)]
    return min(cands, key=prefer)
