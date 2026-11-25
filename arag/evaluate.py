"""Per-query NDCG@k, Recall@k and MAP@k (trec_eval definitions) on a fixed ranking.

Definitions follow trec_eval's ndcg_cut / recall / map_cut as used by BEIR via pytrec_eval:
  * NDCG: gain = graded qrels label (linear), discount 1/log2(rank+1), ideal ranking built
    from all judged documents of the query.
  * Recall@k: relevant (label > 0) retrieved in top-k / all relevant.
  * MAP@k: sum of precision@i over relevant docs found in top-k, divided by all relevant.
Unjudged documents count as non-relevant. We score our own ranking (ties already broken
by document ID), so trec_eval's own tie-breaking never applies. tests/test_evaluate.py
cross-checks these numbers against pytrec_eval.
"""
from __future__ import annotations

import numpy as np

from .fusion import rank


class Evaluator:
    def __init__(self, qrels_split: dict[int, dict[int, int]], k: int = 10):
        self.k = k
        self.rows = np.array(sorted(qrels_split), dtype=np.int64)
        self.rel = [qrels_split[r] for r in self.rows]
        self.n_rel = np.array([sum(1 for v in d.values() if v > 0) for d in self.rel], dtype=np.float64)
        disc = 1.0 / np.log2(np.arange(2, k + 2))
        self.disc = disc
        idcg = []
        for d in self.rel:
            ideal = np.sort([v for v in d.values() if v > 0])[::-1][:k]
            idcg.append(float(np.sum(ideal * disc[: len(ideal)])))
        self.idcg = np.array(idcg)

    def per_query_from_topk(self, topk: np.ndarray) -> dict[str, np.ndarray]:
        """topk: [len(rows), k] doc indices in rank order for self.rows."""
        k = self.k
        gains = np.zeros((len(self.rows), k))
        for i, (d, docs) in enumerate(zip(self.rel, topk)):
            for j, doc in enumerate(docs[:k]):
                g = d.get(int(doc), 0)
                if g > 0:
                    gains[i, j] = g
        hit = gains > 0
        ndcg = np.divide((gains * self.disc).sum(1), self.idcg, out=np.zeros(len(self.rows)), where=self.idcg > 0)
        recall = np.divide(hit.sum(1), self.n_rel, out=np.zeros(len(self.rows)), where=self.n_rel > 0)
        prec_at = np.cumsum(hit, axis=1) / np.arange(1, k + 1)
        ap = np.divide((prec_at * hit).sum(1), self.n_rel, out=np.zeros(len(self.rows)), where=self.n_rel > 0)
        return {f"ndcg@{k}": ndcg, f"recall@{k}": recall, f"map@{k}": ap}

    def per_query(self, S_full: np.ndarray) -> dict[str, np.ndarray]:
        """S_full: [n_queries_total, n_docs] score matrix (rows indexed by query row)."""
        return self.per_query_rows(S_full[self.rows])

    def per_query_rows(self, S_rows: np.ndarray) -> dict[str, np.ndarray]:
        """S_rows: scores already restricted to self.rows (same order)."""
        return self.per_query_from_topk(rank(S_rows, self.k))

    @staticmethod
    def mean(m: dict[str, np.ndarray]) -> dict[str, float]:
        return {k: float(v.mean()) for k, v in m.items()}
