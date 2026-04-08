"""Ranking evaluation metrics: MAP, NDCG, Precision, Recall, MRR."""

from __future__ import annotations

import math

from src.evaluation.base_evaluator import BaseEvaluator


class RankingEvaluator(BaseEvaluator):
    """Compute MAP@k, NDCG@k, Precision@k, Recall@k, MRR@k.

    Parameters
    ----------
    k:
        Cutoff depth (default 10).  Typically taken from ``EvalConfig.k``.
    """

    def __init__(self, k: int = 10) -> None:
        self._k = k

    # ── public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: list[list[int]],
        labels: list[set[int]],
    ) -> dict[str, float]:
        """Return metric dict averaged over all queries."""
        if not predictions or not labels:
            return self._zero_metrics()

        map_vals, ndcg_vals, prec_vals, rec_vals, mrr_vals = [], [], [], [], []

        for pred, rel in zip(predictions, labels):
            if not rel:
                continue
            pred_k = pred[: self._k]
            map_vals.append(self._ap(pred_k, rel))
            ndcg_vals.append(self._ndcg(pred_k, rel))
            prec_vals.append(self._precision(pred_k, rel))
            rec_vals.append(self._recall(pred_k, rel))
            mrr_vals.append(self._mrr(pred_k, rel))

        if not map_vals:
            return self._zero_metrics()

        k = self._k
        return {
            f"map@{k}": float(sum(map_vals) / len(map_vals)),
            f"ndcg@{k}": float(sum(ndcg_vals) / len(ndcg_vals)),
            f"precision@{k}": float(sum(prec_vals) / len(prec_vals)),
            f"recall@{k}": float(sum(rec_vals) / len(rec_vals)),
            f"mrr@{k}": float(sum(mrr_vals) / len(mrr_vals)),
        }

    # ── per-query metrics ─────────────────────────────────────────────────

    def _ap(self, pred: list[int], rel: set[int]) -> float:
        """Average Precision at k."""
        hits = 0
        precision_sum = 0.0
        for rank, mid in enumerate(pred, start=1):
            if mid in rel:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / min(len(rel), self._k) if hits else 0.0

    def _ndcg(self, pred: list[int], rel: set[int]) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, mid in enumerate(pred, start=1)
            if mid in rel
        )
        # Ideal DCG: place all relevant items at the top
        ideal_hits = min(len(rel), self._k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0

    def _precision(self, pred: list[int], rel: set[int]) -> float:
        """Precision at k."""
        hits = sum(1 for mid in pred if mid in rel)
        return hits / len(pred) if pred else 0.0

    def _recall(self, pred: list[int], rel: set[int]) -> float:
        """Recall at k."""
        hits = sum(1 for mid in pred if mid in rel)
        return hits / len(rel) if rel else 0.0

    def _mrr(self, pred: list[int], rel: set[int]) -> float:
        """Mean Reciprocal Rank."""
        for rank, mid in enumerate(pred, start=1):
            if mid in rel:
                return 1.0 / rank
        return 0.0

    def _zero_metrics(self) -> dict[str, float]:
        k = self._k
        return {
            f"map@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
        }
