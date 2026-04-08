"""Diversity evaluation metrics: catalog coverage, ILD, novelty, cold exposure."""

from __future__ import annotations

import math

from src.evaluation.base_evaluator import BaseEvaluator


class DiversityEvaluator(BaseEvaluator):
    """Compute diversity and coverage metrics over a full recommendation run.

    Parameters
    ----------
    catalog_size:
        Total number of distinct items in the catalogue.
    item_popularity:
        Mapping from movie_id → log_rating_count (used for novelty).
    item_genre_vectors:
        Mapping from movie_id → genre binary vector (list of 0/1 floats).
        Used for intra-list diversity.
    cold_item_ids:
        Set of item IDs classified as cold (rating_count < cold_item_threshold).
    k:
        Recommendation list cutoff.
    """

    def __init__(
        self,
        catalog_size: int,
        item_popularity: dict[int, float],
        item_genre_vectors: dict[int, list[float]],
        cold_item_ids: set[int],
        k: int = 10,
    ) -> None:
        self._catalog_size = max(catalog_size, 1)
        self._popularity = item_popularity
        self._genre_vectors = item_genre_vectors
        self._cold_items = cold_item_ids
        self._k = k

    # ── public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        predictions: list[list[int]],
        labels: list[set[int]],
    ) -> dict[str, float]:
        """Compute all diversity metrics and return as a flat dict."""
        all_recs = [pred[: self._k] for pred in predictions if pred]
        if not all_recs:
            return self._zero_metrics()

        return {
            "catalog_coverage": self._catalog_coverage(all_recs),
            "intra_list_diversity": self._mean_ild(all_recs),
            "novelty": self._mean_novelty(all_recs),
            "cold_item_exposure": self._cold_item_exposure(all_recs),
        }

    # ── metric implementations ─────────────────────────────────────────────

    def _catalog_coverage(self, all_recs: list[list[int]]) -> float:
        """Fraction of the catalogue that appears in at least one rec list."""
        recommended = {mid for rec in all_recs for mid in rec}
        return len(recommended) / self._catalog_size

    def _mean_ild(self, all_recs: list[list[int]]) -> float:
        """Mean intra-list diversity (1 − avg pairwise genre cosine similarity)."""
        ild_values = [self._ild(rec) for rec in all_recs]
        return sum(ild_values) / len(ild_values) if ild_values else 0.0

    def _ild(self, rec: list[int]) -> float:
        """Intra-list diversity for a single recommendation list."""
        vecs = [self._genre_vectors.get(mid) for mid in rec]
        vecs = [v for v in vecs if v is not None]
        if len(vecs) < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = self._cosine(vecs[i], vecs[j])
                total += 1.0 - sim
                count += 1
        return total / count if count else 0.0

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        denom = norm_a * norm_b
        return dot / denom if denom > 1e-8 else 0.0

    def _mean_novelty(self, all_recs: list[list[int]]) -> float:
        """Self-information novelty (lower popularity = higher novelty)."""
        novelty_values = []
        for rec in all_recs:
            scores = []
            for mid in rec:
                pop = self._popularity.get(mid, 0.0)
                # popularity stored as log_rating_count; undo log1p then re-log
                raw_count = math.expm1(max(pop, 0.0))
                prob = (raw_count + 1) / (self._catalog_size + 1)
                scores.append(-math.log2(prob))
            if scores:
                novelty_values.append(sum(scores) / len(scores))
        return sum(novelty_values) / len(novelty_values) if novelty_values else 0.0

    def _cold_item_exposure(self, all_recs: list[list[int]]) -> float:
        """Fraction of recommended items that are cold."""
        all_items = [mid for rec in all_recs for mid in rec]
        if not all_items:
            return 0.0
        cold_count = sum(1 for mid in all_items if mid in self._cold_items)
        return cold_count / len(all_items)

    def _zero_metrics(self) -> dict[str, float]:
        return {
            "catalog_coverage": 0.0,
            "intra_list_diversity": 0.0,
            "novelty": 0.0,
            "cold_item_exposure": 0.0,
        }
