"""Orchestrator that runs ranking + diversity evaluation and returns an EvalReport."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.evaluation.ranking_metrics import RankingEvaluator
from src.evaluation.diversity_metrics import DiversityEvaluator

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Holds all evaluation results for one experiment run.

    Attributes
    ----------
    config_name:
        Human-readable name identifying the experiment (e.g. ``"xgb_full_tuned"``).
    ranking:
        Ranking metric dict from :class:`RankingEvaluator`.
    diversity:
        Diversity metric dict from :class:`DiversityEvaluator`.
    split:
        Dataset split this report was computed on (``"val"`` or ``"test"``).
    n_users:
        Number of warm users evaluated.
    """

    config_name: str
    ranking: dict[str, float]
    diversity: dict[str, float]
    split: str = "test"
    n_users: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_mlflow_metrics(self) -> dict[str, float]:
        """Return a flat dict of all scalar metrics suitable for MLflow logging.

        MLflow forbids ``@`` in metric names, so ``map@10`` → ``map_at_10``.
        """
        def _sanitize(key: str) -> str:
            return key.replace("@", "_at_")

        metrics: dict[str, float] = {}
        for k, v in self.ranking.items():
            if isinstance(v, (int, float)):
                metrics[_sanitize(k)] = float(v)
        for k, v in self.diversity.items():
            if isinstance(v, (int, float)):
                metrics[_sanitize(k)] = float(v)
        for k, v in self.extra.items():
            if isinstance(v, (int, float)):
                metrics[_sanitize(k)] = float(v)
        return metrics

    def primary_metric(self, k: int = 10) -> float:
        """Return warm MAP@k — the primary evaluation metric."""
        return self.ranking.get(f"map@{k}", 0.0)


class Evaluator:
    """Orchestrates ranking and diversity evaluation.

    Parameters
    ----------
    config:
        ``ExperimentConfig`` — used for ``eval.k``.
    item_popularity:
        Mapping from movie_id → log_rating_count.
    item_genre_vectors:
        Mapping from movie_id → genre binary vector.
    cold_item_ids:
        Set of cold item IDs.
    catalog_size:
        Total catalogue size.
    """

    def __init__(
        self,
        config,
        item_popularity: dict[int, float],
        item_genre_vectors: dict[int, list[float]],
        cold_item_ids: set[int],
        catalog_size: int,
    ) -> None:
        k = config.eval.k
        self._ranking_eval = RankingEvaluator(k=k)
        self._diversity_eval = DiversityEvaluator(
            catalog_size=catalog_size,
            item_popularity=item_popularity,
            item_genre_vectors=item_genre_vectors,
            cold_item_ids=cold_item_ids,
            k=k,
        )
        self._config = config

    def evaluate_full(
        self,
        predictions: list[list[int]],
        labels: list[set[int]],
        config_name: str,
        split: str = "test",
    ) -> EvalReport:
        """Run all metrics and return an :class:`EvalReport`.

        Parameters
        ----------
        predictions:
            One list per user: ranked movie IDs (best first).
        labels:
            One set per user: relevant movie IDs.
        config_name:
            Experiment / model name for the report.
        split:
            "val" or "test".
        """
        logger.info(
            "Evaluator.evaluate_full: %d users, split=%s, config=%s",
            len(predictions),
            split,
            config_name,
        )

        # ── Fix 5: debug first 3 predictions — overlap must be > 0 ──────────
        for i, (preds, labs) in enumerate(zip(predictions[:3], labels[:3])):
            overlap = len(set(preds) & labs)
            logger.info(
                "  [eval debug user %d] pool(top-k)=%d  val_positives=%d"
                "  overlap(top-k ∩ positives)=%d"
                "  [GOOD: overlap>0 | LEAKAGE: overlap≈positives | BUG: overlap=0]",
                i, len(preds), len(labs), overlap,
            )

        ranking_metrics = self._ranking_eval.evaluate(predictions, labels)
        diversity_metrics = self._diversity_eval.evaluate(predictions, labels)

        report = EvalReport(
            config_name=config_name,
            ranking=ranking_metrics,
            diversity=diversity_metrics,
            split=split,
            n_users=len(predictions),
        )

        logger.info(
            "EvalReport: map@%d=%.4f ndcg@%d=%.4f coverage=%.4f",
            self._config.eval.k,
            report.primary_metric(self._config.eval.k),
            self._config.eval.k,
            ranking_metrics.get(f"ndcg@{self._config.eval.k}", 0.0),
            diversity_metrics.get("catalog_coverage", 0.0),
        )
        return report
