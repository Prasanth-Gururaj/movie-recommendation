"""Evaluation configuration: metrics, diversity targets, warm-user thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config.base_config import BaseConfig

_VALID_K: frozenset[int] = frozenset({5, 10, 20})


@dataclass
class EvalConfig(BaseConfig):
    # ── primary ranking cutoff ────────────────────────────────────────────────
    k: int = 10                             # MAP@k, NDCG@k, etc.

    # ── warm-user eval filter ─────────────────────────────────────────────────
    warm_user_min_positives: int = 3

    # ── ranking metrics ───────────────────────────────────────────────────────
    metrics: list[str] = field(
        default_factory=lambda: ["map", "ndcg", "precision", "recall", "mrr"]
    )

    # ── diversity / coverage metrics ──────────────────────────────────────────
    diversity_metrics: list[str] = field(
        default_factory=lambda: [
            "catalog_coverage",
            "intra_list_diversity",
            "novelty",
            "cold_item_exposure",
        ]
    )

    # ── diversity targets ─────────────────────────────────────────────────────
    catalog_coverage_target: float = 0.15
    intra_list_diversity_target: float = 0.6
    cold_item_exposure_target: float = 0.05

    def validate(self) -> None:
        assert self.k in _VALID_K, (
            f"k ({self.k}) must be one of {sorted(_VALID_K)}"
        )
        assert self.warm_user_min_positives >= 1, (
            f"warm_user_min_positives ({self.warm_user_min_positives}) must be >= 1"
        )
