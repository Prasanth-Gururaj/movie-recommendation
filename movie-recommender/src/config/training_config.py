"""Training pipeline configuration: ALS, FAISS, candidate pools."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.base_config import BaseConfig

_VALID_ALS_FACTORS: frozenset[int] = frozenset({32, 64, 128})


@dataclass
class TrainingConfig(BaseConfig):
    # ── MLflow experiment bookkeeping ─────────────────────────────────────────
    experiment_name: str = "movie_recommender"
    run_name: str = "baseline"
    mlflow_tracking_uri: str = "mlruns"

    # ── recommender type ──────────────────────────────────────────────────────
    # options: "popularity", "genre_pop", "cf", "als", "two_stage"
    recommender_type: str = "two_stage"

    # ── candidate generation ──────────────────────────────────────────────────
    candidate_pool_size: int = 300

    # ── ALS (Stage 1 retrieval) ───────────────────────────────────────────────
    als_factors: int = 64
    als_iterations: int = 20
    als_regularization: float = 0.01

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_index_type: str = "IndexFlatIP"

    # ── per-source candidate caps ─────────────────────────────────────────────
    n_candidates_pop: int = 100
    n_candidates_cf: int = 100
    n_candidates_mf: int = 100

    def validate(self) -> None:
        assert 100 <= self.candidate_pool_size <= 500, (
            f"candidate_pool_size ({self.candidate_pool_size}) must be between 100 and 500"
        )
        assert self.als_factors in _VALID_ALS_FACTORS, (
            f"als_factors ({self.als_factors}) must be one of {sorted(_VALID_ALS_FACTORS)}"
        )
        assert self.als_iterations > 0, (
            f"als_iterations ({self.als_iterations}) must be > 0"
        )
        total_candidates = (
            self.n_candidates_pop + self.n_candidates_cf + self.n_candidates_mf
        )
        assert total_candidates >= self.candidate_pool_size, (
            f"n_candidates_pop + n_candidates_cf + n_candidates_mf "
            f"({total_candidates}) must be >= candidate_pool_size ({self.candidate_pool_size})"
        )
