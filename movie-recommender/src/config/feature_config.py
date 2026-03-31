"""Feature engineering configuration."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.base_config import BaseConfig

_VALID_GENOME_TAG_COUNTS: frozenset[int] = frozenset({20, 50, 100})
_VALID_MF_DIMS: frozenset[int] = frozenset({32, 64, 128})


@dataclass
class FeatureConfig(BaseConfig):
    # ── content dims (LOCKED from EDA) ───────────────────────────────────────
    genre_vector_dim: int = 18          # LOCKED — IMAX dropped, 18 genres remain
    n_genome_tags: int = 50             # top-50 tags by variance

    # ── embedding / window params ─────────────────────────────────────────────
    mf_embedding_dim: int = 64          # ALS/MF embedding size
    recent_activity_days: int = 90      # window for recent genre affinity
    activity_short_window: int = 30
    activity_medium_window: int = 90

    # ── feature toggles ───────────────────────────────────────────────────────
    use_genome_features: bool = True
    use_time_features: bool = True

    def validate(self) -> None:
        assert self.genre_vector_dim == 18, (
            f"genre_vector_dim must be 18 (LOCKED from EDA), got {self.genre_vector_dim}"
        )
        assert self.n_genome_tags in _VALID_GENOME_TAG_COUNTS, (
            f"n_genome_tags ({self.n_genome_tags}) must be one of {sorted(_VALID_GENOME_TAG_COUNTS)}"
        )
        assert self.mf_embedding_dim in _VALID_MF_DIMS, (
            f"mf_embedding_dim ({self.mf_embedding_dim}) must be one of {sorted(_VALID_MF_DIMS)}"
        )
