"""ALS-based candidate generator using the implicit library + FAISS.

OMP_NUM_THREADS is set to 1 before importing faiss to prevent threading
conflicts on Windows.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("OMP_NUM_THREADS", "1")
import faiss  # noqa: E402  (must come after env var)
import implicit.als  # noqa: E402

from src.candidates.base_candidate_generator import BaseCandidateGenerator
from src.config.feature_config import FeatureConfig
from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

_PROCESSED_DIR: Path = Path("data/processed")


class ALSCandidateGenerator(BaseCandidateGenerator):
    """ALS matrix factorization + FAISS nearest-neighbour retrieval.

    Call ``fit(train_df)`` before ``generate()``.

    Artefacts written to ``data/processed/``:
      - ``faiss_item_index.bin``
      - ``als_user_factors.npy``
      - ``als_item_factors.npy``
      - ``als_movie_id_map.npy``
    """

    def __init__(
        self,
        config: TrainingConfig,
        feature_config: FeatureConfig,
    ) -> None:
        self._config = config
        self._feature_config = feature_config
        self._is_fitted: bool = False

        # Set after fit()
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._movie_id_map: np.ndarray | None = None       # idx → movie_id
        self._movie_id_to_idx: dict[int, int] = {}
        self._user_id_to_idx: dict[int, int] = {}
        self._faiss_index: faiss.IndexFlatIP | None = None

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, save_dir: Path | None = None) -> None:
        """Train ALS, build FAISS index, persist artefacts.

        Parameters
        ----------
        train_df:
            Training split — requires ``userId``, ``movieId``, ``rating``.
        save_dir:
            Directory to write artefacts.  Defaults to ``data/processed/``.
            Pass ``data/sample/`` when running in --sample mode.
        """
        out_dir = Path(save_dir) if save_dir is not None else _PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build compact user/item indices
        unique_users = sorted(train_df["userId"].unique().tolist())
        unique_movies = sorted(train_df["movieId"].unique().tolist())
        self._user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        self._movie_id_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
        self._movie_id_map = np.array(unique_movies, dtype=np.int32)

        n_users = len(unique_users)
        n_items = len(unique_movies)

        # Build user-item sparse matrix — implicit ≥0.6 expects (users × items)
        uid_idx = train_df["userId"].map(self._user_id_to_idx).values
        mid_idx = train_df["movieId"].map(self._movie_id_to_idx).values
        ratings = train_df["rating"].values.astype(np.float32)

        user_item_matrix = sp.csr_matrix(
            (ratings, (uid_idx, mid_idx)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        # Train ALS
        model = implicit.als.AlternatingLeastSquares(
            factors=self._config.als_factors,
            iterations=self._config.als_iterations,
            regularization=self._config.als_regularization,
            random_state=42,
        )
        logger.info(
            "ALSCandidateGenerator: training ALS (factors=%d, iters=%d) …",
            self._config.als_factors,
            self._config.als_iterations,
        )
        model.fit(user_item_matrix)

        self._user_factors = model.user_factors   # shape: (n_users, factors)
        self._item_factors = model.item_factors   # shape: (n_items, factors)

        # ── Build FAISS IndexFlatIP ───────────────────────────────────────
        # Normalise item vectors so inner product == cosine similarity
        norms = np.linalg.norm(self._item_factors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        item_norm = (self._item_factors / norms).astype(np.float32)

        dim = item_norm.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(item_norm)
        self._faiss_index = index

        # ── Persist artefacts ─────────────────────────────────────────────
        faiss.write_index(index, str(out_dir / "faiss_item_index.bin"))
        np.save(out_dir / "als_user_factors.npy", self._user_factors)
        np.save(out_dir / "als_item_factors.npy", self._item_factors)
        np.save(out_dir / "als_movie_id_map.npy", self._movie_id_map)
        # Save user ID map so train.py can reconstruct user_id_to_idx exactly
        # (must match the order used during fit — do NOT reconstruct from user_features.parquet)
        user_id_map = np.array(unique_users, dtype=np.int32)
        np.save(out_dir / "als_user_id_map.npy", user_id_map)

        self._is_fitted = True
        logger.info(
            "ALSCandidateGenerator: fit complete — %d users, %d items, dim=%d.",
            n_users,
            n_items,
            dim,
        )

    # ── generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        user_id: int,
        user_features: dict,
        n: int = 100,
        rated_movie_ids: set[int] | None = None,
    ) -> list[int]:
        """Return top-*n* ALS-based candidates using FAISS nearest-neighbour search.

        Returns an empty list if ``user_id`` was not seen during training.
        """
        if not self._is_fitted or self._faiss_index is None:
            logger.warning("ALSCandidateGenerator.generate called before fit().")
            return []

        rated = rated_movie_ids or set()
        user_idx = self._user_id_to_idx.get(user_id)
        if user_idx is None:
            logger.debug("ALSCandidateGenerator: unknown user %d.", user_id)
            return []

        # Normalise user vector (same space as normalised item vectors)
        u_vec = self._user_factors[user_idx].reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(u_vec)
        if norm > 0:
            u_vec = u_vec / norm

        # Search FAISS for top-(n*2) items (over-fetch to cover rated filtering)
        k = min(n * 2, self._faiss_index.ntotal)
        _, indices = self._faiss_index.search(u_vec, k)

        candidates: list[int] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._movie_id_map):  # type: ignore[arg-type]
                continue
            mid = int(self._movie_id_map[idx])  # type: ignore[index]
            if mid not in rated:
                candidates.append(mid)
            if len(candidates) >= n:
                break

        return candidates[:n]

    # ── mf scores ─────────────────────────────────────────────────────────

    def get_mf_scores(
        self,
        user_id: int,
        movie_ids: list[int],
    ) -> dict[int, float]:
        """Dot product score for each (user_id, movie_id) pair.

        Returns ``0.0`` for unknown users or unknown items.
        """
        if not self._is_fitted:
            return {mid: 0.0 for mid in movie_ids}

        user_idx = self._user_id_to_idx.get(user_id)
        if user_idx is None:
            return {mid: 0.0 for mid in movie_ids}

        u_vec = self._user_factors[user_idx]  # (factors,)
        scores: dict[int, float] = {}
        for mid in movie_ids:
            item_idx = self._movie_id_to_idx.get(mid)
            if item_idx is None:
                scores[mid] = 0.0
            else:
                scores[mid] = float(np.dot(u_vec, self._item_factors[item_idx]))
        return scores
