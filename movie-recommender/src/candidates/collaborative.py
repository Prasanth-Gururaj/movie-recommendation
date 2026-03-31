"""Item-item collaborative filtering candidate generator.

Builds a cosine similarity matrix from the user-item interaction matrix
using only warm items.  Uses ``scipy.sparse`` throughout for memory
efficiency.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from src.candidates.base_candidate_generator import BaseCandidateGenerator
from src.config.data_config import DataConfig
from src.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

_RELEVANCE_THRESHOLD: float = 4.0


class CFCandidateGenerator(BaseCandidateGenerator):
    """Item-item CF generator using cosine similarity on the interaction matrix.

    Only warm items (rating_count >= cold_item_threshold) are included so
    the similarity matrix stays tractable.

    Parameters
    ----------
    train_df:
        Full training split (userId, movieId, rating, …).
    config:
        TrainingConfig — supplies ``als_factors`` etc. (not used here, but
        kept for API consistency with other generators).
    data_config:
        DataConfig — supplies ``cold_item_threshold`` and
        ``relevance_threshold``.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        config: TrainingConfig,
        data_config: DataConfig | None = None,
    ) -> None:
        self._config = config
        cold_thresh = data_config.cold_item_threshold if data_config else 10
        threshold = data_config.relevance_threshold if data_config else _RELEVANCE_THRESHOLD

        self._threshold = threshold

        # Identify warm items
        item_counts = train_df.groupby("movieId")["rating"].count()
        warm_item_ids = item_counts[item_counts >= cold_thresh].index
        warm_set = set(warm_item_ids.tolist())

        train_warm = train_df[train_df["movieId"].isin(warm_set)].copy()

        if len(train_warm) == 0:
            logger.warning("CFCandidateGenerator: no warm items found — CF disabled.")
            self._item_sim: sp.csr_matrix | None = None
            self._item_idx: dict[int, int] = {}
            self._idx_item: list[int] = []
            self._user_positives: dict[int, list[int]] = {}
            return

        # Build compact index for warm items and all users
        unique_items = sorted(train_warm["movieId"].unique().tolist())
        unique_users = sorted(train_warm["userId"].unique().tolist())
        self._item_idx: dict[int, int] = {mid: i for i, mid in enumerate(unique_items)}
        self._idx_item: list[int] = unique_items
        user_idx: dict[int, int] = {uid: i for i, uid in enumerate(unique_users)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build user-item sparse matrix (binary implicit feedback)
        rows, cols, data = [], [], []
        for _, row in train_warm.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            if uid in user_idx and mid in self._item_idx:
                rows.append(user_idx[uid])
                cols.append(self._item_idx[mid])
                data.append(1.0)

        ui_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_items), dtype=np.float32
        )

        # Item-item cosine similarity: normalize each item column then multiply
        # item_matrix (items × users)  →  normalise rows → dot
        item_matrix = ui_matrix.T  # shape: (n_items, n_users)
        item_norm = normalize(item_matrix, norm="l2", axis=1)
        self._item_sim = (item_norm @ item_norm.T).tocsr()  # (n_items, n_items)

        # Pre-compute per-user positive item lists for fast lookup
        self._user_positives: dict[int, list[int]] = {}
        pos_df = train_warm[train_warm["rating"] >= threshold]
        for uid, grp in pos_df.groupby("userId"):
            self._user_positives[int(uid)] = [
                mid for mid in grp["movieId"].tolist() if mid in self._item_idx
            ]

        logger.info(
            "CFCandidateGenerator: %d warm items, %d users with positives.",
            n_items,
            len(self._user_positives),
        )

    # ── generate ──────────────────────────────────────────────────────────

    def generate(
        self,
        user_id: int,
        user_features: dict,
        n: int = 100,
        rated_movie_ids: set[int] | None = None,
    ) -> list[int]:
        """Return up to *n* CF-based candidates.

        Returns an empty list for cold users (no positives in training data).
        """
        rated = rated_movie_ids or set()

        if self._item_sim is None:
            return []

        pos_items = self._user_positives.get(user_id, [])
        if not pos_items:
            logger.debug("CFCandidateGenerator: user %d has no positives — cold.", user_id)
            return []

        # Accumulate similarity scores from all user positive items
        n_items = len(self._idx_item)
        scores = np.zeros(n_items, dtype=np.float32)

        for mid in pos_items:
            idx = self._item_idx.get(mid)
            if idx is None:
                continue
            sim_row = self._item_sim[idx]  # sparse row
            scores += np.asarray(sim_row.todense()).ravel()

        # Zero-out rated items
        for mid in rated:
            idx = self._item_idx.get(mid)
            if idx is not None:
                scores[idx] = 0.0

        # Top-n by aggregated score
        top_indices = np.argsort(scores)[::-1][:n]
        candidates = [
            self._idx_item[i]
            for i in top_indices
            if scores[i] > 0 and self._idx_item[i] not in rated
        ]

        return candidates[:n]
