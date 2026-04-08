"""User-item interaction feature builder.

Requires ``data["user_features"]`` and ``data["item_features"]`` to be present
(populated by FeatureStore before this builder is called).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.experiment_config import ExperimentConfig
from src.features.base_feature_builder import BaseFeatureBuilder

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).parents[2] / "configs"
_GENRE_COLUMNS_PATH: Path = _CONFIGS_DIR / "genre_columns.json"


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two same-shape 2-D arrays."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    denom = (norm_a * norm_b).ravel()
    dot = (a * b).sum(axis=1)
    return np.where(denom > 0, dot / denom, 0.0).astype(float)


class InteractionFeatureBuilder(BaseFeatureBuilder):
    """Builds one-row-per-(userId, movieId) interaction feature table.

    Features
    --------
    genre_overlap_score  — dot(user_affinity, item_genre) / (sum(user_affinity) + ε), clipped to [0,1]
    tag_profile_similarity — cosine(user_tag_profile, item_genome_vector)
    rating_gap           — user mean_rating − item avg_rating  (can be negative)
    genre_history_count  — log1p(# user positives sharing item's primary genre)
    mf_score             — placeholder 0.0 (filled after ALS training)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        with _GENRE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
            self._genre_columns: list[str] = json.load(f)

    # ── build ──────────────────────────────────────────────────────────────

    def build(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
        *,
        pairs_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build interaction features without merging wide vectors onto the full
        pairs DataFrame.  All feature vectors are looked up from pre-built
        numpy arrays via integer indices and computed in 200 k-row chunks.

        Parameters
        ----------
        pairs_df:
            If provided, compute features for these (userId, movieId) pairs
            instead of extracting pairs from ``train_df``.  ``train_df`` is
            still used to compute user profiles (tag_profile_similarity,
            genre_history_count) — it must not be ``None``.
        """
        from scipy.sparse import csr_matrix as _csr

        user_feat: pd.DataFrame = data["user_features"]
        item_feat: pd.DataFrame = data["item_features"]
        threshold: float = self._config.data.relevance_threshold

        self._train_max_ts = int(train_df["timestamp"].max())

        # ── unique pairs ─────────────────────────────────────────────────
        _source = pairs_df if pairs_df is not None else train_df
        pairs = (
            _source[["userId", "movieId"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .copy()
        )
        n_pairs = len(pairs)
        CHUNK = 200_000

        genre_cols = self._genre_columns
        ua_cols = [f"genre_affinity_{g}" for g in genre_cols]
        ig_cols = [f"genre_{g}" for g in genre_cols]
        avail_ua = [c for c in ua_cols if c in user_feat.columns]
        avail_ig = [c for c in ig_cols if c in item_feat.columns]

        # ── global index maps (built once, O(n_users + n_items)) ─────────
        user_ids_arr = user_feat["userId"].values
        item_ids_arr = item_feat["movieId"].values
        n_users = len(user_ids_arr)
        n_items = len(item_ids_arr)

        uid_to_idx: dict[int, int] = {int(u): i for i, u in enumerate(user_ids_arr)}
        mid_to_idx: dict[int, int] = {int(m): i for i, m in enumerate(item_ids_arr)}

        pair_uids = pairs["userId"].values
        pair_mids = pairs["movieId"].values
        pair_u_idx = np.fromiter(
            (uid_to_idx.get(int(u), -1) for u in pair_uids), dtype=np.int32, count=n_pairs
        )
        pair_m_idx = np.fromiter(
            (mid_to_idx.get(int(m), -1) for m in pair_mids), dtype=np.int32, count=n_pairs
        )
        valid_pairs = (pair_u_idx >= 0) & (pair_m_idx >= 0)

        # ── genre_overlap_score ───────────────────────────────────────────
        # Look up 18-dim vectors from pre-built arrays; no merge onto pairs.
        pairs["genre_overlap_score"] = 0.0
        if avail_ua and avail_ig:
            ua_full = user_feat[avail_ua].values.astype(np.float32)   # (n_users, 18)
            ig_full = item_feat[avail_ig].values.astype(np.float32)   # (n_items, 18)
            result = np.zeros(n_pairs, dtype=np.float32)
            for s in range(0, n_pairs, CHUNK):
                e = min(s + CHUNK, n_pairs)
                mask = valid_pairs[s:e]
                if not mask.any():
                    continue
                u_v = np.zeros((e - s, len(avail_ua)), dtype=np.float32)
                i_v = np.zeros((e - s, len(avail_ig)), dtype=np.float32)
                u_v[mask] = ua_full[pair_u_idx[s:e][mask]]
                i_v[mask] = ig_full[pair_m_idx[s:e][mask]]
                dot = (u_v * i_v).sum(axis=1)
                ua_sum = u_v.sum(axis=1)
                result[s:e] = np.where(ua_sum > 0, dot / (ua_sum + 1e-8), 0.0).clip(0, 1)
            pairs["genre_overlap_score"] = result

        # ── rating_gap ────────────────────────────────────────────────────
        pairs["rating_gap"] = 0.0
        if "mean_rating" in user_feat.columns and "avg_rating" in item_feat.columns:
            u_mr = user_feat["mean_rating"].values.astype(np.float32)
            i_ar = item_feat["avg_rating"].values.astype(np.float32)
            result = np.zeros(n_pairs, dtype=np.float32)
            result[valid_pairs] = (
                u_mr[pair_u_idx[valid_pairs]] - i_ar[pair_m_idx[valid_pairs]]
            )
            pairs["rating_gap"] = result

        # ── tag_profile_similarity ────────────────────────────────────────
        # Step A: compute user tag profiles via sparse multiply (no 20M-row merge).
        # Step B: look up profiles and item genomes by integer index in chunks.
        genome_cols = [c for c in item_feat.columns if c.startswith("genome_tag_")]
        pairs["tag_profile_similarity"] = 0.0
        if genome_cols:
            positives = train_df[train_df["rating"] >= threshold][["userId", "movieId"]]
            if len(positives) > 0:
                pos_uid_arr = positives["userId"].values
                pos_mid_arr = positives["movieId"].values
                pos_u_idx = np.fromiter(
                    (uid_to_idx.get(int(u), -1) for u in pos_uid_arr),
                    dtype=np.int32, count=len(positives),
                )
                pos_m_idx = np.fromiter(
                    (mid_to_idx.get(int(m), -1) for m in pos_mid_arr),
                    dtype=np.int32, count=len(positives),
                )
                valid_pos = (pos_u_idx >= 0) & (pos_m_idx >= 0)

                # Sparse user-movie matrix → mean genome profile per user
                ones = np.ones(valid_pos.sum(), dtype=np.float32)
                A = _csr(
                    (ones, (pos_u_idx[valid_pos], pos_m_idx[valid_pos])),
                    shape=(n_users, n_items),
                )
                row_counts = np.array(A.sum(axis=1)).ravel()
                row_counts[row_counts == 0] = 1.0
                A = A.multiply(1.0 / row_counts[:, np.newaxis])

                # Item genome matrix — (n_items, n_genome), small
                i_genome_full = (
                    item_feat[genome_cols]
                    .values.astype(np.float32)
                )  # (n_items, n_genome) — ~7 MB for 37 K items × 50 tags

                # User profiles — (n_users, n_genome), ~28 MB
                u_profile_full = np.asarray(A @ i_genome_full, dtype=np.float32)

                # Chunked cosine similarity using array indexing
                n_genome = len(genome_cols)
                result = np.zeros(n_pairs, dtype=np.float32)
                for s in range(0, n_pairs, CHUNK):
                    e = min(s + CHUNK, n_pairs)
                    mask = valid_pairs[s:e]
                    if not mask.any():
                        continue
                    u_chunk = np.zeros((e - s, n_genome), dtype=np.float32)
                    i_chunk = np.zeros((e - s, n_genome), dtype=np.float32)
                    u_chunk[mask] = u_profile_full[pair_u_idx[s:e][mask]]
                    i_chunk[mask] = i_genome_full[pair_m_idx[s:e][mask]]
                    result[s:e] = _cosine_rows(u_chunk, i_chunk)
                pairs["tag_profile_similarity"] = result

        # ── genre_history_count ───────────────────────────────────────────
        # Sparse multiply to count positive-item genre hits per user,
        # then look up via array indexing — no merge onto pairs or positives.
        pairs["genre_history_count"] = 0.0
        if avail_ig:
            positives = train_df[train_df["rating"] >= threshold][["userId", "movieId"]]
            if len(positives) > 0:
                pos_uid_arr = positives["userId"].values
                pos_mid_arr = positives["movieId"].values
                pos_u_idx2 = np.fromiter(
                    (uid_to_idx.get(int(u), -1) for u in pos_uid_arr),
                    dtype=np.int32, count=len(positives),
                )
                pos_m_idx2 = np.fromiter(
                    (mid_to_idx.get(int(m), -1) for m in pos_mid_arr),
                    dtype=np.int32, count=len(positives),
                )
                valid_pos2 = (pos_u_idx2 >= 0) & (pos_m_idx2 >= 0)

                # Item genre matrix (n_items, n_genres) — ~2.5 MB
                ig_full = item_feat[avail_ig].values.astype(np.float32)

                # Sparse multiply: ugc_mat[u, g] = # positive items user u
                # has rated that belong to genre g
                ones2 = np.ones(valid_pos2.sum(), dtype=np.float32)
                B = _csr(
                    (ones2, (pos_u_idx2[valid_pos2], pos_m_idx2[valid_pos2])),
                    shape=(n_users, n_items),
                )
                ugc_mat = np.asarray(B @ ig_full, dtype=np.float32)  # (n_users, n_genres)

                # Item primary genre index: first genre column == 1
                item_primary_g_idx = np.full(n_items, -1, dtype=np.int32)
                for j in range(len(avail_ig)):
                    unset = item_primary_g_idx < 0
                    item_primary_g_idx[unset & (ig_full[:, j] == 1)] = j

                # Vectorised pair-level lookup
                prim_g = item_primary_g_idx[
                    np.where(pair_m_idx >= 0, pair_m_idx, 0)
                ]
                prim_g[pair_m_idx < 0] = -1

                valid_ghc = valid_pairs & (prim_g >= 0)
                result = np.zeros(n_pairs, dtype=np.float32)
                result[valid_ghc] = ugc_mat[
                    pair_u_idx[valid_ghc], prim_g[valid_ghc]
                ]
                pairs["genre_history_count"] = np.log1p(result)

        # ── mf_score placeholder ──────────────────────────────────────────
        pairs["mf_score"] = 0.0

        self.log_feature_stats(pairs)
        return pairs

    # ── feature names ──────────────────────────────────────────────────────

    def get_feature_names(self) -> list[str]:
        return [
            "genre_overlap_score",
            "tag_profile_similarity",
            "rating_gap",
            "genre_history_count",
            "mf_score",
        ]
