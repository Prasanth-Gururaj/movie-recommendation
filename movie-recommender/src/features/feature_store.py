"""FeatureStore — orchestrates all feature builders and assembles the training matrix.

build_all_features()      → builds, assembles, saves parquet + feature_columns.json
assemble_inference_features() → reconstructs feature row(s) from dicts at serving time
get_negative_samples()    → negative sampling for training
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.experiment_config import ExperimentConfig
from src.features.interaction_features import InteractionFeatureBuilder
from src.features.item_features import ItemFeatureBuilder
from src.features.time_features import TimeFeatureBuilder
from src.features.user_features import UserFeatureBuilder

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).parents[2] / "configs"
_FEATURE_COLUMNS_PATH: Path = _CONFIGS_DIR / "feature_columns.json"


class FeatureStore:
    """Central orchestrator for feature construction and serving.

    Parameters
    ----------
    config:
        Master experiment config.  Sub-config fields used:
          - data.relevance_threshold, data.cold_user_threshold, …
          - feature.n_genome_tags, feature.activity_short_window, …
          - data.processed_data_dir  (where parquet files are saved)
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self.feature_columns: list[str] = []

        # Load existing feature columns if available (for inference-only instances)
        if _FEATURE_COLUMNS_PATH.exists():
            try:
                with _FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.feature_columns = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                pass

    # ── main build ─────────────────────────────────────────────────────────

    def build_all_features(
        self,
        data: dict[str, pd.DataFrame],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        """Build all feature tables from training data, assemble matrix, save.

        Steps
        -----
        1. User features (train only)
        2. Item features (train + movies)
        3. Interaction features (train pairs)
        4. Time features (train pairs)
        5. Assemble training matrix
        6. Save feature_columns.json  ← locked column order for inference
        7. Save parquet files to processed_data_dir
        """
        out_dir = Path(self._config.data.processed_data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Ensure is_positive exists
        threshold = self._config.data.relevance_threshold
        if "is_positive" not in train_df.columns:
            train_df = train_df.copy()
            train_df["is_positive"] = (train_df["rating"] >= threshold).astype("int8")

        # ── 1. user features ──────────────────────────────────────────────
        logger.info("Building user features …")
        user_builder = UserFeatureBuilder(self._config)
        user_features_df = user_builder.build(data, train_df)
        data = {**data, "user_features": user_features_df}

        # ── 2. item features ──────────────────────────────────────────────
        logger.info("Building item features …")
        item_builder = ItemFeatureBuilder(self._config)
        item_features_df = item_builder.build(data, train_df)
        data = {**data, "item_features": item_features_df}

        # ── 3. interaction features ───────────────────────────────────────
        logger.info("Building interaction features …")
        interaction_builder = InteractionFeatureBuilder(self._config)
        interaction_df = interaction_builder.build(data, train_df)

        # ── 4. time features ──────────────────────────────────────────────
        logger.info("Building time features …")
        time_builder = TimeFeatureBuilder(self._config)
        time_df = time_builder.build(data, train_df)

        # ── 5. assemble training matrix ───────────────────────────────────
        logger.info("Assembling training matrix …")
        u_feat_names = [c for c in user_builder.get_feature_names() if c in user_features_df.columns]
        i_feat_names = [c for c in item_builder.get_feature_names() if c in item_features_df.columns]
        intr_feat_names = [c for c in interaction_builder.get_feature_names() if c in interaction_df.columns]
        time_feat_names = [c for c in time_builder.get_feature_names() if c in time_df.columns]

        train_matrix = (
            train_df[["userId", "movieId", "is_positive"]]
            .drop_duplicates(subset=["userId", "movieId"])
            .reset_index(drop=True)
            .copy()
        )

        train_matrix = train_matrix.merge(
            user_features_df[["userId"] + u_feat_names], on="userId", how="left"
        )
        train_matrix = train_matrix.merge(
            item_features_df[["movieId"] + i_feat_names], on="movieId", how="left"
        )
        train_matrix = train_matrix.merge(
            interaction_df[["userId", "movieId"] + intr_feat_names],
            on=["userId", "movieId"],
            how="left",
        )
        train_matrix = train_matrix.merge(
            time_df[["userId", "movieId"] + time_feat_names],
            on=["userId", "movieId"],
            how="left",
        )

        # ── 6. feature_columns.json ───────────────────────────────────────
        self.feature_columns = u_feat_names + i_feat_names + intr_feat_names + time_feat_names
        with _FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info("Saved feature_columns.json with %d columns.", len(self.feature_columns))

        # ── 7. save parquet ───────────────────────────────────────────────
        train_matrix.to_parquet(
            out_dir / "train_features.parquet", engine="pyarrow", index=False
        )
        user_features_df.to_parquet(
            out_dir / "user_features.parquet", engine="pyarrow", index=False
        )
        item_features_df.to_parquet(
            out_dir / "item_features.parquet", engine="pyarrow", index=False
        )
        logger.info("Feature parquet files saved to %s.", out_dir)

    # ── inference assembly ─────────────────────────────────────────────────

    def assemble_inference_features(
        self,
        user_features: dict,
        item_features_list: list[dict],
        request_context: dict,
    ) -> pd.DataFrame:
        """Assemble a feature DataFrame for a batch of candidate items.

        Each row = one candidate item.  The output columns are guaranteed to
        match ``self.feature_columns`` (loaded from feature_columns.json).
        Missing features are filled with 0.0 — never NaN.

        Parameters
        ----------
        user_features:
            Flat dict of the requesting user's pre-computed features.
        item_features_list:
            One dict per candidate item.
        request_context:
            Additional context such as ``{"timestamp": <unix_ts>}``.
        """
        rows = []
        for item_feat in item_features_list:
            row = {**user_features, **item_feat, **request_context}
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.feature_columns).astype(float)

        df = pd.DataFrame(rows)
        df = df.reindex(columns=self.feature_columns, fill_value=0.0)

        # Ensure numeric types and no NaN
        df = df.fillna(0.0)
        return df

    # ── negative sampling ──────────────────────────────────────────────────

    def get_negative_samples(
        self,
        train_df: pd.DataFrame,
        warm_items: set[int],
        ratio: int = 4,
    ) -> pd.DataFrame:
        """Sample implicit negatives for each positive training pair.

        For each positive (user, item) in ``train_df``, sample ``ratio``
        unseen warm items the user has NOT rated at all in train.

        Parameters
        ----------
        train_df:
            Training pairs (must have ``rating`` and ``userId``, ``movieId``).
        warm_items:
            Set of item IDs eligible for negative sampling (warm catalog).
        ratio:
            Number of negatives per positive.

        Returns
        -------
        DataFrame with columns matching train_df schema plus ``is_positive=0``.
        """
        threshold = self._config.data.relevance_threshold
        seed = self._config.data.random_seed
        rng = np.random.default_rng(seed)

        warm_items_arr = np.array(sorted(warm_items), dtype=np.int32)

        # All positives from train
        positives = train_df[train_df["rating"] >= threshold][
            ["userId", "movieId", "timestamp"]
        ].copy()

        if len(positives) == 0:
            empty = pd.DataFrame(
                columns=["userId", "movieId", "rating", "timestamp", "is_positive"]
            )
            return empty

        # Pre-build per-user rated sets for O(1) lookup
        user_rated: dict[int, set[int]] = (
            train_df.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        neg_rows: list[dict] = []

        for _, pos_row in positives.iterrows():
            uid = int(pos_row["userId"])
            rated = user_rated.get(uid, set())

            candidates = warm_items_arr[
                ~np.isin(warm_items_arr, list(rated))
            ]
            if len(candidates) == 0:
                continue

            n_sample = min(ratio, len(candidates))
            sampled = rng.choice(candidates, size=n_sample, replace=False)

            for mid in sampled:
                neg_rows.append(
                    {
                        "userId": uid,
                        "movieId": int(mid),
                        "rating": 0.0,
                        "timestamp": int(pos_row["timestamp"]),
                        "is_positive": 0,
                    }
                )

        if not neg_rows:
            return pd.DataFrame(
                columns=["userId", "movieId", "rating", "timestamp", "is_positive"]
            )

        neg_df = pd.DataFrame(neg_rows)
        neg_df["userId"] = neg_df["userId"].astype("int32")
        neg_df["movieId"] = neg_df["movieId"].astype("int32")
        neg_df["rating"] = neg_df["rating"].astype("float32")
        neg_df["timestamp"] = neg_df["timestamp"].astype("int64")
        neg_df["is_positive"] = neg_df["is_positive"].astype("int8")
        return neg_df
