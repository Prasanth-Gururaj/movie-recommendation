"""FeatureStore — orchestrates all feature builders and assembles the training matrix.

build_all_features()          → builds train/val/test parquets + user/item parquets
                                + injects mf_scores if als_gen provided
                                + saves feature_columns.json LAST
assemble_inference_features() → reconstructs feature row(s) from dicts at serving time
get_negative_samples()        → negative sampling for training
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
from src.ingestion.splitter import get_warm_items

logger = logging.getLogger(__name__)

_CONFIGS_DIR: Path = Path(__file__).parents[2] / "configs"
_FEATURE_COLUMNS_PATH: Path = _CONFIGS_DIR / "feature_columns.json"


class FeatureStore:
    """Central orchestrator for feature construction and serving.

    Parameters
    ----------
    config:
        Master experiment config.
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
        als_gen=None,
    ) -> None:
        """Build all feature tables and save to processed_data_dir.

        Steps
        -----
        1.  User features (from train)
        2.  Item features (from train)
        3.  Interaction features (train pairs)
        4.  Time features (train pairs)
        5.  Determine feature_columns order
        6.  Write train_features.parquet (chunked, ~20 M rows)
        7.  Write val_features.parquet   (same schema)
        8.  Write test_features.parquet  (same schema)
        9.  Save user_features.parquet + item_features.parquet
        10. Save feature_columns.json (LAST — only after all three verified)

        Parameters
        ----------
        als_gen:
            Optional fitted ALSCandidateGenerator.  When provided the
            ``mf_score`` column is populated with real ALS dot-product
            scores instead of the 0.0 placeholder.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_dir = Path(self._config.data.processed_data_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        threshold = self._config.data.relevance_threshold

        # Ensure is_positive exists on all splits
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "is_positive" not in df.columns:
                df["is_positive"] = (df["rating"] >= threshold).astype("int8")

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

        # Use actual train ratings only — no synthetic negatives
        # Ratings >= threshold = positive, ratings < threshold = negative
        train_extended = train_df.copy()
        logger.info(
            "Using actual train ratings only: %d rows (no synthetic negatives).",
            len(train_extended),
        )

        # ── 3. interaction features (train) ───────────────────────────────
        # train_df (actual ratings only) is used for user-profile computation;
        # train_extended (actual + synthetic negatives) defines which pairs get rows.
        logger.info("Building interaction features (train) …")
        interaction_builder = InteractionFeatureBuilder(self._config)
        interaction_df = interaction_builder.build(data, train_df, pairs_df=train_extended)

        # ── 4. time features (train) ──────────────────────────────────────
        logger.info("Building time features (train) …")
        time_builder = TimeFeatureBuilder(self._config)
        time_df = time_builder.build(data, train_df, pairs_df=train_extended)

        # ── 5. feature_columns ────────────────────────────────────────────
        u_feat_names = [c for c in user_builder.get_feature_names() if c in user_features_df.columns]
        i_feat_names = [c for c in item_builder.get_feature_names() if c in item_features_df.columns]
        intr_feat_names = [c for c in interaction_builder.get_feature_names() if c in interaction_df.columns]
        time_feat_names = [c for c in time_builder.get_feature_names() if c in time_df.columns]

        self.feature_columns = list(
            dict.fromkeys(u_feat_names + i_feat_names + intr_feat_names + time_feat_names)
        )

        # Build shared index maps and pre-loaded feature matrices (all small)
        uid_to_idx = {int(u): i for i, u in enumerate(user_features_df["userId"].values)}
        mid_to_idx = {int(m): i for i, m in enumerate(item_features_df["movieId"].values)}
        u_mat = user_features_df[u_feat_names].values.astype(np.float32)
        i_mat = item_features_df[i_feat_names].values.astype(np.float32)

        # ── 6. train_features.parquet (chunked for 20 M rows) ─────────────
        logger.info("Assembling train_features.parquet …")
        interaction_df = interaction_df.sort_values("userId").reset_index(drop=True)
        _write_feature_parquet(
            split_df=train_extended,
            interaction_df=interaction_df,
            time_df=time_df,
            u_feat_names=u_feat_names,
            i_feat_names=i_feat_names,
            intr_feat_names=intr_feat_names,
            time_feat_names=time_feat_names,
            feature_columns=self.feature_columns,
            uid_to_idx=uid_to_idx,
            mid_to_idx=mid_to_idx,
            u_mat=u_mat,
            i_mat=i_mat,
            als_gen=als_gen,
            output_path=out_dir / "train_features.parquet",
        )

        # ── 7. val_features.parquet ───────────────────────────────────────
        logger.info("Building interaction features (val) …")
        val_interaction_df = interaction_builder.build(data, train_df, pairs_df=val_df)
        val_time_df = time_builder.build(data, train_df, pairs_df=val_df)

        logger.info("Assembling val_features.parquet …")
        _write_feature_parquet(
            split_df=val_df,
            interaction_df=val_interaction_df,
            time_df=val_time_df,
            u_feat_names=u_feat_names,
            i_feat_names=i_feat_names,
            intr_feat_names=intr_feat_names,
            time_feat_names=time_feat_names,
            feature_columns=self.feature_columns,
            uid_to_idx=uid_to_idx,
            mid_to_idx=mid_to_idx,
            u_mat=u_mat,
            i_mat=i_mat,
            als_gen=als_gen,
            output_path=out_dir / "val_features.parquet",
        )

        # ── 8. test_features.parquet ──────────────────────────────────────
        logger.info("Building interaction features (test) …")
        test_interaction_df = interaction_builder.build(data, train_df, pairs_df=test_df)
        test_time_df = time_builder.build(data, train_df, pairs_df=test_df)

        logger.info("Assembling test_features.parquet …")
        _write_feature_parquet(
            split_df=test_df,
            interaction_df=test_interaction_df,
            time_df=test_time_df,
            u_feat_names=u_feat_names,
            i_feat_names=i_feat_names,
            intr_feat_names=intr_feat_names,
            time_feat_names=time_feat_names,
            feature_columns=self.feature_columns,
            uid_to_idx=uid_to_idx,
            mid_to_idx=mid_to_idx,
            u_mat=u_mat,
            i_mat=i_mat,
            als_gen=als_gen,
            output_path=out_dir / "test_features.parquet",
        )

        # ── 9. save user / item parquet ───────────────────────────────────
        user_features_df.to_parquet(
            out_dir / "user_features.parquet", engine="pyarrow", index=False
        )
        item_features_df.to_parquet(
            out_dir / "item_features.parquet", engine="pyarrow", index=False
        )
        logger.info("User/item feature parquet files saved to %s.", out_dir)

        # ── 10. feature_columns.json — written LAST ───────────────────────
        with _FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.feature_columns, f, indent=2)
        logger.info("Saved feature_columns.json with %d columns.", len(self.feature_columns))

    # ── inference assembly ─────────────────────────────────────────────────

    def assemble_inference_features(
        self,
        user_features: dict,
        item_features_list: list[dict],
        request_context: dict,
    ) -> pd.DataFrame:
        """Assemble a feature DataFrame for a batch of candidate items."""
        rows = []
        for item_feat in item_features_list:
            row = {**user_features, **item_feat, **request_context}
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.feature_columns).astype(float)

        df = pd.DataFrame(rows)
        df = df.reindex(columns=self.feature_columns, fill_value=0.0)
        df = df.fillna(0.0)
        return df

    # ── negative sampling ──────────────────────────────────────────────────

    def get_negative_samples(
        self,
        train_df: pd.DataFrame,
        warm_items: set[int],
        all_items: set[int] | None = None,
        ratio: int = 4,
    ) -> pd.DataFrame:
        """Sample implicit negatives for each positive training pair.

        Stratified sampling: half the budget from warm items (≥cold_item_threshold
        ratings), the other half from cold/obscure items not in warm_items.
        This prevents the model from learning "unrated cold item = positive"
        when cold items appear only as positives in the actual rating data.

        Parameters
        ----------
        warm_items:
            Items with enough ratings to be considered "warm" (used for first
            half of negative budget).
        all_items:
            Full item catalog. When provided, samples from all catalog items
            (warm and cold) instead of warm_items only, so the model sees
            cold/obscure items as negatives. When None, falls back to
            warm_items only (legacy behaviour).
        ratio:
            Exact number of synthetic negatives to generate per positive.
            Total synthetic rows = n_positives × ratio.
        """
        threshold = self._config.data.relevance_threshold
        seed = self._config.data.random_seed
        rng = np.random.default_rng(seed)

        # Pool = all catalog items when provided, else warm items only (legacy)
        pool_set = all_items if all_items else warm_items
        pool_arr = np.array(sorted(pool_set), dtype=np.int32)
        n_pool = len(pool_arr)

        _EMPTY = pd.DataFrame(
            columns=["userId", "movieId", "rating", "timestamp", "is_positive"]
        )

        if n_pool == 0:
            return _EMPTY

        positives = train_df[train_df["rating"] >= threshold]
        if len(positives) == 0:
            return _EMPTY

        pos_counts = positives.groupby("userId").size()
        user_ts = positives.groupby("userId")["timestamp"].median().astype("int64")

        user_rated: dict[int, frozenset] = (
            train_df.groupby("userId")["movieId"]
            .apply(frozenset)
            .to_dict()
        )

        uid_chunks: list[np.ndarray] = []
        mid_chunks: list[np.ndarray] = []
        ts_chunks: list[np.ndarray] = []

        for uid, n_pos in pos_counts.items():
            rated_fs = user_rated.get(int(uid), frozenset())
            ts_val = int(user_ts.get(uid, 0))

            n_target = min(ratio * int(n_pos), n_pool)
            if n_target <= 0:
                continue
            n_oversample = min(n_target + len(rated_fs) + 10, n_pool)
            cands = rng.choice(pool_arr, size=n_oversample, replace=False)
            keep = np.array([m not in rated_fs for m in cands], dtype=bool)
            combined = cands[keep][:n_target].astype(np.int32)

            n = len(combined)
            if n == 0:
                continue

            uid_chunks.append(np.full(n, uid, dtype=np.int32))
            mid_chunks.append(combined)
            ts_chunks.append(np.full(n, ts_val, dtype=np.int64))

        if not uid_chunks:
            return _EMPTY

        all_uids = np.concatenate(uid_chunks)
        all_mids = np.concatenate(mid_chunks)
        all_ts = np.concatenate(ts_chunks)
        n_total = len(all_uids)

        return pd.DataFrame({
            "userId": all_uids,
            "movieId": all_mids,
            "rating": np.zeros(n_total, dtype=np.float32),
            "timestamp": all_ts,
            "is_positive": np.zeros(n_total, dtype=np.int8),
        })


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _inject_mf_scores_batch(chunk: pd.DataFrame, als_gen) -> pd.DataFrame:
    """Inject real ALS dot-product scores into the ``mf_score`` column."""
    u_idx = chunk["userId"].map(als_gen._user_id_to_idx).fillna(-1).astype(int).values
    m_idx = chunk["movieId"].map(als_gen._movie_id_to_idx).fillna(-1).astype(int).values
    valid = (u_idx >= 0) & (m_idx >= 0)
    mf_scores = np.zeros(len(chunk), dtype=np.float32)
    if valid.any():
        mf_scores[valid] = np.einsum(
            "ij,ij->i",
            als_gen._user_factors[u_idx[valid]],
            als_gen._item_factors[m_idx[valid]],
        )
    chunk = chunk.copy()
    chunk["mf_score"] = mf_scores
    return chunk


def _write_feature_parquet(
    split_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    time_df: pd.DataFrame,
    u_feat_names: list[str],
    i_feat_names: list[str],
    intr_feat_names: list[str],
    time_feat_names: list[str],
    feature_columns: list[str],
    uid_to_idx: dict[int, int],
    mid_to_idx: dict[int, int],
    u_mat: np.ndarray,
    i_mat: np.ndarray,
    als_gen,
    output_path: Path,
    chunk_size: int = 500_000,
) -> None:
    """Assemble and write a feature parquet for one split (train/val/test).

    Works the same way for all three splits.  For train (20 M rows) the
    chunked write keeps peak RAM bounded to ~1 GB.  For val/test (~1–2 M
    rows) it completes in a few chunks.

    Column order in the output is always:
        userId | movieId | is_positive | <feature_columns in order>
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    inject_mf = (
        als_gen is not None
        and getattr(als_gen, "_is_fitted", False)
        and "mf_score" in feature_columns
    )

    # Merge interaction + time to form base (only the narrow interaction cols)
    base = (
        interaction_df[["userId", "movieId"] + intr_feat_names]
        .merge(
            time_df[["userId", "movieId"] + time_feat_names],
            on=["userId", "movieId"],
            how="left",
        )
    )

    # Add is_positive from split_df
    pos_lookup = (
        split_df[["userId", "movieId", "is_positive"]]
        .drop_duplicates(["userId", "movieId"])
        .set_index(["userId", "movieId"])["is_positive"]
    )
    base_idx = pd.MultiIndex.from_arrays([base["userId"], base["movieId"]])
    base["is_positive"] = pos_lookup.reindex(base_idx).fillna(0).astype("int8").values

    base_uids = base["userId"].values
    base_mids = base["movieId"].values
    n_base = len(base)

    u_idx = np.fromiter(
        (uid_to_idx.get(int(u), -1) for u in base_uids), dtype=np.int32, count=n_base
    )
    m_idx = np.fromiter(
        (mid_to_idx.get(int(m), -1) for m in base_mids), dtype=np.int32, count=n_base
    )

    pq_writer: pq.ParquetWriter | None = None
    try:
        for s in range(0, n_base, chunk_size):
            e = min(s + chunk_size, n_base)
            chu: dict = {}
            chu["userId"] = base_uids[s:e]
            chu["movieId"] = base_mids[s:e]
            chu["is_positive"] = base["is_positive"].values[s:e]

            u_sel = u_idx[s:e]
            m_sel = m_idx[s:e]
            valid_u = u_sel >= 0
            valid_m = m_sel >= 0

            for j, col in enumerate(u_feat_names):
                vals = np.zeros(e - s, dtype=np.float32)
                vals[valid_u] = u_mat[u_sel[valid_u], j]
                chu[col] = vals

            for j, col in enumerate(i_feat_names):
                vals = np.zeros(e - s, dtype=np.float32)
                vals[valid_m] = i_mat[m_sel[valid_m], j]
                chu[col] = vals

            for col in intr_feat_names + time_feat_names:
                chu[col] = base[col].values[s:e]

            chunk_df = pd.DataFrame(chu)

            if inject_mf:
                chunk_df = _inject_mf_scores_batch(chunk_df, als_gen)

            # Ensure final column order matches feature_columns exactly
            out_cols = list(dict.fromkeys(
                ["userId", "movieId", "is_positive"]
                + [c for c in feature_columns if c in chunk_df.columns]
            ))
            # Fill any feature columns absent from this chunk with 0.0
            for missing in set(feature_columns) - set(chunk_df.columns):
                chunk_df[missing] = np.float32(0.0)
            chunk_df = chunk_df[out_cols]

            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if pq_writer is None:
                pq_writer = pq.ParquetWriter(str(output_path), table.schema)
            pq_writer.write_table(table)
            logger.info("  %s: wrote rows %d–%d / %d", output_path.name, s, e, n_base)
            del chunk_df, table
    finally:
        if pq_writer:
            pq_writer.close()

    logger.info(
        "_write_feature_parquet: %s written (%d rows, %d feature cols%s).",
        output_path.name,
        n_base,
        len(feature_columns),
        " + mf_scores injected" if inject_mf else "",
    )
