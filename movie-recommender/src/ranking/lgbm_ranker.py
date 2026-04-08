"""LightGBM learning-to-rank wrapper (lambdarank objective)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.ranking.base_ranker import BaseRanker

logger = logging.getLogger(__name__)


class LGBMRanker(BaseRanker):
    """Thin wrapper around ``lightgbm.LGBMRanker`` with ``lambdarank``.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth (-1 = unlimited).
    learning_rate:
        Step-shrinkage.
    num_leaves:
        Maximum number of leaves per tree.
    subsample / colsample_bytree:
        Row / column sub-sampling ratios.
    random_state:
        Random seed for reproducibility.
    """

    _MODEL_FILE = "lgbm_ranker.txt"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping: int = 50,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self._early_stopping = early_stopping
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            device=device,
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10],
            verbose=-1,
        )
        self._model: lgb.LGBMRanker | None = None

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: Union[pd.DataFrame, Path, str],
        val_df: pd.DataFrame,
        feature_columns: list[str],
        label_col: str = "is_positive",
        group_col: str = "userId",
    ) -> None:
        # ── train data — stream from parquet or use in-memory DataFrame ───
        if isinstance(train_df, (str, Path)):
            X_train, y_train, groups_train = _load_ranking_parquet(
                path=Path(train_df),
                feature_columns=feature_columns,
                label_col=label_col,
                group_col=group_col,
            )
        else:
            X_train = train_df[feature_columns].to_numpy(dtype=np.float32, copy=False)
            y_train = train_df[label_col].to_numpy(dtype=np.int32, copy=False)
            groups_train = train_df.sort_values(group_col).groupby(
                group_col, sort=False
            ).size().values.tolist()

        # ── val data — always a DataFrame ────────────────────────────────
        # Reindex so missing interaction/time features are filled with NaN.
        feature_columns = list(dict.fromkeys(feature_columns))
        val_sorted = val_df.sort_values(group_col)
        missing_in_val = [c for c in feature_columns if c not in val_sorted.columns]
        if missing_in_val:
            logger.warning(
                "Val is missing %d feature(s) — filled with NaN: %s",
                len(missing_in_val), missing_in_val,
            )
        X_val = val_sorted.reindex(columns=feature_columns).to_numpy(dtype=np.float32)
        y_val = val_sorted[label_col].to_numpy(dtype=np.int32, copy=False)
        groups_val = val_sorted.groupby(group_col, sort=False).size().values.tolist()
        del val_sorted

        self._model = lgb.LGBMRanker(**self._params)
        self._model.fit(
            X_train,
            y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)],
            eval_group=[groups_val],
            callbacks=[
                lgb.early_stopping(self._early_stopping, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        logger.info("LGBMRanker.fit complete.")

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LGBMRanker: call fit() before predict().")
        X = features_df[feature_columns].values.astype(np.float32)
        return self._model.predict(X)

    # ── persistence ───────────────────────────────────────────────────────

    def save_artifacts(self, output_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("LGBMRanker: nothing to save — model not fitted.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(output_dir / self._MODEL_FILE))
        logger.info("LGBMRanker saved to %s.", output_dir)

    def load_artifacts(self, output_dir: Path) -> None:
        model_path = output_dir / self._MODEL_FILE
        booster = lgb.Booster(model_file=str(model_path))
        self._model = lgb.LGBMRanker()
        self._model._Booster = booster  # type: ignore[attr-defined]
        logger.info("LGBMRanker loaded from %s.", output_dir)

    # ── introspection ─────────────────────────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        booster = self._model.booster_
        names = booster.feature_name()
        scores = booster.feature_importance(importance_type="gain").astype(float)
        total = scores.sum() or 1.0
        return {n: float(s / total) for n, s in sorted(zip(names, scores), key=lambda x: -x[1])}

    def log_to_mlflow(self) -> None:
        try:
            import mlflow
            loggable = {k: v for k, v in self._params.items() if isinstance(v, (int, float, str))}
            mlflow.log_params({f"lgbm.{k}": v for k, v in loggable.items()})
        except Exception as exc:  # pragma: no cover
            logger.warning("LGBMRanker.log_to_mlflow failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _load_ranking_parquet(
    path: Path,
    feature_columns: list[str],
    label_col: str,
    group_col: str,
    batch_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Read a ranking parquet in chunks → (X, y, groups) sorted by group_col.

    LightGBM requires consecutive rows per query group, so we sort by
    *group_col* after concatenating all numpy chunks.  The sort is done on
    the integer uid array (cheap) and used as an index into X and y so only
    one extra index array (~160 MB for 20 M rows) is allocated.

    Returns
    -------
    X_train : float32 ndarray (n_rows, n_features)
    y_train : int32 ndarray  (n_rows,)
    groups  : list[int]      consecutive group sizes sorted by group_col
    """
    import pyarrow.parquet as pq

    feature_columns = list(dict.fromkeys(feature_columns))  # deduplicate, preserve order
    load_cols = list(dict.fromkeys([group_col, label_col] + feature_columns))
    pf = pq.ParquetFile(str(path))
    available = set(pf.schema_arrow.names)
    load_cols = [c for c in load_cols if c in available]

    X_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    uid_chunks: list[np.ndarray] = []

    for batch in pf.iter_batches(batch_size=batch_size, columns=load_cols):
        chunk = batch.to_pandas()
        # reindex so absent columns become NaN rather than KeyError
        X_chunks.append(chunk.reindex(columns=feature_columns).to_numpy(dtype=np.float32))
        y_chunks.append(chunk[label_col].to_numpy(dtype=np.int32, copy=False))
        uid_chunks.append(chunk[group_col].to_numpy(copy=False))
        del chunk, batch

    X = np.concatenate(X_chunks); del X_chunks
    y = np.concatenate(y_chunks); del y_chunks
    uids = np.concatenate(uid_chunks); del uid_chunks

    # Sort by userId so group boundaries are contiguous
    sort_idx = np.argsort(uids, kind="stable")
    X = X[sort_idx]
    y = y[sort_idx]
    uids = uids[sort_idx]
    del sort_idx

    # Compute consecutive group sizes
    _, counts = np.unique(uids, return_counts=True)
    del uids

    logger.info(
        "_load_ranking_parquet: %d rows, %d features, %d groups",
        len(y), X.shape[1], len(counts),
    )
    return X, y, counts.tolist()
