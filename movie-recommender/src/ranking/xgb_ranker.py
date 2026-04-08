"""XGBoost learning-to-rank wrapper (rank:pairwise objective).

Memory-efficient path
---------------------
When ``train_df`` is a ``Path``, ``fit()`` streams the parquet row-group by
row-group via ``xgb.QuantileDMatrix`` + ``xgb.DataIter``.  Peak RAM for a
20 M × 118 training set is ~300 MB (one batch) + the DMatrix histogram
structures (~1–2 GB) rather than the full 9.4 GB numpy array.

In-memory path
--------------
When ``train_df`` is a ``pd.DataFrame`` (e.g. small test datasets), the
original dense path is used unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from src.ranking.base_ranker import BaseRanker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streaming data iterator for QuantileDMatrix
# ---------------------------------------------------------------------------

class _ParquetRankingIter(xgb.DataIter):
    """Iterates over a parquet file in fixed-size batches for QuantileDMatrix.

    Passes ``qid`` (userId) per row so XGBoost can handle non-contiguous
    query groups — no global sort of the parquet required.
    """

    def __init__(
        self,
        path: Path,
        feature_columns: list[str],
        label_col: str,
        uid_col: str,
        batch_size: int = 500_000,
    ) -> None:
        self._path = Path(path)
        self._feat_cols = list(dict.fromkeys(feature_columns))  # deduplicate, preserve order
        self._label_col = label_col
        self._uid_col = uid_col
        self._batch_size = batch_size
        self._iter = None
        super().__init__()

    # xgb.DataIter protocol ------------------------------------------------

    def reset(self) -> None:
        """Reopen the parquet file and restart the batch iterator."""
        import pyarrow.parquet as pq
        load_cols = list(dict.fromkeys(
            [self._uid_col, self._label_col] + self._feat_cols
        ))
        pf = pq.ParquetFile(str(self._path))
        available = set(pf.schema_arrow.names)
        load_cols = [c for c in load_cols if c in available]
        self._iter = pf.iter_batches(batch_size=self._batch_size, columns=load_cols)

    def next(self, input_data: Callable) -> int:
        if self._iter is None:
            self.reset()
        try:
            batch = next(self._iter)  # type: ignore[arg-type]
        except StopIteration:
            return 0

        chunk = batch.to_pandas()
        # reindex handles columns present in feature_columns.json but absent
        # from ltr_train.parquet (fills with NaN; XGBoost treats as missing).
        X = chunk.reindex(columns=self._feat_cols).to_numpy(dtype=np.float32)
        y = chunk[self._label_col].to_numpy(dtype=np.int32, copy=False)
        qid = chunk[self._uid_col].to_numpy(copy=False)
        del chunk, batch

        input_data(data=X, label=y, qid=qid)
        del X, y, qid
        return 1


# ---------------------------------------------------------------------------
# XGBRanker
# ---------------------------------------------------------------------------

class XGBRanker(BaseRanker):
    """Thin wrapper around the XGBoost booster with ``rank:pairwise``.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth.
    learning_rate:
        Step-shrinkage.
    subsample / colsample_bytree:
        Row / column sub-sampling ratios.
    early_stopping:
        Patience (rounds) for early stopping on the val set.
    random_state:
        Random seed.
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    _MODEL_FILE = "xgb_ranker.json"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping: int = 50,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self._n_estimators = n_estimators
        self._early_stopping = early_stopping
        self._train_params: dict = dict(
            objective="rank:pairwise",
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            device=device,
            eval_metric="ndcg@10",
            seed=random_state,
        )
        # Keep a copy of all params for MLflow logging
        self._params = dict(
            n_estimators=n_estimators,
            early_stopping=early_stopping,
            random_state=random_state,
            **self._train_params,
        )
        self._model: xgb.Booster | None = None

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: Union[pd.DataFrame, Path, str],
        val_df: pd.DataFrame,
        feature_columns: list[str],
        label_col: str = "is_positive",
        group_col: str = "userId",
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        feature_columns_dedup = list(dict.fromkeys(feature_columns))

        # ── train DMatrix + early-stopping holdout ────────────────────────
        if isinstance(train_df, (str, Path)):
            path = Path(train_df)
            pf = pq.ParquetFile(str(path))
            n_groups = pf.metadata.num_row_groups
            available_cols = set(pf.schema_arrow.names)
            load_cols = list(dict.fromkeys(
                [group_col, label_col] + feature_columns_dedup
            ))
            load_cols = [c for c in load_cols if c in available_cols]

            # Sample every 10th row group as early-stopping holdout (~10% of train).
            # This gives XGBoost a large, properly distributed validation signal
            # instead of val_features.parquet (tiny per-user pools → noisy NDCG).
            holdout_indices = list(range(0, n_groups, 10)) or [0]
            holdout_tables = [
                pf.read_row_group(i, columns=load_cols) for i in holdout_indices
            ]
            holdout_df = pa.concat_tables(holdout_tables).to_pandas()
            logger.info(
                "XGBRanker: early-stopping holdout = %d rows "
                "(%d/%d train row groups, every 10th).",
                len(holdout_df), len(holdout_indices), n_groups,
            )

            # Build holdout DMatrix from the in-memory sample
            holdout_sorted = holdout_df.sort_values(group_col)
            X_es = holdout_sorted.reindex(
                columns=feature_columns_dedup, fill_value=np.nan
            ).to_numpy(dtype=np.float32)
            y_es = holdout_sorted[label_col].to_numpy(dtype=np.int32, copy=False)
            g_es = holdout_sorted.groupby(group_col, sort=False).size().values
            d_holdout = xgb.DMatrix(
                X_es, label=y_es,
                feature_names=feature_columns_dedup,  # name every column
                enable_categorical=False,
            )
            d_holdout.set_group(g_es)
            del X_es, y_es, g_es, holdout_sorted, holdout_df, holdout_tables

            # Stream the full parquet into QuantileDMatrix for training
            it = _ParquetRankingIter(
                path=path,
                feature_columns=feature_columns,
                label_col=label_col,
                uid_col=group_col,
            )
            dtrain = xgb.QuantileDMatrix(it, enable_categorical=False)
            # QuantileDMatrix built from a DataIter doesn't pick up column names
            # from the numpy arrays inside next() — set them explicitly so the
            # saved model stores real names instead of f0, f1, f2, ...
            dtrain.feature_names = feature_columns_dedup
            logger.info(
                "Built QuantileDMatrix from parquet (%d row groups) — "
                "feature_names set (%d cols).",
                n_groups, len(feature_columns_dedup),
            )

            evals_list = [(d_holdout, "train_holdout")]
            es_data_name = "train_holdout"

        else:
            # In-memory path (small DataFrames / unit tests): use val_df for early stopping
            X_tr = train_df.reindex(columns=feature_columns_dedup, fill_value=0.0).to_numpy(dtype=np.float32)
            y_tr = train_df[label_col].to_numpy(dtype=np.int32, copy=False)
            g_tr = train_df.groupby(group_col, sort=False).size().values
            dtrain = xgb.DMatrix(
                X_tr, label=y_tr,
                feature_names=feature_columns_dedup,
                enable_categorical=False,
            )
            dtrain.set_group(g_tr)
            del X_tr, y_tr, g_tr

            val_sorted = val_df.sort_values(group_col)
            X_val = val_sorted.reindex(
                columns=feature_columns_dedup, fill_value=np.nan
            ).to_numpy(dtype=np.float32)
            y_val = val_sorted[label_col].to_numpy(dtype=np.int32, copy=False)
            g_val = val_sorted.groupby(group_col, sort=False).size().values
            dval = xgb.DMatrix(
                X_val, label=y_val,
                feature_names=feature_columns_dedup,
                enable_categorical=False,
            )
            dval.set_group(g_val)
            del X_val, y_val, g_val, val_sorted

            evals_list = [(dval, "val")]
            es_data_name = "val"

        # ── train ─────────────────────────────────────────────────────────
        callbacks = [
            xgb.callback.EarlyStopping(
                rounds=self._early_stopping,
                metric_name="ndcg@10",
                data_name=es_data_name,
                maximize=True,
                save_best=True,
            )
        ]
        self._model = xgb.train(
            params=self._train_params,
            dtrain=dtrain,
            num_boost_round=self._n_estimators,
            evals=evals_list,
            callbacks=callbacks,
            verbose_eval=50,
        )
        logger.info(
            "XGBRanker.fit complete — best_iteration=%d  (early_stopping=%d rounds).",
            self._model.best_iteration,
            self._early_stopping,
        )

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, features_df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBRanker: call fit() before predict().")
        feature_columns = list(dict.fromkeys(feature_columns))
        X = features_df.reindex(columns=feature_columns, fill_value=0.0).to_numpy(dtype=np.float32)
        # Pass feature_names so XGBoost validates column alignment against the saved model.
        # If the model was trained with named features and inference passes wrong names,
        # XGBoost will raise rather than silently misalign.
        dm = xgb.DMatrix(X, feature_names=feature_columns, enable_categorical=False)
        return self._model.predict(dm)

    # ── persistence ───────────────────────────────────────────────────────

    def save_artifacts(self, output_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("XGBRanker: nothing to save — model not fitted.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / self._MODEL_FILE  # always xgb_ranker.json — never .pkl
        self._model.save_model(str(save_path))
        logger.info(
            "XGBRanker saved to %s  (best_iteration=%d).",
            save_path,
            self._model.best_iteration,
        )

    def load_artifacts(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        model_path = output_dir / self._MODEL_FILE  # xgb_ranker.json
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBRanker artifact not found: {model_path}\n"
                "Expected file: xgb_ranker.json  (NOT model.pkl)\n"
                "Run train.py first to produce the artifact."
            )
        self._model = xgb.Booster()
        self._model.load_model(str(model_path))
        logger.info("XGBRanker loaded from %s.", output_dir)

    # ── introspection ─────────────────────────────────────────────────────

    def get_feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        scores = self._model.get_fscore()
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in sorted(scores.items(), key=lambda x: -x[1])}

    def log_to_mlflow(self) -> None:
        try:
            import mlflow
            loggable = {k: v for k, v in self._params.items() if isinstance(v, (int, float, str))}
            mlflow.log_params({f"xgb.{k}": v for k, v in loggable.items()})
        except Exception as exc:  # pragma: no cover
            logger.warning("XGBRanker.log_to_mlflow failed: %s", exc)
