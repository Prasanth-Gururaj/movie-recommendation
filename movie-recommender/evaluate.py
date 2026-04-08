"""Test-set evaluation CLI.

Usage
-----
    python evaluate.py --config configs/experiments/xgb_full_tuned.yaml

Loads the Production model from MLflow (or local artifacts), evaluates on
``data/processed/test_features.parquet`` (NO raw data loading, NO feature
recomputation), and saves reports to ``reports/``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("evaluate")


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to ExperimentConfig YAML.",
)
@click.option(
    "--model-name",
    default=None,
    help="MLflow registered model name. Defaults to <experiment_name>_ranker.",
)
@click.option(
    "--stage",
    default="Production",
    show_default=True,
    help="Model Registry stage to load.",
)
def main(config: Path, model_name: str | None, stage: str) -> None:
    """Evaluate the best model on the held-out test split."""
    from src.config.experiment_config import ExperimentConfig
    from src.ranking.ranker_factory import RankerFactory
    from src.evaluation.evaluator import Evaluator

    click.echo(f"Loading config: {config}")
    cfg = ExperimentConfig.from_yaml(config)
    recommender_type = getattr(cfg.training, "recommender_type", "two_stage")

    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)

    if model_name is None:
        model_name = f"{cfg.training.experiment_name}_ranker"

    processed_dir = Path(cfg.data.processed_data_dir)
    _require_processed(processed_dir)

    # ── Load test features (frozen parquet — no raw data) ─────────────────
    click.echo("Loading test_features.parquet …")
    test_feat_df = pd.read_parquet(processed_dir / "test_features.parquet")
    click.echo(f"  test_features: {len(test_feat_df):,} rows")

    feat_cols_path = Path("configs/feature_columns.json")
    feat_cols: list[str] = json.loads(feat_cols_path.read_text())

    # ── Load ranker (two_stage only) ──────────────────────────────────────
    ranker = None
    if recommender_type == "two_stage":
        artifact_dir = Path("artifacts") / cfg.training.run_name
        click.echo(f"Loading ranker from {artifact_dir} …")
        ranker = RankerFactory.create(cfg)
        ranker.load_artifacts(artifact_dir)

    # ── Two-track evaluation: warm + cold ─────────────────────────────────
    click.echo("Generating test predictions …")
    warm_preds, warm_labels = [], []
    cold_preds, cold_labels = [], []

    all_test_users = test_feat_df["userId"].unique()
    cold_threshold = cfg.data.cold_user_threshold

    for uid in all_test_users:
        user_rows = test_feat_df[test_feat_df["userId"] == uid]
        rel_items = set(
            user_rows[user_rows["is_positive"] == 1]["movieId"].astype(int)
        )
        if not rel_items:
            continue

        if recommender_type == "two_stage" and ranker is not None:
            feat_df = user_rows.reindex(columns=feat_cols, fill_value=0.0)
            scores = ranker.predict(feat_df, feat_cols)
        else:
            scores = _baseline_scores(user_rows, recommender_type)

        order = np.argsort(-scores)
        ranked = user_rows["movieId"].values[order].astype(int).tolist()
        ranked_k = ranked[: cfg.eval.k]

        is_warm = int(user_rows["is_positive"].sum()) >= cold_threshold
        if is_warm:
            warm_preds.append(ranked_k)
            warm_labels.append(rel_items)
        else:
            cold_preds.append(ranked_k)
            cold_labels.append(rel_items)

    # ── Build evaluator from item_features.parquet ─────────────────────────
    item_feat_df = pd.read_parquet(processed_dir / "item_features.parquet")
    genre_path = Path("configs/genre_columns.json")
    genre_columns = json.loads(genre_path.read_text()) if genre_path.exists() else []

    evaluator = Evaluator(
        cfg,
        item_popularity=_build_item_popularity(item_feat_df),
        item_genre_vectors=_build_genre_vectors(item_feat_df, genre_columns),
        cold_item_ids=_build_cold_ids(item_feat_df),
        catalog_size=len(item_feat_df),
    )

    warm_report = evaluator.evaluate_full(
        warm_preds, warm_labels,
        config_name=cfg.training.run_name,
        split="test_warm",
    )
    cold_report = evaluator.evaluate_full(
        cold_preds, cold_labels,
        config_name=cfg.training.run_name,
        split="test_cold",
    )

    # ── Save reports ──────────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    for report, suffix in [(warm_report, "warm"), (cold_report, "cold")]:
        path = reports_dir / f"{cfg.training.run_name}_{suffix}_test_report.json"
        path.write_text(
            json.dumps(
                {
                    "config_name": report.config_name,
                    "split": report.split,
                    "n_users": report.n_users,
                    "ranking": report.ranking,
                    "diversity": report.diversity,
                },
                indent=2,
            )
        )

    click.echo(
        f"Warm-test MAP@{cfg.eval.k} = {warm_report.primary_metric(cfg.eval.k):.4f}"
        f"  ({warm_report.n_users} users)"
    )
    click.echo(
        f"Cold-test MAP@{cfg.eval.k} = {cold_report.primary_metric(cfg.eval.k):.4f}"
        f"  ({cold_report.n_users} users)"
    )
    click.echo(f"Reports saved to {reports_dir}/")


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_processed(processed_dir: Path) -> None:
    for required in ("test_features.parquet",):
        if not (processed_dir / required).exists():
            raise SystemExit(
                f"\nERROR: {required} not found in {processed_dir}/\n"
                "Run this first:\n\n"
                "    python scripts/build_features.py --config configs/experiments/baseline_popularity.yaml\n"
            )


def _baseline_scores(user_rows: pd.DataFrame, recommender_type: str) -> np.ndarray:
    if recommender_type == "popularity":
        col = "log_rating_count"
    elif recommender_type == "genre_pop":
        pop = user_rows["log_rating_count"].fillna(0.0).values if "log_rating_count" in user_rows.columns else np.zeros(len(user_rows))
        overlap = user_rows["genre_overlap_score"].fillna(0.0).values if "genre_overlap_score" in user_rows.columns else np.zeros(len(user_rows))
        return (overlap * pop).astype(np.float32)
    elif recommender_type in ("cf", "als"):
        col = "mf_score"
    else:
        col = "log_rating_count"

    return user_rows[col].fillna(0.0).values.astype(np.float32) if col in user_rows.columns else np.zeros(len(user_rows), dtype=np.float32)


def _build_item_popularity(item_feat_df: pd.DataFrame) -> dict[int, float]:
    if "movieId" not in item_feat_df.columns or "log_rating_count" not in item_feat_df.columns:
        return {}
    return dict(zip(item_feat_df["movieId"].astype(int), item_feat_df["log_rating_count"].astype(float)))


def _build_genre_vectors(item_feat_df: pd.DataFrame, genre_columns: list[str]) -> dict[int, list[float]]:
    genre_cols = [f"genre_{g}" for g in genre_columns if f"genre_{g}" in item_feat_df.columns]
    if not genre_cols or "movieId" not in item_feat_df.columns:
        return {}
    result = {}
    for _, row in item_feat_df.iterrows():
        result[int(row["movieId"])] = [float(row[c]) for c in genre_cols]
    return result


def _build_cold_ids(item_feat_df: pd.DataFrame) -> set[int]:
    if "movieId" not in item_feat_df.columns or "is_cold" not in item_feat_df.columns:
        return set()
    return set(item_feat_df[item_feat_df["is_cold"] == 1]["movieId"].astype(int).tolist())


if __name__ == "__main__":
    main()
