"""Feature selection: trains a fast XGBoost probe on a sample of training data,
scores gain-based feature importances, drops low-importance genome_tag_* features,
and saves an updated feature_columns.json.

Usage
-----
    # Preview without saving:
    python scripts/select_features.py --config configs/experiments/xgb_full_tuned.yaml --dry-run

    # Save with default threshold (0.001):
    python scripts/select_features.py --config configs/experiments/xgb_full_tuned.yaml

    # Keep only top-20 genome tags instead of threshold:
    python scripts/select_features.py --config configs/experiments/xgb_full_tuned.yaml --top-n 20

    # Stricter threshold:
    python scripts/select_features.py --config configs/experiments/xgb_full_tuned.yaml --threshold 0.005

Protection policy
-----------------
Only ``genome_tag_*`` columns are eligible for automatic removal.
All user-base, genre-affinity, item-base, interaction, time, and MF features
are always kept regardless of their measured importance.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Add repo root to sys.path so src.* imports work when called from any cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("select_features")

# ---------------------------------------------------------------------------
# Feature protection rules
# ---------------------------------------------------------------------------

# genome_tag_* are the ONLY columns eligible for automatic removal.
# Everything else is protected by exact name or prefix match.
_PROTECTED_PREFIXES: tuple[str, ...] = (
    "genre_affinity_",       # user genre tastes
    "recent_genre_affinity_",
    "genre_",                # item genre flags (genre_Action, genre_Drama, …)
    "interaction_",          # time features: interaction_month, interaction_dayofweek
    "days_since_",           # days_since_active, days_since_user_active, days_since_item_rated
    "activity_",             # activity_30d, activity_90d
)

_PROTECTED_EXACT: frozenset[str] = frozenset({
    # user base
    "log_total_ratings", "log_positive_count", "mean_rating", "rating_variance",
    # user activity (covered by prefix too, but explicit is clearer)
    "days_since_active", "activity_30d", "activity_90d",
    # item base
    "log_rating_count", "avg_rating", "popularity_pct", "recent_pop_30d",
    "is_cold", "has_genre", "release_year", "movie_age", "log_movie_age",
    # genome flag
    "has_genome",
    # interaction features
    "genre_overlap_score", "tag_profile_similarity",
    "rating_gap", "genre_history_count",
    # MF score
    "mf_score",
})

_FEATURE_COLUMNS_PATH: Path = _REPO_ROOT / "configs" / "feature_columns.json"
_N_TRAIN_ROWS_ESTIMATE: int = 20_000_000  # used only for RAM-saving estimate in report


def _is_protected(feature: str) -> bool:
    """Return True if the feature must never be dropped."""
    if feature in _PROTECTED_EXACT:
        return True
    return any(feature.startswith(p) for p in _PROTECTED_PREFIXES)


def _is_droppable(feature: str) -> bool:
    """Return True if the feature is eligible for importance-based removal."""
    return feature.startswith("genome_tag_") and not _is_protected(feature)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to ExperimentConfig YAML.",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help=(
        "Drop genome_tag_* features with normalised gain importance below this value. "
        "Defaults to feature_selection.threshold in the YAML, or 0.001 if not set."
    ),
)
@click.option(
    "--top-n",
    "top_n",
    default=None,
    type=int,
    help=(
        "Keep only the top-N genome_tag_* features by importance. "
        "Overrides --threshold when set."
    ),
)
@click.option(
    "--sample-ratio",
    "sample_ratio",
    default=None,
    type=float,
    help=(
        "Fraction of train_features.parquet rows to sample for the probe model. "
        "Defaults to feature_selection.sample_ratio in the YAML, or 0.10."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be dropped without writing feature_columns.json.",
)
def main(
    config: Path,
    threshold: float | None,
    top_n: int | None,
    sample_ratio: float | None,
    dry_run: bool,
) -> None:
    """Train a fast XGBoost probe, score feature importances, prune genome tags."""
    import xgboost as xgb

    from src.config.experiment_config import ExperimentConfig

    # ── load configs ──────────────────────────────────────────────────────────
    cfg = ExperimentConfig.from_yaml(config)

    # Read the raw YAML to extract the optional feature_selection section
    raw_yaml: dict = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    fs_raw: dict = raw_yaml.get("feature_selection", {})

    # CLI flags override YAML; YAML overrides hard-coded defaults
    effective_threshold: float = threshold if threshold is not None else fs_raw.get("threshold", 0.001)
    effective_sample_ratio: float = sample_ratio if sample_ratio is not None else fs_raw.get("sample_ratio", 0.10)
    min_features: int = fs_raw.get("min_features", 0)
    seed: int = cfg.data.random_seed

    click.echo(
        f"Config: threshold={effective_threshold}  sample_ratio={effective_sample_ratio:.0%}"
        + (f"  top_n={top_n}" if top_n is not None else "")
        + ("  [DRY RUN]" if dry_run else "")
    )

    # ── load feature_columns.json ─────────────────────────────────────────────
    if not _FEATURE_COLUMNS_PATH.exists():
        raise SystemExit(
            f"\nERROR: {_FEATURE_COLUMNS_PATH} not found.\n"
            "Run scripts/build_features.py first.\n"
        )
    all_feature_cols: list[str] = json.loads(_FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
    click.echo(f"Loaded {len(all_feature_cols)} feature columns from feature_columns.json")

    # ── load sample of train_features.parquet ─────────────────────────────────
    parquet_path = Path(cfg.data.processed_data_dir) / "train_features.parquet"
    if not parquet_path.exists():
        raise SystemExit(
            f"\nERROR: {parquet_path} not found.\n"
            "Run scripts/build_features.py first.\n"
        )

    click.echo(f"Reading {parquet_path} …")
    full_df = pd.read_parquet(parquet_path)
    n_total = len(full_df)
    n_sample = max(1000, int(n_total * effective_sample_ratio))
    df = full_df.sample(n=n_sample, random_state=42).copy()  # fixed seed=42 for reproducibility
    del full_df
    click.echo(f"Sampled {n_sample:,} / {n_total:,} rows ({effective_sample_ratio:.0%})")

    # ── prepare X, y, groups ──────────────────────────────────────────────────
    label_col = "is_positive" if "is_positive" in df.columns else "label"
    if label_col not in df.columns:
        raise SystemExit(
            "ERROR: Neither 'is_positive' nor 'label' column found in training parquet."
        )

    present_cols = [c for c in all_feature_cols if c in df.columns]
    missing_cols = [c for c in all_feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning(
            "%d columns in feature_columns.json are absent from parquet and will be skipped: %s",
            len(missing_cols), missing_cols,
        )

    df_sorted = df.sort_values("userId").reset_index(drop=True)
    X = df_sorted[present_cols].values.astype(np.float32)
    y = df_sorted[label_col].values.astype(np.int32)
    groups = df_sorted.groupby("userId", sort=False).size().values

    click.echo(
        f"Probe model input: {len(present_cols)} features | "
        f"{len(y):,} samples | {len(groups):,} user groups"
    )

    # ── fit probe ranker ──────────────────────────────────────────────────────
    click.echo("Fitting probe XGBoost (n_estimators=50, max_depth=4) …")
    probe = xgb.XGBRanker(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cpu",       # always CPU — fast probe, not production training
        objective="rank:pairwise",
        random_state=seed,
        verbosity=0,
        eval_metric="ndcg",
    )
    probe.fit(X, y, group=groups)
    click.echo("Probe fit complete.")

    # ── compute and normalise gain importances ────────────────────────────────
    raw_scores = probe.get_booster().get_score(importance_type="gain")
    # XGBoost names features "f0", "f1", … when trained on a numpy array
    importance_by_col: dict[str, float] = {}
    for fname, score in raw_scores.items():
        idx = int(fname[1:])
        if idx < len(present_cols):
            importance_by_col[present_cols[idx]] = float(score)

    # Features with zero splits get importance 0.0
    for col in present_cols:
        importance_by_col.setdefault(col, 0.0)

    total_imp = sum(importance_by_col.values()) or 1.0
    importance_norm: dict[str, float] = {k: v / total_imp for k, v in importance_by_col.items()}

    # ── decide which genome_tag_* columns to drop ─────────────────────────────
    genome_cols = [c for c in present_cols if _is_droppable(c)]
    genome_imp = {c: importance_norm.get(c, 0.0) for c in genome_cols}

    if top_n is not None:
        # Keep the top-N genome tags; drop everything else
        sorted_genome = sorted(genome_imp.items(), key=lambda x: -x[1])
        keep_genome: set[str] = {c for c, _ in sorted_genome[:top_n]}
        to_drop = [c for c in genome_cols if c not in keep_genome]
        click.echo(
            f"top-n={top_n}: keeping {len(keep_genome)} / {len(genome_cols)} genome tags"
        )
    else:
        to_drop = [c for c in genome_cols if genome_imp.get(c, 0.0) < effective_threshold]
        click.echo(
            f"threshold={effective_threshold}: dropping {len(to_drop)} / {len(genome_cols)} genome tags"
        )

    # ── enforce min_features constraint ──────────────────────────────────────
    selected = [c for c in all_feature_cols if c not in set(to_drop)]
    if min_features > 0 and len(selected) < min_features:
        shortfall = min_features - len(selected)
        rescue_candidates = sorted(
            [(c, genome_imp.get(c, 0.0)) for c in to_drop],
            key=lambda x: -x[1],
        )
        rescued = {c for c, _ in rescue_candidates[:shortfall]}
        to_drop = [c for c in to_drop if c not in rescued]
        selected = [c for c in all_feature_cols if c not in set(to_drop)]
        logger.info(
            "Rescued %d features to satisfy min_features=%d: %s",
            len(rescued), min_features, sorted(rescued),
        )

    # ── print report ──────────────────────────────────────────────────────────
    _print_report(
        all_feature_cols=all_feature_cols,
        selected=selected,
        to_drop=to_drop,
        genome_imp=genome_imp,
        importance_norm=importance_norm,
    )

    # ── log importance plot to MLflow ─────────────────────────────────────────
    _log_importance_plot(
        importance_norm=importance_norm,
        to_drop=to_drop,
        cfg=cfg,
    )

    # ── save or dry-run ───────────────────────────────────────────────────────
    click.echo("=" * 70)
    if dry_run:
        click.echo("DRY RUN — feature_columns.json NOT updated.")
        click.echo(f"Re-run without --dry-run to commit the {len(to_drop)} dropped features.")
    else:
        _FEATURE_COLUMNS_PATH.write_text(
            json.dumps(selected, indent=2), encoding="utf-8"
        )
        click.echo(
            f"Saved {_FEATURE_COLUMNS_PATH}  "
            f"({len(selected)} features, reduced from {len(all_feature_cols)})"
        )
        click.echo("Next: python train.py --config <your_config.yaml>")
    click.echo("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_report(
    all_feature_cols: list[str],
    selected: list[str],
    to_drop: list[str],
    genome_imp: dict[str, float],
    importance_norm: dict[str, float],
) -> None:
    """Print a human-readable selection report to stdout."""
    n_before = len(all_feature_cols)
    n_dropped = len(to_drop)
    n_kept = len(selected)

    click.echo("\n" + "=" * 70)
    click.echo("FEATURE SELECTION REPORT")
    click.echo("=" * 70)
    click.echo(f"  Total features before : {n_before}")
    click.echo(f"  Features dropped      : {n_dropped}")
    click.echo(f"  Features kept         : {n_kept}")

    if to_drop:
        click.echo(f"\n  Dropped genome_tag_* ({n_dropped} features):")
        for col in sorted(to_drop, key=lambda c: genome_imp.get(c, 0.0)):
            click.echo(f"    {col:<44}  importance = {genome_imp.get(col, 0.0):.6f}")
    else:
        click.echo("\n  No features dropped (all genome tags meet the threshold).")

    # RAM saving estimate
    bytes_per_cell = 4  # float32
    saved_bytes = n_dropped * _N_TRAIN_ROWS_ESTIMATE * bytes_per_cell
    saved_gib = saved_bytes / (1024 ** 3)
    click.echo(
        f"\n  Estimated RAM saving for full training:\n"
        f"    {saved_gib:.2f} GiB  "
        f"({n_dropped} cols × {_N_TRAIN_ROWS_ESTIMATE / 1e6:.0f}M rows × 4 B)"
    )

    # Top-20 most important features
    top20 = sorted(importance_norm.items(), key=lambda x: -x[1])[:20]
    click.echo(f"\n  Top 20 features by gain importance:")
    click.echo(f"    {'Feature':<46} {'Importance':>10}  Note")
    click.echo(f"    {'-' * 46} {'-' * 10}  ----")
    for feat, imp in top20:
        if feat.startswith("genome_tag_"):
            note = "DROPPED" if feat in to_drop else "genome (kept)"
        else:
            note = ""
        click.echo(f"    {feat:<46} {imp:10.6f}  {note}")

    click.echo()


def _log_importance_plot(
    importance_norm: dict[str, float],
    to_drop: list[str],
    cfg,
) -> None:
    """Render a top-40 horizontal bar chart and log it to MLflow."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow

        mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.training.experiment_name)

        n_plot = 40
        top_items = sorted(importance_norm.items(), key=lambda x: -x[1])[:n_plot]
        names = [n for n, _ in top_items]
        scores = [s for _, s in top_items]
        to_drop_set = set(to_drop)

        # Color coding: red = dropped, blue = genome kept, green = other
        colors = [
            "#e74c3c" if n in to_drop_set
            else ("#3498db" if n.startswith("genome_tag_") else "#2ecc71")
            for n in names
        ]

        fig, ax = plt.subplots(figsize=(11, max(7, n_plot * 0.28)))
        y_pos = list(range(len(names)))
        ax.barh(y_pos, scores, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised gain importance")
        ax.set_title(
            f"Feature importances — probe XGBoost (top {n_plot} of {len(importance_norm)})\n"
            "Red = dropped  |  Blue = genome tag kept  |  Green = protected feature"
        )
        plt.tight_layout()

        plot_path = _REPO_ROOT / "artifacts" / "feature_importance.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(plot_path), dpi=120, bbox_inches="tight")
        plt.close(fig)

        n_before = len(importance_norm)
        n_dropped = len(to_drop)

        with mlflow.start_run(run_name="feature_selection"):
            mlflow.log_artifact(str(plot_path), artifact_path="feature_selection")
            mlflow.log_params({
                "n_features_before": n_before,
                "n_features_dropped": n_dropped,
                "n_features_kept": n_before - n_dropped,
            })

        click.echo(
            f"  Importance plot saved → {plot_path.relative_to(_REPO_ROOT)}"
            "  (also logged to MLflow)"
        )
    except Exception as exc:
        logger.warning("Could not generate/log importance plot: %s", exc)


if __name__ == "__main__":
    main()
