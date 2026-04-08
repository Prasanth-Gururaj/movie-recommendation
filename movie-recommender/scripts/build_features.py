"""One-time feature pipeline script.

Run this ONCE before any training runs:

    python scripts/build_features.py --config configs/experiments/baseline_popularity.yaml

What it does
------------
1. load_all()               — load raw MovieLens 25M data
2. clean                    — parse release year, build genre vectors
3. split_ratings()          — train/val/test splits saved as parquet
4. Fit ALS                  — user/item factors + FAISS index + ALS artifacts
5. feature_store.build_all_features()
                            — user, item, interaction, time features
                            — train_features.parquet  (with real mf_scores)
                            — val_features.parquet    (same schema, same mf_scores)
                            — test_features.parquet   (same schema, same mf_scores)
                            — user_features.parquet
                            — item_features.parquet
                            — feature_columns.json (locked, written LAST)
6. verify_feature_consistency() — assert identical columns across all 3 splits

Output files in data/processed/
--------------------------------
    train_pairs.parquet
    val_pairs.parquet
    test_pairs.parquet
    splits_metadata.json
    train_features.parquet
    val_features.parquet
    test_features.parquet
    user_features.parquet
    item_features.parquet
    als_user_factors.npy
    als_item_factors.npy
    als_movie_id_map.npy
    faiss_item_index.bin
    configs/feature_columns.json  (in configs/)

After this script completes, data/processed/ is READ ONLY.
train.py and evaluate.py NEVER write to data/processed/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build_features")


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to any ExperimentConfig YAML (only data/feature/training sections used).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-build even if processed files already exist.",
)
@click.option(
    "--sample",
    is_flag=True,
    default=False,
    help="Write outputs to data/sample/ instead of data/processed/. "
         "Requires data/sample/train_pairs.parquet to already exist "
         "(run scripts/create_sample_dataset.py first).",
)
def main(config: Path, force: bool, sample: bool) -> None:
    """Build and save all features to data/processed/."""
    from src.config.experiment_config import ExperimentConfig
    from src.ingestion.loader import load_all
    from src.ingestion.cleaner import parse_release_year, build_genre_vector
    from src.ingestion.splitter import split_ratings, save_splits
    from src.features.feature_store import FeatureStore
    from src.candidates.matrix_factorization import ALSCandidateGenerator

    cfg = ExperimentConfig.from_yaml(config)
    processed_dir = Path("data/sample") if sample else Path(cfg.data.processed_data_dir)
    if sample:
        # Override config so every downstream call (save_splits, FeatureStore,
        # ALSCandidateGenerator) writes to data/sample/ — not data/processed/.
        cfg.data.processed_data_dir = str(processed_dir)
        click.echo(f"  [--sample] Writing outputs to {processed_dir}/ (not data/processed/)")
        if not (processed_dir / "train_pairs.parquet").exists():
            raise SystemExit(
                f"\nERROR: {processed_dir}/train_pairs.parquet not found.\n"
                "Run this first:\n\n"
                "    python scripts/create_sample_dataset.py\n"
            )
    sentinel = processed_dir / "test_features.parquet"  # final file produced

    if sentinel.exists() and not force:
        click.echo(
            f"Features already exist at {processed_dir}/\n"
            "Use --force to rebuild."
        )
        _print_file_sizes(processed_dir)
        return

    # ── Step 1: load ──────────────────────────────────────────────────────────
    click.echo("Step 1/6 — Loading raw data …")
    data = load_all(cfg.data.raw_data_dir)

    # ── Step 2: clean ─────────────────────────────────────────────────────────
    click.echo("Step 2/6 — Cleaning …")
    genre_path = Path("configs/genre_columns.json")
    genre_columns = json.loads(genre_path.read_text()) if genre_path.exists() else []
    data["movies"] = parse_release_year(data["movies"])
    data["movies"] = build_genre_vector(data["movies"], genre_columns)

    # ── Step 3: split ─────────────────────────────────────────────────────────
    click.echo("Step 3/6 — Splitting and saving pairs …")
    train_df, val_df, test_df = split_ratings(data["ratings"], cfg.data)

    threshold = cfg.data.relevance_threshold
    for df in (train_df, val_df, test_df):
        df["is_positive"] = (df["rating"] >= threshold).astype("int8")

    save_splits(train_df, val_df, test_df, str(processed_dir), cfg.data)
    click.echo(
        f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,} rows"
    )

    # ── Step 4: fit ALS ───────────────────────────────────────────────────────
    click.echo("Step 4/6 — Fitting ALS (user/item factors + FAISS index) …")
    als_gen = ALSCandidateGenerator(cfg.training, cfg.feature)
    als_gen.fit(train_df, save_dir=processed_dir)
    click.echo(
        f"  ALS done — user_factors={als_gen._user_factors.shape}  "  # type: ignore[union-attr]
        f"item_factors={als_gen._item_factors.shape}"  # type: ignore[union-attr]
    )

    # ── Step 5: features ──────────────────────────────────────────────────────
    click.echo("Step 5/6 — Building features (train + val + test) …")
    feature_store = FeatureStore(cfg)
    feature_store.build_all_features(data, train_df, val_df, test_df, als_gen=als_gen)

    # ── Step 6: verify consistency ────────────────────────────────────────────
    click.echo("Step 6/6 — Verifying feature consistency …")
    verify_feature_consistency(processed_dir)

    click.echo("\nDone. Files written to data/processed/:")
    _print_file_sizes(processed_dir)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_feature_consistency(processed_dir: Path | None = None) -> None:
    """Assert that train/val/test parquets have identical column lists.

    Reads only the parquet schema (no data loaded into RAM).
    Raises ``AssertionError`` if any mismatch is found.
    """
    import pyarrow.parquet as pq

    base = Path(processed_dir) if processed_dir else Path("data/processed")

    train_cols = pq.read_schema(base / "train_features.parquet").names
    val_cols   = pq.read_schema(base / "val_features.parquet").names
    test_cols  = pq.read_schema(base / "test_features.parquet").names

    assert train_cols == val_cols == test_cols, (
        f"Column mismatch!\n"
        f"Train has {len(train_cols)} cols\n"
        f"Val   has {len(val_cols)} cols\n"
        f"Test  has {len(test_cols)} cols\n"
        f"Missing in val:  {sorted(set(train_cols) - set(val_cols))}\n"
        f"Missing in test: {sorted(set(train_cols) - set(test_cols))}\n"
        f"Extra in val:    {sorted(set(val_cols) - set(train_cols))}\n"
        f"Extra in test:   {sorted(set(test_cols) - set(train_cols))}"
    )
    click.echo(
        f"Column consistency verified — all 3 splits have {len(train_cols)} identical columns."
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_file_sizes(processed_dir: Path) -> None:
    for p in sorted(processed_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / 1_048_576
            click.echo(f"  {p.name:<45} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
