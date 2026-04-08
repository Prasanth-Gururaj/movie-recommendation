"""Create a 1% sample of the full dataset for fast pipeline testing.

Usage
-----
    python scripts/create_sample_dataset.py

Output
------
    data/sample/  — identical structure to data/processed/

The full two-stage pipeline (train.py --sample) completes in under 2 minutes
on the sample data, making it easy to iterate on bugs without waiting for
the full 20M-row training set.

What is sampled
---------------
- 1 000 users who have at least 20 train ratings
- All parquet splits filtered to those users (train/val/test pairs + features)
- ALS artifacts copied as-is (full model — not refit on sample)
- Config JSON files copied as-is
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_USERS = 1_000
SAMPLE_SEED = 42
PROCESSED_DIR = Path("data/processed")
SAMPLE_DIR = Path("data/sample")


def create_sample() -> None:
    if not PROCESSED_DIR.exists():
        print(
            f"ERROR: {PROCESSED_DIR} does not exist.\n"
            "Run scripts/build_features.py first."
        )
        sys.exit(1)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Creating sample dataset in {SAMPLE_DIR}/ …")

    # ── 1. Pick SAMPLE_USERS eligible users from train_pairs ──────────────
    train_pairs = pd.read_parquet(PROCESSED_DIR / "train_pairs.parquet")
    user_counts = train_pairs.groupby("userId").size()
    eligible = user_counts[user_counts >= 20].index.tolist()

    n_sample = min(SAMPLE_USERS, len(eligible))
    sampled_users: set[int] = set(
        pd.Series(eligible)
        .sample(n=n_sample, random_state=SAMPLE_SEED)
        .astype(int)
        .tolist()
    )
    print(f"  Sampled {len(sampled_users):,} users from {len(eligible):,} eligible.")

    # ── 2. Filter each parquet split ──────────────────────────────────────
    # Small splits (<500 MB): read fully into RAM.
    # Large splits (train_features, ~10 GB): stream row-group by row-group.
    LARGE_SPLITS = {"train_features"}  # OOM if read whole; stream these
    splits = [
        "train_pairs",
        "val_pairs",
        "test_pairs",
        "train_features",
        "val_features",
        "test_features",
        "user_features",
        "item_features",
    ]
    import pyarrow as pa
    import pyarrow.parquet as pq

    for split in splits:
        src = PROCESSED_DIR / f"{split}.parquet"
        dst = SAMPLE_DIR / f"{split}.parquet"
        if not src.exists():
            print(f"  SKIP {split}.parquet — not found in {PROCESSED_DIR}/")
            continue

        if split in LARGE_SPLITS:
            # Stream row-group by row-group — never loads full file into RAM
            pf = pq.ParquetFile(str(src))
            has_user_col = "userId" in pf.schema_arrow.names
            writer = None
            total_rows = 0
            for batch in pf.iter_batches(batch_size=500_000):
                chunk = batch.to_pandas()
                if has_user_col:
                    chunk = chunk[chunk["userId"].astype(int).isin(sampled_users)]
                if chunk.empty:
                    continue
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(dst), table.schema)
                writer.write_table(table)
                total_rows += len(chunk)
                del chunk, table
            if writer:
                writer.close()
            print(f"  {split:<25} {total_rows:>10,} rows  →  {dst}  (streamed)")
        else:
            df = pd.read_parquet(src)
            if "userId" in df.columns:
                df = df[df["userId"].astype(int).isin(sampled_users)]
            df.to_parquet(dst, engine="pyarrow", index=False)
            print(f"  {split:<25} {len(df):>10,} rows  →  {dst}")

    # ── 3. Copy config JSON files ─────────────────────────────────────────
    for fname in ["feature_columns.json", "genre_columns.json", "genome_tag_columns.json"]:
        src = Path("configs") / fname
        if src.exists():
            shutil.copy(src, SAMPLE_DIR / fname)
            print(f"  configs/{fname}  →  {SAMPLE_DIR / fname}")

    # ── 4. Copy ALS artifacts (full model — not refit on sample) ──────────
    als_files = [
        "als_user_factors.npy",
        "als_item_factors.npy",
        "als_movie_id_map.npy",
        "als_user_id_map.npy",
        "faiss_item_index.bin",
        "splits_metadata.json",
    ]
    for fname in als_files:
        src = PROCESSED_DIR / fname
        if src.exists():
            shutil.copy(src, SAMPLE_DIR / fname)
            print(f"  {fname}  →  {SAMPLE_DIR / fname}")
        else:
            print(f"  SKIP {fname} — not found")

    print(f"\nDone. Sample dataset in {SAMPLE_DIR}/")
    print(f"  Users sampled : {len(sampled_users):,}")
    print(
        "\nRun the fast pipeline with:\n"
        "  python train.py --config configs/experiments/xgb_user_item_only.yaml --sample"
    )


if __name__ == "__main__":
    create_sample()
