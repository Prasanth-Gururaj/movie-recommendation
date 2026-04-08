# DATA FLOW — read this before editing train.py
#
# scripts/build_features.py  (run ONCE before any training)
#   └── reads:  data/raw/*.csv
#   └── writes: data/processed/train_features.parquet  ← frozen
#               data/processed/val_features.parquet    ← frozen
#               data/processed/test_features.parquet   ← frozen
#               data/processed/user_features.parquet   ← frozen
#               data/processed/item_features.parquet   ← frozen
#               data/processed/als_*.npy               ← frozen
#               data/processed/faiss_item_index.bin    ← frozen
#               configs/feature_columns.json           ← frozen
#
# train.py --config configs/experiments/X.yaml  (run 9 times, one per variant)
#   └── reads:  data/processed/train_features.parquet  (NO transforms)
#               data/processed/val_features.parquet    (NO transforms)
#               configs/feature_columns.json
#   └── writes: mlruns/ (MLflow artifacts, metrics, model)
#
# evaluate.py --config configs/experiments/X.yaml  (run after all training)
#   └── reads:  data/processed/test_features.parquet   (NO transforms)
#               MLflow registry (Production model)
#   └── writes: reports/eval_report.json
#
# RULE: After build_features.py completes, data/processed/ is READ ONLY.
#       train.py and evaluate.py NEVER write to data/processed/.
#       train.py and evaluate.py NEVER compute features.

"""Training pipeline CLI.

Usage
-----
    python train.py --config configs/experiments/xgb_full_tuned.yaml

Steps (all recommender types)
------------------------------
1.  Load ExperimentConfig from --config YAML
2.  Start MLflow run, log config params
3.  Load val_features.parquet (always needed for evaluation)
4.  If recommender_type == "two_stage":
      a. Fit ranker (streams train_features.parquet via QuantileDMatrix / LightGBM)
      b. Evaluate on warm-val users by scoring val_features rows with ranker
    Else (baseline: popularity / genre_pop / cf / als):
      a. Skip ranker training
      b. Evaluate on warm-val users by scoring val_features rows with baseline signal
5.  Log metrics to MLflow
6.  Save artefacts and register if new best MAP@10
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to ExperimentConfig YAML.",
)
@click.option(
    "--sample",
    is_flag=True,
    default=False,
    help="Use data/sample/ instead of data/processed/ for fast pipeline testing.",
)
def main(config: Path, sample: bool) -> None:
    """Run the full training pipeline for one experiment config."""
    from src.config.experiment_config import ExperimentConfig
    from src.ranking.ranker_factory import RankerFactory
    from src.evaluation.evaluator import Evaluator

    click.echo(f"Loading config: {config}")
    cfg = ExperimentConfig.from_yaml(config)
    recommender_type = getattr(cfg.training, "recommender_type", "two_stage")

    mlflow.set_tracking_uri(cfg.training.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.training.experiment_name)

    # --sample overrides processed_data_dir from config
    processed_dir = Path("data/sample") if sample else Path(cfg.data.processed_data_dir)
    if sample:
        click.echo(f"  [--sample] Using {processed_dir}/ instead of data/processed/")
    _require_processed(processed_dir)

    with mlflow.start_run(run_name=cfg.training.run_name) as run:
        # ── Step 1: log config params ─────────────────────────────────────
        click.echo("Logging config params to MLflow …")
        mlflow.log_params(cfg.to_mlflow_params())

        # ── Step 2: load feature columns ──────────────────────────────────
        feat_cols_path = Path("configs/feature_columns.json")
        feat_cols: list[str] = json.loads(feat_cols_path.read_text())
        click.echo(f"  feature_columns: {len(feat_cols)} columns")

        # ── Step 3: load val features (always needed) ─────────────────────
        click.echo("Loading val_features.parquet …")
        val_feat_df = pd.read_parquet(processed_dir / "val_features.parquet")
        click.echo(f"  val_features: {len(val_feat_df):,} rows")

        # ── Step 4: train ranker or skip ──────────────────────────────────
        ranker = None
        if recommender_type == "two_stage":
            click.echo("Step 4 — Fitting ranker (two_stage) …")
            ranker = RankerFactory.create(cfg)
            ranker.fit(
                processed_dir / "train_features.parquet",
                val_feat_df,
                feature_columns=feat_cols,
                label_col="is_positive",
                group_col="userId",
            )
            ranker.log_to_mlflow()
            click.echo("  Ranker fit complete.")
        else:
            click.echo(
                f"Step 4 — Skipping ranker training (recommender_type={recommender_type})."
            )

        # ── Step 5: evaluate on warm and cold val users ───────────────────
        click.echo("Step 5 — Evaluating on val users (warm + cold tracks) …")

        # Load item + user features for catalog-based evaluation
        item_feat_df = pd.read_parquet(processed_dir / "item_features.parquet")
        user_feat_df = pd.read_parquet(processed_dir / "user_features.parquet")

        # Split val users into warm (seen in train) and cold (new users)
        all_warm_candidate_ids = _get_warm_users(val_feat_df, cfg)
        train_user_ids: set[int] = set(user_feat_df["userId"].astype(int).tolist())
        warm_user_ids = [uid for uid in all_warm_candidate_ids if uid in train_user_ids]
        cold_user_ids = [uid for uid in all_warm_candidate_ids if uid not in train_user_ids]
        logger.info(
            "Val user split — warm (in train): %d  cold (new users): %d  "
            "(from %d total with ≥%d val positives)",
            len(warm_user_ids), len(cold_user_ids),
            len(all_warm_candidate_ids), cfg.eval.warm_user_min_positives,
        )
        click.echo(
            f"  warm_val_users (in train): {len(warm_user_ids):,}  "
            f"cold_val_users (new): {len(cold_user_ids):,}"
        )

        # Load ALS arrays — needed for als/cf baselines and two_stage eval
        als_arrays = _try_load_als_arrays(processed_dir, user_feat_df)

        # ALS coverage check
        if als_arrays is not None:
            val_pos_items = set(
                val_feat_df[val_feat_df["is_positive"] == 1]["movieId"].astype(int).unique()
            )
            als_item_set = set(als_arrays["movie_id_to_idx"].keys())
            coverage = len(val_pos_items & als_item_set) / max(len(val_pos_items), 1)
            logger.info(
                "ALS covers %.1f%% of val positive items (%d / %d).",
                coverage * 100, len(val_pos_items & als_item_set), len(val_pos_items),
            )

        # Build evaluator (shared for both tracks)
        genre_path = Path("configs/genre_columns.json")
        genre_columns = json.loads(genre_path.read_text()) if genre_path.exists() else []
        evaluator = Evaluator(
            cfg,
            item_popularity=_build_item_popularity(item_feat_df),
            item_genre_vectors=_build_genre_vectors(item_feat_df, genre_columns),
            cold_item_ids=_build_cold_ids(item_feat_df),
            catalog_size=len(item_feat_df),
        )

        if recommender_type == "two_stage":
            assert ranker is not None, "Ranker is None — fit step did not complete"
            assert hasattr(ranker, "_model") and ranker._model is not None, (
                "Ranker._model is None — model was not trained"
            )
            best_iter = getattr(ranker._model, "best_iteration", "n/a")
            logger.info(
                "Using in-memory ranker for eval — best_iteration=%s  (no disk load)",
                best_iter,
            )
            from src.features.feature_store import FeatureStore
            feature_store = FeatureStore(cfg)

            # Track 1 — warm users (real per-user features)
            warm_preds, warm_labels = _score_with_ranker(
                val_feat_df, ranker, feat_cols, warm_user_ids, cfg,
                als_arrays, user_feat_df, item_feat_df, feature_store,
                processed_dir=processed_dir,
            )
            # Track 2 — cold users (mean fallback features)
            cold_preds, cold_labels = _score_with_ranker(
                val_feat_df, ranker, feat_cols, cold_user_ids, cfg,
                als_arrays, user_feat_df, item_feat_df, feature_store,
                processed_dir=processed_dir,
            )
        else:
            warm_preds, warm_labels = _score_baseline(
                val_feat_df, user_feat_df, item_feat_df,
                recommender_type, warm_user_ids, cfg, als_arrays,
            )
            cold_preds, cold_labels = _score_baseline(
                val_feat_df, user_feat_df, item_feat_df,
                recommender_type, cold_user_ids, cfg, als_arrays,
            )

        # ── Step 6: compute and log metrics (two tracks) ──────────────────
        warm_report = evaluator.evaluate_full(
            warm_preds, warm_labels,
            config_name=cfg.training.run_name,
            split="val_warm",
        )
        cold_report = evaluator.evaluate_full(
            cold_preds, cold_labels,
            config_name=cfg.training.run_name,
            split="val_cold",
        )

        warm_map = warm_report.primary_metric(cfg.eval.k)
        cold_map = cold_report.primary_metric(cfg.eval.k)

        click.echo("Step 6 — Logging metrics …")
        # Prefix warm/cold metrics so they don't collide in MLflow
        warm_metrics = {f"warm_{k}": v for k, v in warm_report.to_mlflow_metrics().items()}
        cold_metrics = {f"cold_{k}": v for k, v in cold_report.to_mlflow_metrics().items()}
        mlflow.log_metrics({
            **warm_metrics,
            **cold_metrics,
            "warm_user_count": len(warm_user_ids),
            "cold_user_count": len(cold_user_ids),
        })
        click.echo(
            f"  WARM MAP@{cfg.eval.k} = {warm_map:.4f}  ({len(warm_user_ids):,} users)  "
            f"| COLD MAP@{cfg.eval.k} = {cold_map:.4f}  ({len(cold_user_ids):,} users)"
        )

        # Primary metric for model registration = warm MAP (meaningful signal)
        report = warm_report

        # ── Step 7: save artefacts + register ─────────────────────────────
        if recommender_type == "two_stage" and ranker is not None:
            click.echo("Step 7 — Saving artefacts …")
            artifact_dir = Path("artifacts") / cfg.training.run_name
            ranker.save_artifacts(artifact_dir)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="ranker")

            _register_if_best(
                run_id=run.info.run_id,
                experiment_name=cfg.training.experiment_name,
                current_map=report.primary_metric(cfg.eval.k),
                model_name=f"{cfg.training.experiment_name}_ranker",
            )
        else:
            click.echo("Step 7 — No ranker to save (baseline run).")

    click.echo("Training complete.")


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_processed(processed_dir: Path) -> None:
    """Raise a clear error if features have not been built yet."""
    for required in ("train_features.parquet", "val_features.parquet", "test_features.parquet"):
        if not (processed_dir / required).exists():
            raise SystemExit(
                f"\nERROR: {required} not found in {processed_dir}/\n"
                "Run this first:\n\n"
                "    python scripts/build_features.py --config configs/experiments/baseline_popularity.yaml\n"
            )


def _get_warm_users(val_feat_df: pd.DataFrame, cfg) -> list[int]:
    """Users in val with at least warm_user_min_positives positive labels."""
    min_pos = cfg.eval.warm_user_min_positives
    pos_counts = val_feat_df.groupby("userId")["is_positive"].sum()
    return pos_counts[pos_counts >= min_pos].index.tolist()


def _try_load_als_arrays(processed_dir: Path, user_feat_df: pd.DataFrame) -> dict | None:
    """Load ALS factor matrices from data/processed/.

    Loads als_user_id_map.npy (saved by ALSCandidateGenerator.fit) to reconstruct
    user_id_to_idx with the EXACT same order used during training.
    Falls back to sorted user_features.parquet userIds if the file is missing
    (legacy compatibility — may be incorrect if cold-user filtering was applied).

    Returns None if ALS files are missing.
    """
    try:
        user_factors = np.load(processed_dir / "als_user_factors.npy")
        item_factors = np.load(processed_dir / "als_item_factors.npy")
        movie_id_map = np.load(processed_dir / "als_movie_id_map.npy")  # idx → movie_id

        user_id_map_path = processed_dir / "als_user_id_map.npy"
        if user_id_map_path.exists():
            user_id_map = np.load(user_id_map_path)  # saved by ALSCandidateGenerator.fit
            user_id_to_idx = {int(uid): i for i, uid in enumerate(user_id_map)}
            logger.info("Loaded als_user_id_map.npy (%d users).", len(user_id_to_idx))
        else:
            # Fallback: reconstruct from user_features.parquet — may misalign if
            # user_features has fewer users than train_df (cold-user filtering etc.)
            logger.warning(
                "als_user_id_map.npy not found — reconstructing from user_features.parquet. "
                "Re-run build_features.py --force to fix this."
            )
            sorted_users = sorted(user_feat_df["userId"].astype(int).unique().tolist())
            user_id_to_idx = {uid: i for i, uid in enumerate(sorted_users)}

        movie_id_to_idx = {int(mid): i for i, mid in enumerate(movie_id_map)}
        import faiss as _faiss
        faiss_index = _faiss.read_index(str(processed_dir / "faiss_item_index.bin"))
        logger.info(
            "ALS arrays loaded: %d users, %d items, dim=%d.",
            len(user_id_to_idx), len(movie_id_to_idx), user_factors.shape[1],
        )
        return {
            "user_factors": user_factors,
            "item_factors": item_factors,
            "movie_id_map": movie_id_map,
            "user_id_to_idx": user_id_to_idx,
            "movie_id_to_idx": movie_id_to_idx,
            "faiss_index": faiss_index,
        }
    except Exception as exc:
        logger.warning("Could not load ALS arrays from %s: %s", processed_dir, exc)
        return None


def _als_candidates(als: dict, uid: int, n: int) -> list[int]:
    """Return top-n ALS candidates for a user via FAISS search."""
    u_idx = als["user_id_to_idx"].get(uid)
    if u_idx is None:
        return []
    u_vec = als["user_factors"][u_idx].reshape(1, -1).astype(np.float32)
    norm = np.linalg.norm(u_vec)
    if norm > 0:
        u_vec = u_vec / norm
    k = min(n * 2, als["faiss_index"].ntotal)
    _, indices = als["faiss_index"].search(u_vec, k)
    candidates = []
    for idx in indices[0]:
        if 0 <= idx < len(als["movie_id_map"]):
            candidates.append(int(als["movie_id_map"][idx]))
        if len(candidates) >= n:
            break
    return candidates


def _score_with_ranker(
    val_feat_df: pd.DataFrame,
    ranker,
    feat_cols: list[str],
    warm_user_ids: list[int],
    cfg,
    als_arrays: dict | None,
    user_feat_df: pd.DataFrame,
    item_feat_df: pd.DataFrame,
    feature_store,
    processed_dir: Path = Path("data/processed"),
) -> tuple[list[list[int]], list[set[int]]]:
    """Evaluate two_stage ranker: hybrid candidate pool → assemble features → rank → cut.

    Fix 1: candidates are ranked FIRST (all 300+), top-k cut happens LAST.
    Fix 2: pool = popularity-top-100 ∪ ALS-top-200 → dedup → ≤300 candidates.
           Popularity ensures coverage of items ALS may miss.
    Fix 5: first 5 users log pool size, val positives, and overlap for diagnosis.
    """
    k = cfg.eval.k
    # Eval-specific larger pools (from EvalConfig) to maximise retrieval recall
    N_POP = getattr(cfg.eval, "n_candidates_pop_eval", 500)
    N_ALS = getattr(cfg.eval, "n_candidates_mf_eval", 500)
    N_HIST = getattr(cfg.eval, "n_history_candidates", 50)
    MAX_POOL = N_POP + N_ALS + N_HIST  # no hard cap — dedup handles it

    predictions: list[list[int]] = []
    labels: list[set[int]] = []

    # Val positives per user
    user_positives: dict[int, set[int]] = (
        val_feat_df[val_feat_df["is_positive"] == 1]
        .groupby("userId")["movieId"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

    # User train positives — loaded once, used for history candidates.
    # Adding train positives to the pool does NOT cause leakage:
    #   val positives are 2017 only; train positives are ≤2016.
    #   The ranker learns taste signal by ranking them vs. 1000 other candidates.
    train_pairs_path = processed_dir / "train_pairs.parquet"
    if train_pairs_path.exists() and N_HIST > 0:
        _tpdf = pd.read_parquet(
            train_pairs_path,
            columns=["userId", "movieId", "rating"],
        )
        threshold = cfg.data.relevance_threshold
        _tpdf = _tpdf[_tpdf["rating"] >= threshold]
        user_train_positives: dict[int, list[int]] = (
            _tpdf.sort_values("rating", ascending=False)
            .groupby("userId")["movieId"]
            .apply(lambda x: x.astype(int).tolist()[:N_HIST])
            .to_dict()
        )
        logger.info(
            "Loaded train positives for history candidates — %d users have train history.",
            len(user_train_positives),
        )
        del _tpdf
    else:
        user_train_positives = {}

    # Popularity-ordered item IDs (same for all users — built once)
    if "log_rating_count" in item_feat_df.columns:
        pop_ranked_ids: list[int] = (
            item_feat_df.sort_values("log_rating_count", ascending=False)
            ["movieId"].astype(int).tolist()
        )
    else:
        pop_ranked_ids = item_feat_df["movieId"].astype(int).tolist()

    import time as _time

    # Index by the ID column so .loc[id] gives the correct row — not row position
    user_feat_indexed = user_feat_df.set_index("userId")
    item_feat_indexed = item_feat_df.set_index("movieId")

    # Precompute mean user feature vector as fallback for cold val users
    # (users present in val but absent from user_features.parquet have no train history)
    numeric_user_cols = user_feat_df.select_dtypes(include="number").columns.tolist()
    if "userId" in numeric_user_cols:
        numeric_user_cols.remove("userId")
    mean_user_feat: dict = user_feat_df[numeric_user_cols].mean().to_dict()

    known_user_ids: set[int] = set(user_feat_df["userId"].astype(int).values)
    n_cold_skipped = sum(1 for uid in warm_user_ids if int(uid) not in known_user_ids)
    if n_cold_skipped:
        logger.info(
            "%d warm val users absent from user_features.parquet (cold val users) "
            "— will use mean user feature vector as fallback.",
            n_cold_skipped,
        )

    affinity_cols = [c for c in user_feat_df.columns if c.startswith("genre_affinity_")]
    genre_suffix = [c[len("genre_affinity_"):] for c in affinity_cols]

    debug_count = 0  # log first 5 users only

    for uid in warm_user_ids:
        rel_items = user_positives.get(int(uid), set())
        if not rel_items:
            continue

        # Build hybrid candidate pool:
        #   history_cands: user's own train positives (guarantees some overlap)
        #   als_cands:     personalised ALS/FAISS retrieval
        #   pop_cands:     popularity fallback (covers items ALS misses)
        # Priority: history first → ALS → pop. All deduped. No hard cap (dedup handles it).
        hist_cands = user_train_positives.get(int(uid), [])[:N_HIST]
        als_cands = _als_candidates(als_arrays, int(uid), N_ALS) if als_arrays else []
        pop_cands = pop_ranked_ids[:N_POP]

        seen: set[int] = set()
        candidates: list[int] = []
        for mid in hist_cands + als_cands + pop_cands:
            if mid not in seen:
                seen.add(mid)
                candidates.append(mid)

        if not candidates:
            continue

        # Per-user feature dict — keyed lookup by actual userId, not row position.
        # Falls back to mean user features for cold val users (no train history).
        # Empty dict {} would make ALL interaction features 0 → identical scores for all cold users.
        uid_int = int(uid)
        if uid_int in known_user_ids:
            user_feat_dict: dict = user_feat_indexed.loc[uid_int].to_dict()
        else:
            user_feat_dict = dict(mean_user_feat)  # copy so loop mutations don't bleed

        if debug_count < 5:
            logger.info(
                "[user feat lookup] user=%d  in_train=%s  "
                "log_total_ratings=%.3f  mean_rating=%.3f  genre_affinity_Drama=%.3f",
                uid_int,
                uid_int in known_user_ids,
                float(user_feat_dict.get("log_total_ratings", 0.0)),
                float(user_feat_dict.get("mean_rating", 0.0)),
                float(user_feat_dict.get("genre_affinity_Drama", 0.0)),
            )

        # ALS vectors for interaction feature injection
        u_vec = (
            als_arrays["user_factors"][als_arrays["user_id_to_idx"][int(uid)]]
            if als_arrays and int(uid) in als_arrays["user_id_to_idx"]
            else None
        )
        u_aff = np.array(
            [user_feat_dict.get(c, 0.0) for c in affinity_cols], dtype=np.float32
        ) if affinity_cols else None

        # Per-item feature dicts with injected interaction features
        item_feat_dicts: list[dict] = []
        for mid in candidates:
            mid_int = int(mid)
            if mid_int not in item_feat_indexed.index:
                continue
            irow: dict = item_feat_indexed.loc[mid_int].to_dict()

            # Inject (user, item) interaction features
            if u_vec is not None:
                m_idx = als_arrays["movie_id_to_idx"].get(mid_int)  # type: ignore[index]
                irow["mf_score"] = (
                    float(np.dot(u_vec, als_arrays["item_factors"][m_idx]))  # type: ignore[index]
                    if m_idx is not None else 0.0
                )
            if "mean_rating" in user_feat_dict and "avg_rating" in irow:
                irow["rating_gap"] = float(user_feat_dict["mean_rating"]) - float(irow["avg_rating"])
            if u_aff is not None:
                i_gen = np.array(
                    [irow.get(f"genre_{g}", 0.0) for g in genre_suffix], dtype=np.float32
                )
                dot = float(np.dot(u_aff, i_gen))
                aff_sum = float(u_aff.sum())
                irow["genre_overlap_score"] = float(dot / (aff_sum + 1e-8)) if aff_sum > 0 else 0.0
            item_feat_dicts.append(irow)

        if not item_feat_dicts:
            continue

        # Assemble feature matrix — one row per candidate, columns in feat_cols order
        feat_df = feature_store.assemble_inference_features(
            user_features=user_feat_dict,
            item_features_list=item_feat_dicts,
            request_context={"timestamp": int(_time.time())},
        )

        # Align candidates to only those that survived item_feat_indexed.index check
        valid_candidates = [
            int(mid) for mid in candidates if int(mid) in item_feat_indexed.index
        ]

        scores = ranker.predict(feat_df, feat_cols)

        # Rank ALL candidates — pass FULL ranked list to evaluator.
        # The evaluator slices to top-k internally when computing AP/NDCG.
        # Slicing here to [:k] causes overlap=0 when positives rank outside top-10.
        order = np.argsort(-scores)
        ranked = [int(valid_candidates[i]) for i in order]  # full list, no [:k] slice

        # Debug: overlap must be computed on full ranked list (not top-k)
        if debug_count < 5:
            overlap_full = len(set(ranked) & rel_items)
            overlap_topk = len(set(ranked[:k]) & rel_items)
            logger.info(
                "[two_stage eval] user=%d  pool=%d  val_positives=%d  "
                "overlap(full)=%d  overlap(top-%d)=%d",
                uid, len(ranked), len(rel_items), overlap_full, k, overlap_topk,
            )
            logger.info(
                "  scores: min=%.4f  max=%.4f  std=%.4f",
                float(scores.min()), float(scores.max()), float(scores.std()),
            )
            logger.info("  top-3 ranked movieIds: %s", ranked[:3])
            logger.info("  predictions[:5]: %s  val_positives[:5]: %s", ranked[:5], sorted(rel_items)[:5])
            debug_count += 1

        predictions.append(ranked)
        labels.append(rel_items)

    return predictions, labels


def _score_baseline(
    val_feat_df: pd.DataFrame,
    user_feat_df: pd.DataFrame,
    item_feat_df: pd.DataFrame,
    recommender_type: str,
    warm_user_ids: list[int],
    cfg,
    als_arrays: dict | None,
) -> tuple[list[list[int]], list[set[int]]]:
    """Score ALL catalog items per warm user using the baseline signal.

    Candidates = full item catalog from item_features.parquet.
    Labels     = val positives from val_feat_df.
    This gives a proper recommendation recall metric (~0.05-0.12 for popularity).
    """
    k = cfg.eval.k
    predictions: list[list[int]] = []
    labels: list[set[int]] = []

    # Val positives per user (from val_feat_df — no train contamination)
    user_positives: dict[int, set[int]] = (
        val_feat_df[val_feat_df["is_positive"] == 1]
        .groupby("userId")["movieId"]
        .apply(lambda x: set(x.astype(int)))
        .to_dict()
    )

    item_ids = item_feat_df["movieId"].astype(int).values  # (n_items,)

    # Item popularity scores (used by popularity and genre_pop)
    pop_scores = (
        item_feat_df["log_rating_count"].fillna(0.0).values
        if "log_rating_count" in item_feat_df.columns
        else np.zeros(len(item_feat_df))
    )

    # Precompute for genre_pop: item genre matrix and affinity column names
    affinity_cols = [c for c in user_feat_df.columns if c.startswith("genre_affinity_")]
    genre_suffix = [c[len("genre_affinity_"):] for c in affinity_cols]
    item_genre_mat: np.ndarray | None = None
    if recommender_type == "genre_pop" and affinity_cols:
        genre_item_cols = [f"genre_{g}" for g in genre_suffix if f"genre_{g}" in item_feat_df.columns]
        if genre_item_cols:
            item_genre_mat = item_feat_df[genre_item_cols].values.astype(np.float32)  # (n_items, n_genres)

    # User feature lookup (for genre_pop)
    user_feat_records: dict[int, dict] = {}
    if recommender_type == "genre_pop":
        user_feat_records = user_feat_df.set_index("userId").to_dict("index")

    for uid in warm_user_ids:
        rel_items = user_positives.get(int(uid), set())
        if not rel_items:
            continue

        if recommender_type == "popularity":
            top_idx = np.argsort(-pop_scores)[:k]
            ranked = item_ids[top_idx].tolist()

        elif recommender_type == "genre_pop":
            u_row = user_feat_records.get(int(uid), {})
            if item_genre_mat is not None and affinity_cols:
                u_aff = np.array([u_row.get(c, 0.0) for c in affinity_cols], dtype=np.float32)
                genre_scores = item_genre_mat @ u_aff  # (n_items,)
                combined = genre_scores * pop_scores
                top_idx = np.argsort(-combined)[:k]
                ranked = item_ids[top_idx].tolist()
            else:
                top_idx = np.argsort(-pop_scores)[:k]
                ranked = item_ids[top_idx].tolist()

        elif recommender_type in ("cf", "als"):
            if als_arrays is not None:
                u_idx = als_arrays["user_id_to_idx"].get(int(uid))
                if u_idx is None:
                    continue
                u_vec = als_arrays["user_factors"][u_idx]
                # Score every item in the catalog via dot product
                mf_scores = als_arrays["item_factors"] @ u_vec  # (n_als_items,)
                top_als_idx = np.argsort(-mf_scores)[:k]
                ranked = als_arrays["movie_id_map"][top_als_idx].tolist()
            else:
                logger.warning("ALS arrays missing for als/cf eval; using popularity fallback.")
                top_idx = np.argsort(-pop_scores)[:k]
                ranked = item_ids[top_idx].tolist()

        else:
            logger.warning("Unknown recommender_type '%s'; using popularity.", recommender_type)
            top_idx = np.argsort(-pop_scores)[:k]
            ranked = item_ids[top_idx].tolist()

        predictions.append(ranked)
        labels.append(rel_items)

    return predictions, labels


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


def _register_if_best(
    run_id: str,
    experiment_name: str,
    current_map: float,
    model_name: str,
) -> None:
    """Register the current run in MLflow Model Registry if it beats the best."""
    client = mlflow.tracking.MlflowClient()
    metric_key = "map_at_10"

    try:
        best_map = 0.0
        try:
            for mv in client.search_model_versions(f"name='{model_name}'"):
                v_map = client.get_run(mv.run_id).data.metrics.get(metric_key, 0.0)
                if v_map > best_map:
                    best_map = v_map
        except Exception:
            pass  # Model doesn't exist yet

        if current_map > best_map:
            click.echo(f"New best MAP@10 {current_map:.4f} > {best_map:.4f}. Registering …")
            try:
                client.create_registered_model(model_name)
            except Exception:
                pass  # Already exists
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/ranker",
                name=model_name,
            )
            click.echo(f"Model {model_name} v{result.version} registered.")
        else:
            click.echo(f"MAP@10 {current_map:.4f} did not beat best {best_map:.4f}. Not registering.")
    except Exception as exc:
        logger.warning("Model registration failed: %s", exc)


if __name__ == "__main__":
    main()
