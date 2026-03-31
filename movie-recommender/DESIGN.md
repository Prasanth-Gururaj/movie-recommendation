# Movie Recommender System — Design Document

**Version:** 2.0 (Final)
**Dataset:** MovieLens 25M
**Python:** 3.11
**Last updated:** March 2026

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Data Facts — Grounded in EDA](#2-data-facts--grounded-in-eda)
3. [System Architecture](#3-system-architecture)
4. [Data Storage Design](#4-data-storage-design)
5. [Data Preparation Pipeline](#5-data-preparation-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [OOP Class Hierarchy](#7-oop-class-hierarchy)
8. [Config System](#8-config-system)
9. [Model Design](#9-model-design)
10. [Evaluation Strategy](#10-evaluation-strategy)
11. [Cold-Start Handling](#11-cold-start-handling)
12. [MLflow Experiment Tracking](#12-mlflow-experiment-tracking)
13. [A/B Testing](#13-ab-testing)
14. [Drift Monitoring](#14-drift-monitoring)
15. [Caching Strategy](#15-caching-strategy)
16. [Inference Pipeline](#16-inference-pipeline)
17. [API Design](#17-api-design)
18. [Testing Strategy](#18-testing-strategy)
19. [Functional Requirements](#19-functional-requirements)
20. [Non-Functional Requirements](#20-non-functional-requirements)
21. [Deployment](#21-deployment)
22. [Library Decisions](#22-library-decisions)
23. [KPI Targets](#23-kpi-targets)
24. [Build Plan](#24-build-plan)
25. [Known Limitations](#25-known-limitations)

---

## 1. Project Goal

Build a production-style two-stage personalized movie recommendation system
on MovieLens 25M that:

- Retrieves 100–300 candidate movies per user via ALS + FAISS + popularity
- Ranks candidates using XGBoost/LightGBM Ranker trained on 4 feature groups
- Routes users by history depth: warm (≥20 positives), light (1–19), new (0)
- Handles cold items (58.8% of catalog has <10 ratings) via content fallback
- Reports honest two-track evaluation — warm and cold users always separate
- Tracks all 9 model variants in MLflow across 3 experiment groups
- Runs offline A/B test with deterministic hash bucketing + Mann-Whitney U
- Monitors drift on monthly 2018–2019 test slices (real data, not simulated)
- Serves via FastAPI with Redis caching, PostgreSQL feature store,
  Prometheus + Grafana observability
- Deploys locally via docker-compose (6 services, one command)

**One-line summary:**
Given a user's movie rating history, rank the top-10 unseen movies most likely
to be rated ≥ 4.0 — beating a popularity baseline by ≥20% on warm-user MAP@10
— while handling cold start, running A/B tests, and monitoring drift.

---

## 2. Data Facts — Grounded in EDA

Every design decision traces back to one of these numbers.

| Fact | Value | Design impact |
|---|---|---|
| Total ratings | 25,000,095 | Scale of training data |
| Users | 162,541 | Min 20 ratings — dataset pre-filtered |
| Movies (rated) | 59,047 | Candidate pool size |
| Matrix sparsity | 99.74% | Content features mandatory not optional |
| Relevance threshold | 4.0 | 49.8% positive — perfectly balanced labels |
| User activity mean/median | 153.8 / 71 | Right-skewed — log-transform count features |
| Users with <20 positives | 23.4% | Light-user routing threshold |
| Item popularity: top 1% | 47.6% of ratings | Extreme skew — diversity KPIs mandatory |
| Cold items (<10 ratings) | 58.8% of catalog | Cold-item fallback is primary path |
| Val users new to train | 63.1% | Two-track evaluation required |
| Test users new to train | 74.4% | Two-track evaluation required |
| Test items new to train | 39.6% | Content fallback required at eval time |
| Warm val users ≥3 positives | 83.2% (4,069 users) | Stable MAP@10 evaluation |
| Warm test users ≥3 positives | 85.5% (4,051 users) | Stable MAP@10 evaluation |
| Genre vector dimension | 18 | IMAX dropped — format not content |
| Release year parse rate | 99.3% | Safe feature — 412 missing → median |
| Genome coverage | 23.4% of rated movies | Supplementary signal not primary |
| Timestamp range | Jan 1995 – Nov 2019 | 24 years — rich temporal signal |
| Train / Val / Test split | ≤2016 / 2017 / ≥2018 | 83.2% / 6.8% / 10.0% |

---

## 3. System Architecture

### Local development stack

```
Training pipeline (runs offline)
  src/ingestion → src/features → src/candidates →
  src/ranking → src/evaluation → MLflow registry
          │
          │ model artifact + feature_columns.json
          ▼
Serving stack (docker-compose up)
  ┌─────────────────────────────────────────┐
  │  FastAPI  ──►  Redis (feature cache)    │
  │     │              └──► PostgreSQL      │
  │     └──►  MLflow registry (startup)     │
  │                                         │
  │  Prometheus ◄── /metrics               │
  │  Grafana    ◄── Prometheus             │
  └─────────────────────────────────────────┘
```

### Startup order (first run)

```
1. python train.py --config configs/experiments/xgb_full_tuned.yaml
2. python scripts/populate_feature_store.py
3. docker-compose up --build
4. python scripts/warm_cache.py   (optional — pre-warms Redis)
```

### AWS equivalent (documented, not required to run)

| Local | AWS equivalent |
|---|---|
| docker-compose | ECS Fargate + ALB |
| PostgreSQL container | RDS PostgreSQL |
| Redis container | ElastiCache Redis |
| MLflow local | MLflow on EC2 or S3 backend |
| Prometheus/Grafana | CloudWatch + Managed Grafana |
| Local image | ECR + ECS task definition |

ECS task definition in `infra/ecs_task_definition.json`.
Free cloud live link: deploy FastAPI to **Fly.io** (Docker-native, free tier).

---

## 4. Data Storage Design

### Four layers

| Layer | Store | Purpose | TTL |
|---|---|---|---|
| L1 — Raw files | data/raw/ | Never modified | Permanent |
| L2 — Processed files | data/processed/ (Parquet) | Training inputs | Regenerated per run |
| L3 — PostgreSQL | Feature store | API feature lookups | Updated by batch job |
| L4 — Redis | Feature cache | Low-latency serving | 24h user / 6h item |

### Processed files

```
data/processed/
  user_features.parquet
  item_features.parquet
  interaction_features.parquet
  train_pairs.parquet          # ≤ 2016
  val_pairs.parquet            # 2017 — warm users only
  test_pairs.parquet           # ≥ 2018 — warm users only
  candidate_pool.parquet       # warm items ≥10 ratings
  faiss_item_index.bin         # FAISS ANN index
  splits_metadata.json         # counts, dates, checksums
```

### PostgreSQL schema (key tables)

```sql
CREATE TABLE user_features (
    user_id           INTEGER PRIMARY KEY,
    user_tier         VARCHAR(10),      -- 'warm', 'light', 'new'
    total_ratings     INTEGER,
    positive_count    INTEGER,
    mean_rating       FLOAT,
    rating_variance   FLOAT,
    genre_affinity    JSONB,
    recent_genre_aff  JSONB,
    days_since_active INTEGER,
    activity_30d      INTEGER,
    mf_embedding      JSONB,
    updated_at        TIMESTAMP
);

CREATE TABLE item_features (
    movie_id          INTEGER PRIMARY KEY,
    title             TEXT,
    genres            TEXT[],
    genre_vector      JSONB,            -- 18-dim binary
    release_year      INTEGER,
    movie_age         INTEGER,
    total_ratings     INTEGER,
    avg_rating        FLOAT,
    popularity_pct    FLOAT,
    log_rating_count  FLOAT,
    genome_features   JSONB,            -- top-50 tags, null if unavailable
    has_genome        BOOLEAN,
    is_cold           BOOLEAN,
    mf_embedding      JSONB,
    updated_at        TIMESTAMP
);

CREATE TABLE user_rated_movies (
    user_id   INTEGER,
    movie_id  INTEGER,
    PRIMARY KEY (user_id, movie_id)
);
```

### Redis key schema

```
uf:{user_id}               user features JSON         TTL 24h
if:{movie_id}              item features JSON          TTL 6h
rated:{user_id}            Set of rated movie_ids      TTL 24h
cands:{user_id}            candidate pool JSON list    TTL 1h
recs:{user_id}:{variant}   final top-10 response JSON  TTL 30min
```

Redis memory budget: ~250MB total. Config: `maxmemory 512mb`, `allkeys-lru`.

---

## 5. Data Preparation Pipeline

Run in this exact order via `src/ingestion/`.

| Step | Script | What it does |
|---|---|---|
| 1 | cleaner.py | Drop 16 null tags. No other nulls. |
| 2 | cleaner.py | Parse release year (99.3%). Fill 412 missing with median. Compute movie_age. |
| 3 | cleaner.py | Build 18-dim genre multi-hot. Drop IMAX. Zero vector + flag for no-genre. |
| 4 | splitter.py | Temporal split: train ≤2016, val 2017, test ≥2018. Hardcoded in config. |
| 5 | feature_store.py | Binary labels: is_positive = (rating ≥ 4.0). Train only. |
| 6 | feature_store.py | User routing flags: warm/light/new from train interactions. |
| 7 | feature_store.py | Negative sampling: 1:4 ratio, warm item pool only, fixed seed. |
| 8 | feature_store.py | All 4 feature groups computed from train only. |
| 9 | feature_store.py | Save to data/processed/ as Parquet + checksums. |

### Skewness handling (at feature level — NOT raw data transforms)

| Skew | Feature treatment |
|---|---|
| User activity (mean 153, median 71) | log1p on count features. % not counts for genre affinity. |
| Item popularity (top 1% = 47.6%) | log1p on rating_count. Add popularity percentile rank. |
| Release year (heavy modern skew) | movie_age = 2019 − year. log1p(movie_age). Decade bin. |
| Tag count (mean 24.2, median 5) | log1p on tag_count. Top-50 genome tags by variance only. |
| Rating values | No transform — converting to binary 0/1 at threshold 4.0. |

---

## 6. Feature Engineering

### Group 1 — User features (src/features/user_features.py)

| Feature | Skew treatment | Why |
|---|---|---|
| log_total_ratings | log1p | How much history exists |
| log_positive_count | log1p | Strength of positive signal |
| mean_rating | none | Strict vs generous rater |
| rating_variance | none | Selective vs broad taste |
| genre_affinity_* (×18) | % not count | Core taste signal |
| recent_genre_aff_* (×18) | % 90-day window | Recent taste shift |
| days_since_active | log1p | Is user still active? |
| activity_30d | log1p | Short-term engagement |
| activity_90d | log1p | Medium-term engagement |

### Group 2 — Item features (src/features/item_features.py)

| Feature | Skew treatment | Why |
|---|---|---|
| log_rating_count | already log | Popularity signal |
| avg_rating | none | Overall quality |
| rating_variance | none | Polarising vs liked |
| genre_vector_* (×18) | none | Primary content signal |
| has_genre | none | Flag no-genre movies |
| movie_age | log1p | Era signal |
| decade_bin_* | one-hot | Era grouping |
| popularity_pct | none | Rank percentile 0–1 |
| recent_pop_30d | log1p | Trending signal |
| pop_trend | clip ±3σ | Rising vs fading |
| genome_tag_* (×50) | none | Semantic content (23% coverage) |
| has_genome | none | Coverage flag |
| is_cold | none | Cold item routing flag |

### Group 3 — User-item interaction features (src/features/interaction_features.py)

| Feature | Description |
|---|---|
| genre_overlap_score | dot(user_genre_affinity, item_genre_vector) |
| tag_profile_similarity | cosine(user_tag_profile, item_genome) |
| rating_gap | user_mean_rating − item_avg_rating |
| genre_history_count | # user positives sharing item's primary genre |
| mf_score | dot(user_mf_embedding, item_mf_embedding) |

### Group 4 — Time features (src/features/time_features.py)

| Feature | Description |
|---|---|
| interaction_month | Month of request (1–12) |
| interaction_dayofweek | Day of week (0–6) |
| days_since_user_active | request_time − user's last rating |
| days_since_item_rated | request_time − item's last rating |

Feature column list locked to `configs/feature_columns.json` after first run.
Inference must use `df.reindex(columns=feature_columns, fill_value=0.0)`.

---

## 7. OOP Class Hierarchy

```
BaseRecommender (ABC)
  ├── recommend(user_id, n) → List[Recommendation]  [abstract]
  ├── PopularityRecommender
  ├── GenrePopularityRecommender
  ├── CFRecommender
  ├── ALSRecommender
  └── TwoStageRecommender          ← main model
        ├── has-a: BaseCandidateGenerator
        └── has-a: BaseRanker

BaseRanker (ABC)
  ├── fit(X, y, groups)            [abstract]
  ├── predict(X) → np.ndarray      [abstract]
  ├── log_to_mlflow()              [inherited]
  ├── save_artifacts()             [inherited]
  ├── XGBRanker
  └── LGBMRanker                   ← polymorphic swap via config

BaseCandidateGenerator (ABC)
  ├── generate(user_id, n) → List[int]  [abstract]
  ├── PopularityCandidateGenerator
  ├── CFCandidateGenerator
  ├── ALSCandidateGenerator         ← uses FAISS index
  └── HybridCandidateGenerator      ← union of all three

BaseFeatureBuilder (ABC)
  ├── build(data) → pd.DataFrame    [abstract]
  ├── get_feature_names() → List[str]  [abstract]
  ├── validate_no_leakage()         [inherited]
  ├── UserFeatureBuilder
  ├── ItemFeatureBuilder
  ├── InteractionFeatureBuilder
  └── TimeFeatureBuilder

BaseEvaluator (ABC)
  ├── evaluate(predictions, labels) → EvalResult  [abstract]
  ├── RankingEvaluator
  └── DiversityEvaluator

BaseDriftMonitor (ABC)
  ├── compute_drift(baseline, current) → DriftReport  [abstract]
  ├── FeatureDriftMonitor
  ├── PredictionDriftMonitor
  └── PerformanceDriftMonitor

BaseConfig (dataclass)
  ├── from_yaml(path) → Self       [inherited]
  ├── validate()                   [inherited]
  ├── to_mlflow_params() → dict    [inherited]
  ├── DataConfig
  ├── FeatureConfig
  ├── XGBConfig / LGBMConfig
  ├── TrainingConfig
  ├── EvalConfig
  └── ExperimentConfig             ← master config
```

**Rules:**
- Inherit when classes share real behaviour (logging, validation, MLflow tracking)
- Compose when classes use each other (TwoStageRecommender has-a Ranker)
- RankerFactory.create(cfg) reads model_type from config → returns XGBRanker or LGBMRanker

---

## 8. Config System

One YAML per experiment in `configs/experiments/`.
`ExperimentConfig.from_yaml(path)` validates on load — bad values raise immediately.

```
configs/
  data_config.yaml          # shared: paths, split dates, thresholds
  feature_config.yaml       # shared: feature settings
  serving_config.yaml       # API, Redis, PostgreSQL
  genre_columns.json        # LOCKED: 18 genre names in order
  genome_tag_columns.json   # LOCKED: 50 tag IDs in order
  feature_columns.json      # GENERATED: after first training run
  experiments/
    baseline_popularity.yaml
    baseline_genre_pop.yaml
    baseline_cf.yaml
    baseline_mf.yaml
    xgb_user_item_only.yaml
    xgb_plus_interaction.yaml
    xgb_plus_time.yaml
    lgbm_full.yaml
    xgb_full_tuned.yaml     ← production candidate
    ab_test.yaml
```

Switching XGBoost → LightGBM = change one line: `model_type: lightgbm`.
`to_mlflow_params()` returns flat dict of all params as strings for `mlflow.log_params()`.

---

## 9. Model Design

### Stage 1 — Candidate retrieval

| Source | Method | Size |
|---|---|---|
| Popularity | Top-N globally + top-N per user's top genres | ~100 |
| CF | Item-item cosine similarity on user positives | ~100 |
| MF | ALS (implicit) + FAISS IndexFlatIP ANN search | ~100 |

Union → deduplicate → remove already-rated → cap at 300.

### Stage 2 — Learning-to-rank

- Objective: `rank:pairwise` (XGBoost) / `lambdarank` (LightGBM)
- Label: is_positive (0 or 1)
- Group: all candidates for one user = one ranking group
- Output: score per candidate → sort → top-10

### Nine model variants (in training order)

| Run | Features | Purpose |
|---|---|---|
| popularity_baseline | none | Floor to beat |
| genre_popularity | genre only | Improved baseline |
| cf_baseline | item-item CF score | CF ceiling |
| mf_baseline | MF score | MF ceiling |
| xgb_user_item | user + item | First real ranker |
| xgb_plus_interaction | + interaction | Isolate contribution |
| xgb_plus_time | + time | Isolate contribution |
| lgbm_full | all, LightGBM | Compare to XGBoost |
| xgb_full_tuned | all, tuned | Production candidate |

---

## 10. Evaluation Strategy

### Two-track — mandatory

**Track 1 — Warm user eval (primary)**
- 4,051 warm test users with ≥3 positives in 2018+ window
- Metrics: MAP@10, NDCG@10, Precision@10, Recall@10, MRR@10

**Track 2 — Cold user eval (secondary)**
- 13,775 new test users served by fallback
- Metrics: genre coverage, novelty, diversity of fallback recommendations

Never combine into a single number. Report both in README and MLflow.

### Diversity metrics

| Metric | Target | Why |
|---|---|---|
| Catalog coverage | >15% | Top 1% = 47.6% of ratings |
| Intra-list diversity | >0.6 | Prevent 10-Drama lists |
| Novelty | >baseline | Long-tail exposure |
| Cold-item exposure | >5% | 58.8% catalog is cold |

---

## 11. Cold-Start Handling

### User routing

```
positive_count ≥ 20  →  WARM:  full two-stage ranker + all features
positive_count 1–19  →  LIGHT: simplified ranker, content-heavy, 100 candidates
positive_count = 0   →  NEW:   onboarding → genre preferences → popularity-within-genre
```

### Item routing

```
rating_count ≥ 10    →  WARM ITEM:  full ranker with CF + content features
rating_count < 10    →  COLD ITEM:  genre overlap + genome similarity + ε=0.05 boost
has_genre = False    →  NO-GENRE:   zero vector → global popularity fallback
```

### At evaluation time

- 39.6% of test items are new to training → content fallback scores them
- Report cold-item MAP@10 separately to show fallback quality honestly

---

## 12. MLflow Experiment Tracking

### Three experiment groups

| Group | Runs | Purpose |
|---|---|---|
| baselines | 4 baseline models | Set the floor to beat |
| ranker_iterations | 5 ranker variants | Isolate feature group contributions |
| drift_evaluation | 5 monthly test slices | MAP@10 over time chart |

### Per-run logging

**Parameters (14):** model_type, n_estimators, learning_rate, max_depth,
min_child_weight, subsample, colsample_bytree, feature_set_version,
train_end_year, relevance_threshold, cold_user_threshold, cold_item_threshold,
candidate_pool_size, negative_sample_ratio

**Metrics — warm track:** warm_map_at_10, warm_ndcg_at_10, warm_precision_at_10,
warm_recall_at_10, warm_mrr_at_10

**Metrics — diversity:** catalog_coverage, intra_list_diversity, novelty_score,
cold_item_exposure_rate

**Metrics — system:** inference_latency_p95_ms, training_duration_s, model_size_mb

**Artifacts:** model.pkl, feature_columns.json, genre_columns.json,
genome_tag_columns.json, configs/model_config.yaml, eval_report.json,
feature_importance.png

### Registry lifecycle

```
Staging    → new model awaiting review
Production → currently loaded by API
Archived   → previous production models

API loads: mlflow.pyfunc.load_model("models:/recommender/Production")
Rollback: promote any Archived version to Production — zero code change
```

---

## 13. A/B Testing

### Setup

- Control (bucket 0–49): popularity baseline
- Treatment (bucket 50–99): full XGBoost ranker
- Assignment: `int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100`
- Deterministic — same user always gets same bucket

### Measurement

- Per-user MAP@10 computed for both groups on 2018–2019 warm test users
- Statistical test: Mann-Whitney U (non-parametric, correct for MAP distributions)
- Significance threshold: p < 0.05
- Report: p-value, effect size %, 95% confidence interval

### Logged to MLflow (experiment: ab_test)

```
Parameters: control_model, treatment_model, test_period, n_users_per_group
Metrics: control_map_at_10, treatment_map_at_10, p_value, effect_size_pct,
         is_significant, control_n_users, treatment_n_users
Artifacts: ab_test_report.json, per_user_scores.parquet, map_distribution_plot.png
```

### API integration

Bucket assigned at request time → routes to different model → logs variant
to Prometheus counter `recommender_requests_total{ab_variant}`.

---

## 14. Drift Monitoring

### Three drift categories

**Input drift** (src/monitoring/feature_drift.py)
- KS test: genre distribution, user activity, item popularity — monthly windows
- Baseline: 2016 training period
- Alert: KS statistic > 0.1

**Prediction drift** (src/monitoring/prediction_drift.py)
- Score distribution, top-item concentration, cold-item exposure — monthly
- Compare val-period baseline vs each monthly test window

**Performance drift** (src/monitoring/performance_drift.py)
- MAP@10 on monthly slices: 2018 Q1, Q2, Q3, Q4, 2019
- Output: MAP@10-over-time chart → key README artifact
- Retraining trigger: MAP@10 drops >10% from val baseline

### Output

```
reports/drift/
  feature_drift_report.html
  prediction_drift_report.html
  performance_drift_report.html
  drift_summary.json
```

---

## 15. Caching Strategy

### Latency breakdown

| Without cache | With cache |
|---|---|
| PostgreSQL user lookup: 5ms | Redis user GET: 0.5ms |
| PostgreSQL ×300 item lookups: 1,500ms | Redis MGET ×300: 2ms |
| XGBoost inference: 80ms | XGBoost inference: 80ms (uncacheable) |
| **Total: ~1,510ms** | **Total: ~89ms** |

### Critical pattern — always MGET not loop

```python
# ONE round-trip for all 300 candidates
keys = [f"if:{mid}" for mid in movie_ids]
cached = await redis.mget(*keys)   # 2ms

# NOT 300 individual GETs = 900ms
```

### Cache invalidation

- Batch pipeline writes new values with refreshed TTLs (SET key value EX ttl)
- Never delete all keys at once — causes stampede
- Redis eviction: allkeys-lru — cold users evicted naturally
- On Redis down: serve from PostgreSQL, log error, alert Prometheus

### L1 — Process memory (most important)

Model loaded once at startup via FastAPI lifespan pattern. Stored in `app.state`.
If MLflow unreachable at startup → fail fast (container restarts).
Startup health check: run one dummy inference to confirm model works.

---

## 16. Inference Pipeline

### Request flow (10 steps)

```
1.  Receive user_id
2.  Determine AB bucket: hash(user_id) % 100
3.  Determine user tier: warm / light / new (from feature store)
4.  Lookup user features: Redis → PostgreSQL fallback
5.  Generate candidates: popularity + CF + FAISS MF, union, cap 300
6.  Lookup item features: Redis MGET for all candidates at once
7.  Assemble inference DataFrame via InferenceFeatureAssembler
8.  Score with loaded model (or content fallback if cold user/item)
9.  Sort, exclude rated movies, take top-10, attach reason codes
10. Log to Prometheus: variant, tier, latency, cache hit/miss
```

### InferenceFeatureAssembler (src/api/inference_assembler.py)

Critical class — ensures inference DataFrame matches training shape exactly.

```python
df = pd.DataFrame(rows)
df = df.reindex(columns=self.feature_columns, fill_value=0.0)
# missing columns → 0.0, extra columns → dropped
# column order locked by feature_columns.json from MLflow artifact
```

### Graceful degradation

```
Model fails at startup     → FAIL FAST — container restarts
Model fails at inference   → popularity fallback, log error, never 500
Redis down                 → PostgreSQL for all lookups, log error
PostgreSQL down            → in-memory emergency cache (60s), then fail
User not in system         → new user path, onboarding fallback
```

---

## 17. API Design

### Endpoints

```
POST /recommend/{user_id}    top-10 recommendations
GET  /health                 service status
GET  /metrics                Prometheus-format metrics
GET  /model-info             model version, run ID, val MAP@10
```

### Response shape

```json
{
  "user_id": 1234,
  "user_tier": "warm",
  "ab_variant": "treatment",
  "recommendations": [
    {
      "rank": 1,
      "movie_id": 318,
      "title": "The Shawshank Redemption (1994)",
      "score": 0.847,
      "genres": ["Drama"],
      "reason_code": "matches your Drama preference"
    }
  ],
  "model_version": "3",
  "latency_ms": 94
}
```

### Pydantic validation (automatic 422 on bad input)

- `n` must be ≥1 and ≤50
- `user_tier` must be Literal["warm", "light", "new"]
- `ab_variant` must be Literal["control", "treatment"]
- `score` must be 0.0–1.0
- `val_map_at_10 > 0.99` triggers leakage warning validator

### Prometheus metrics

```
recommender_requests_total{ab_variant, user_tier}
recommender_latency_seconds{quantile}
recommender_errors_total{error_type}
cache_hits_total{key_type}
cache_misses_total{key_type}
recommender_model_version
```

---

## 18. Testing Strategy

### 11 test files

| File | What it tests |
|---|---|
| conftest.py | Shared fixtures: toy dataset, mock redis, mock postgres, TestClient |
| test_config.py | Config validation: invalid values raise, YAML round-trips, MLflow params are strings |
| test_schemas.py | Pydantic: 422 on bad input, boundary values, leakage validator |
| test_features.py | No leakage, log transforms, genre dim=18, affinity sums to 1 |
| test_inference_assembler.py | Training shape == inference shape, no NaN, log features applied |
| test_ranking_metrics.py | MAP, NDCG hand-verified on toy examples, perfect=1.0 |
| test_cold_start.py | User routing, cold item scoring, AB bucketing deterministic |
| test_candidate_generator.py | ≤300 results, rated excluded, output deduped |
| test_cache.py | Redis hit/miss/fallback, MGET batch, Redis-down → PostgreSQL |
| test_data_pipeline.py | Temporal split correctness, no val/test rows in train |
| test_api.py | Endpoint schemas, unknown user 200, rated excluded, AB deterministic |

### Coverage gate: 80% minimum (enforced in CI)

---

## 19. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Given userId, return top-10 ranked unseen movies (rating ≥ 4.0 threshold) | Must |
| FR-2 | Two-stage pipeline: retrieval (100–300) → learning-to-rank | Must |
| FR-3 | Route users: warm / light / new with appropriate fallback | Must |
| FR-4 | Cold items (<10 ratings): content scoring via genre + genome | Must |
| FR-5 | New API users: onboarding → genre preferences → popularity fallback | Must |
| FR-6 | Two-track eval: warm (4,051) and cold (13,775) always separate | Must |
| FR-7 | A/B test: hash bucketing + Mann-Whitney U + API routing | Must |
| FR-8 | MLflow: all 9 variants logged with params, metrics, artifacts | Must |
| FR-9 | MLflow registry: production model loaded by name at startup | Must |
| FR-10 | Drift: input + prediction + performance on monthly test slices | Must |
| FR-11 | API: /recommend, /health, /metrics, /model-info | Must |
| FR-12 | Response includes score, reason code, AB variant, user tier | Must |
| FR-13 | Retraining trigger: MAP@10 drops >10% from val baseline | Should |
| FR-14 | No-genre movies (8.1%) → popularity fallback, not content scoring | Should |
| FR-15 | AWS architecture documented in infra/ even if not deployed | Should |

---

## 20. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | P95 API latency (warm user, cached) | <200ms |
| NFR-2 | P95 API latency (cold user, fallback) | <50ms |
| NFR-3 | API uptime | >99% |
| NFR-4 | Graceful degradation — always return 200 | Never 500 |
| NFR-5 | Batch feature pipeline runtime | <2 hours |
| NFR-6 | Every training run reproducible from run ID | 100% |
| NFR-7 | All random seeds fixed | numpy, sklearn, xgboost |
| NFR-8 | Docker image builds cleanly from scratch | CI check |
| NFR-9 | Secrets via environment variables only | Zero in code |
| NFR-10 | Raw data never modified | Checksums stable |
| NFR-11 | Feature column order locked per model version | feature_columns.json |
| NFR-12 | Model loaded from registry — no hardcoded paths | Registry name only |

---

## 21. Deployment

### docker-compose services (6)

| Service | Image | Purpose |
|---|---|---|
| api | your FastAPI image | Recommendation service |
| postgres | postgres:15 | Feature store |
| redis | redis:7 | Feature cache |
| mlflow | ghcr.io/mlflow/mlflow | Experiment tracking + registry |
| prometheus | prom/prometheus | Metrics scraping |
| grafana | grafana/grafana | Dashboards |

### Environment variables (.env.example)

```
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_MODEL_NAME=recommender
MLFLOW_MODEL_STAGE=Production
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=recommender
POSTGRES_USER=recommender
POSTGRES_PASSWORD=changeme
REDIS_HOST=redis
REDIS_PORT=6379
API_HOST=0.0.0.0
API_PORT=8000
COLD_USER_THRESHOLD=20
COLD_ITEM_THRESHOLD=10
RELEVANCE_THRESHOLD=4.0
DATA_PROCESSED_DIR=data/processed
FAISS_INDEX_PATH=data/processed/faiss_item_index.bin
```

### Free live link (no cost)

Deploy FastAPI container to Fly.io: `fly launch --dockerfile Dockerfile && fly deploy`

---

## 22. Library Decisions

| Component | Library | Reason |
|---|---|---|
| Matrix factorization | implicit==0.7.2 (ALS) | Built for sparse implicit data — 99.74% sparsity |
| ANN candidate retrieval | faiss-cpu==1.8.0 (IndexFlatIP) | Inner product search over 59k item embeddings |
| Learning-to-rank | xgboost==2.1.4 + lightgbm==4.5.0 | Both trained, best registered in MLflow |
| Feature store serving | asyncpg==0.29.0 | Async PostgreSQL — non-blocking in FastAPI |
| Redis client | redis==5.0.8 (async) | MGET pipeline support |
| API framework | fastapi==0.111.1 + uvicorn==0.30.3 | Async, Pydantic, OpenAPI docs free |
| Experiment tracking | mlflow==2.14.3 | Tracking + registry + artifact storage |
| Statistical test | scipy.stats.mannwhitneyu | Non-parametric, correct for MAP@10 |
| Data processing | pandas==2.2.2 + pyarrow==15.0.2 | Parquet read/write |
| Linting | ruff==0.4.10 | Replaces flake8 + isort + pyupgrade |
| Type checking | mypy==1.10.1 | Enforces type annotations |
| Testing | pytest==8.2.2 + pytest-cov==5.0.0 | 80% coverage gate |

---

## 23. KPI Targets

### Primary (model must beat these to ship)

| Metric | Target | Baseline |
|---|---|---|
| warm_map_at_10 | ≥0.12 | Popularity ~0.081 |
| warm_ndcg_at_10 | ≥0.18 | Popularity ~0.124 |
| warm_precision_at_10 | ≥0.25 | Popularity ~0.162 |
| warm_recall_at_10 | ≥0.15 | Popularity ~0.10 |

### Diversity (prevents popularity collapse)

| Metric | Target |
|---|---|
| Catalog coverage | >15% |
| Intra-list diversity | >0.6 |
| Cold-item exposure | >5% |

### A/B test

| Metric | Target |
|---|---|
| MAP@10 lift (treatment vs control) | >20% relative |
| p-value | <0.05 |

### Operational

| Metric | Target |
|---|---|
| P95 latency (warm) | <200ms |
| API success rate | >99% |
| Retraining trigger | MAP@10 drops >10% |

---

## 24. Build Plan

### Phase 1 — Config + ingestion (Week 1)
- src/config/ — all 7 config classes with validation
- src/ingestion/ — loader, cleaner, splitter
- Verify: parquet files written, checksums in splits_metadata.json

### Phase 2 — Feature engineering (Week 2)
- src/features/ — all 4 feature builders + feature_store.py
- Verify: no leakage, feature_columns.json generated, tests pass

### Phase 3 — Candidates + baselines (Week 3)
- src/candidates/ — popularity, CF, ALS+FAISS, hybrid
- src/ranking/baselines.py — 4 baseline runs logged to MLflow

### Phase 4 — Ranker (Week 4)
- src/ranking/ — XGBRanker, LGBMRanker, RankerFactory, TwoStageRecommender
- Train all 9 variants, log to MLflow, register best model

### Phase 5 — Cold-start + A/B (Week 5)
- src/ranking/cold_start.py — routing logic
- src/monitoring/ab_testing.py — hash bucketing, Mann-Whitney U
- Verify two-track evaluation produces honest separate numbers

### Phase 6 — API + caching (Week 6)
- src/api/ — all files including InferenceFeatureAssembler
- scripts/populate_feature_store.py
- scripts/warm_cache.py
- Verify P95 latency <200ms with Redis warmed

### Phase 7 — Monitoring + deployment (Week 7)
- src/monitoring/ — all 3 drift monitors + reporter
- docker-compose with all 6 services
- Grafana dashboard
- CI/CD pipelines tested

### Phase 8 — Polish (Week 8)
- README: architecture diagram, results table, drift chart, A/B result
- Screen-recorded demo
- Fly.io live link (optional)

---

## 25. Known Limitations

| Limitation | Why acceptable |
|---|---|
| No real-time user feedback loop | MovieLens is static — batch nightly update is realistic |
| Offline A/B testing only | No live traffic — offline simulation is standard for portfolio |
| Fixed ε=0.05 exploration boost | Simple but honest — not a full bandit algorithm |
| No managed feature store (Feast) | PostgreSQL is production-style — Feast is future work |
| Manual retraining | Trigger defined, execution manual — Airflow/Prefect is future work |
| links.csv not used | IMDB/TMDB enrichment not in scope — would add complexity without core value |