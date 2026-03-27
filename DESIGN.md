# Movie Recommender System — Design Document

**Version:** 1.0  
**Status:** Active  
**Dataset:** MovieLens 25M  
**Author:** [Your name]  
**Last updated:** March 2026

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Data Facts — Grounded in EDA](#2-data-facts--grounded-in-eda)
3. [System Architecture](#3-system-architecture)
4. [Data Storage Design](#4-data-storage-design)
5. [Data Preparation Pipeline](#5-data-preparation-pipeline)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Design](#7-model-design)
8. [Evaluation Strategy](#8-evaluation-strategy)
9. [Cold-Start Handling](#9-cold-start-handling)
10. [MLflow Experiment Tracking](#10-mlflow-experiment-tracking)
11. [A/B Testing](#11-ab-testing)
12. [Drift Monitoring](#12-drift-monitoring)
13. [API Design](#13-api-design)
14. [Functional Requirements](#14-functional-requirements)
15. [Non-Functional Requirements](#15-non-functional-requirements)
16. [Deployment](#16-deployment)
17. [Build Plan](#17-build-plan)
18. [KPI Targets](#18-kpi-targets)

---

## 1. Project Goal

Build a production-style two-stage personalized movie recommendation system on
MovieLens 25M that:

- Retrieves 100–300 candidate movies per user then ranks them with a
  learning-to-rank model (XGBoost/LightGBM Ranker)
- Routes users by history depth: warm (≥20 positives), light (1–19), new (0)
- Handles cold items (58.8% of catalog has <10 ratings) via content fallback
- Reports **honest two-track evaluation** — warm and cold users always separate
- Tracks every experiment in MLflow with full parameter, metric, and artifact logging
- Runs a **simulated A/B test** with deterministic hash-based user bucketing
- Monitors drift using monthly slices of the 2018–2019 test period
- Serves via a Dockerized FastAPI with PostgreSQL + Redis + Prometheus + Grafana

**One-line summary:**  
Given a user's movie rating history, rank the top-10 unseen movies most likely to
be rated ≥ 4.0 — beating a popularity baseline by ≥20% on warm-user MAP@10 —
while handling cold start, running A/B tests, and monitoring drift over time.

---

## 2. Data Facts — Grounded in EDA

Every design decision below traces back to one of these numbers.

| Fact | Value | Design impact |
|---|---|---|
| Total ratings | 25,000,095 | Scale of training data |
| Users | 162,541 | Min 20 ratings each — dataset pre-filtered |
| Movies (rated) | 59,047 | Candidate pool size |
| Matrix sparsity | 99.74% | Content features are mandatory, not optional |
| Relevance threshold | 4.0 | Gives 49.8% positive — perfectly balanced labels |
| Positive rate | 49.8% / 50.2% | No class weighting needed |
| User activity (mean/median) | 153.8 / 71 | Right-skewed — use log transforms on count features |
| Users with <20 positives | 23.4% | Light-user routing threshold |
| Item popularity: top 1% | 47.6% of ratings | Extreme skew — diversity KPIs mandatory |
| Cold items (<10 ratings) | 58.8% of catalog | Cold-item fallback is a primary path, not edge case |
| Val users new to train | 63.1% | Two-track evaluation required |
| Test users new to train | 74.4% | Two-track evaluation required |
| Test items new to train | 39.6% | Content fallback required at eval time |
| Warm val users with ≥3 positives | 83.2% (4,069 users) | Stable MAP@10 evaluation |
| Warm test users with ≥3 positives | 85.5% (4,051 users) | Stable MAP@10 evaluation |
| Genre vector dimension | 18 | IMAX dropped — format not content |
| Release year parse rate | 99.3% | Safe feature — 412 missing filled with median |
| Genome coverage | 23.4% of rated movies | Supplementary signal, not primary |
| Timestamp range | Jan 1995 – Nov 2019 | 24 years — rich temporal signal |
| Train / Val / Test split | ≤2016 / 2017 / ≥2018 | 83.2% / 6.8% / 10.0% |

---

## 3. System Architecture

### Local development stack (docker-compose)

```
┌─────────────────────────────────────────────────────────────┐
│  Training pipeline (runs offline)                           │
│  src/ingestion → src/features → src/candidates →           │
│  src/ranking → src/evaluation → MLflow registry            │
└────────────────────────┬────────────────────────────────────┘
                         │ model artifact + feature_columns.json
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Serving stack (docker-compose up)                          │
│                                                             │
│  FastAPI ──► Redis (feature cache)                         │
│     │            └──► PostgreSQL (feature store)           │
│     │                                                       │
│     └──► MLflow (model registry — loads at startup)        │
│                                                             │
│  Prometheus ◄── /metrics (FastAPI)                         │
│  Grafana    ◄── Prometheus                                  │
└─────────────────────────────────────────────────────────────┘
```

### AWS equivalent (documented, not required to run)

| Local service | AWS equivalent |
|---|---|
| docker-compose | ECS Fargate + ALB |
| PostgreSQL container | RDS PostgreSQL |
| Redis container | ElastiCache Redis |
| MLflow local | MLflow on EC2 or S3 backend |
| Prometheus/Grafana | CloudWatch + Managed Grafana |
| Local image | ECR + ECS task definition |

The ECS task definition JSON lives in `infra/ecs_task_definition.json`.
The architecture diagram in `architecture.png` shows both the local and AWS
equivalent side by side.

### Free cloud deployment option

For a live link without AWS cost: deploy the FastAPI container to **Fly.io**
(free tier, Docker-native, closest to ECS experience). MLflow and monitoring
can stay local — they don't need to be public.

---

## 4. Data Storage Design

### Four storage layers — each has a distinct job

#### Layer 1 — Raw files (`data/raw/`)
Never modified after download. Single source of truth.

```
data/raw/
  ratings.csv          # userId, movieId, rating, timestamp
  movies.csv           # movieId, title, genres
  tags.csv             # userId, movieId, tag, timestamp
  genome-scores.csv    # movieId, tagId, relevance
  genome-tags.csv      # tagId, tag
```

#### Layer 2 — Processed files (`data/processed/`)
Parquet files produced by `src/ingestion/` and `src/features/`.
These are the inputs to all training and evaluation code.

```
data/processed/
  user_features.parquet        # one row per userId
  item_features.parquet        # one row per movieId
  interaction_features.parquet # one row per (userId, movieId) training pair
  train_pairs.parquet          # (userId, movieId, label, features) ≤ 2016
  val_pairs.parquet            # 2017 — warm users only
  test_pairs.parquet           # ≥ 2018 — warm users only
  candidate_pool.parquet       # pre-filtered warm items (≥10 ratings)
  splits_metadata.json         # split dates, counts, checksums
```

#### Layer 3 — PostgreSQL (serving feature store)
Holds precomputed features for the API to look up at request time.
Populated by a batch job after training is complete.

```sql
-- user_features table
CREATE TABLE user_features (
    user_id            INTEGER PRIMARY KEY,
    user_tier          VARCHAR(10),     -- 'warm', 'light', 'new'
    total_ratings      INTEGER,
    positive_count     INTEGER,
    mean_rating        FLOAT,
    rating_variance    FLOAT,
    genre_affinity     JSONB,           -- {Drama: 0.42, Comedy: 0.18, ...}
    recent_genre_aff   JSONB,           -- last 90 days
    days_since_active  INTEGER,
    activity_30d       INTEGER,
    activity_90d       INTEGER,
    mf_embedding       JSONB,           -- list of floats
    updated_at         TIMESTAMP
);

-- item_features table
CREATE TABLE item_features (
    movie_id           INTEGER PRIMARY KEY,
    title              TEXT,
    genres             TEXT[],          -- ['Drama', 'Comedy']
    genre_vector       JSONB,           -- 18-dim binary vector
    release_year       INTEGER,
    movie_age          INTEGER,
    decade_bin         VARCHAR(10),
    total_ratings      INTEGER,
    avg_rating         FLOAT,
    rating_variance    FLOAT,
    log_rating_count   FLOAT,
    popularity_pct     FLOAT,           -- percentile rank
    recent_pop_30d     INTEGER,
    recent_pop_90d     INTEGER,
    pop_trend          FLOAT,
    tag_count          INTEGER,
    genome_features    JSONB,           -- top-50 tag scores, null if unavailable
    has_genome         BOOLEAN,
    is_cold            BOOLEAN,         -- rating_count < 10
    mf_embedding       JSONB,
    updated_at         TIMESTAMP
);

-- rated_movies table (for excluding already-seen items)
CREATE TABLE user_rated_movies (
    user_id   INTEGER,
    movie_id  INTEGER,
    PRIMARY KEY (user_id, movie_id)
);
```

#### Layer 4 — Redis (feature cache)
Caches user_features and item_features for frequently requested users/items.
TTL: 24 hours. Reduces PostgreSQL load and keeps P95 latency < 200ms.

```
Key format:
  user_features:{user_id}     → JSON blob, TTL 24h
  item_features:{movie_id}    → JSON blob, TTL 24h
  rated_movies:{user_id}      → set of movie_ids, TTL 24h
```

#### MLflow artifact storage (`mlruns/` or S3 backend)
All model artifacts, configs, and evaluation reports stored by run ID.
The API loads the production model by registry name at startup:
```python
mlflow.pyfunc.load_model("models:/recommender/Production")
```

---

## 5. Data Preparation Pipeline

Run in this exact order. Each step is a script in `src/ingestion/`.

### Step 1 — Clean (`cleaner.py`)
- Drop 16 null rows in tags.csv — only file with nulls
- No other nulls, no duplicates across any file
- Verify all movieIds in ratings exist in movies.csv (already confirmed clean)

### Step 2 — Parse release year (`cleaner.py`)
- Regex extract year from title: `r'\((\d{4})\)\s*$'`
- 99.3% parse rate (62,011 of 62,423 movies)
- Fill 412 missing with median year
- Compute `movie_age = 2019 - release_year`
- Compute `decade_bin`: '1970s', '1980s', '1990s', '2000s', '2010s', 'pre-1970'

### Step 3 — Build genre multi-hot (`cleaner.py`)
- 18 clean genres (drop IMAX — format not content, only 195 movies)
- Final genre list: Action, Adventure, Animation, Children, Comedy, Crime,
  Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance,
  Sci-Fi, Thriller, War, Western
- No-genre movies (8.1%): zero vector + `has_genre = False` flag
- Fixed column order saved to `configs/genre_columns.json`

### Step 4 — Temporal split (`splitter.py`)
- Train: `year <= 2016`  → 20,798,765 ratings (83.2%)
- Val:   `year == 2017`  →  1,689,935 ratings  (6.8%)
- Test:  `year >= 2018`  →  2,511,395 ratings (10.0%)
- Dates hardcoded in `configs/data_config.yaml` — never recomputed at runtime

### Step 5 — Binary labels
- `is_positive = (rating >= 4.0).astype(int)`
- Applied to train only — val/test labels held out during training
- 49.8% positive / 50.2% negative — no class weighting needed

### Step 6 — User routing flags (`feature_store.py`)
- Computed from train interactions only
- `warm`:  positive_count >= 20  (76.6% of users)
- `light`: 1 <= positive_count < 20  (23.3% of users)
- `new`:   positive_count == 0  (0.1% — 199 users in training data)

### Step 7 — Negative sampling
- For each warm/light user in train: sample 4 unseen movies per positive (1:4 ratio)
- Sample from warm item pool only (rating_count >= 10)
- Deterministic sampling using fixed seed per userId

### Step 8 — Skewness handling
Do NOT transform raw ratings. Handle skewness at feature level:

| Skew source | Feature treatment |
|---|---|
| User activity (mean 153, median 71) | Log-transform count features. Use % for genre affinity. |
| Item popularity (top 1% = 47.6%) | Log-transform rating_count. Add popularity percentile rank. |
| Release year (heavy right skew to modern) | Use movie_age. Log-transform movie_age. Add decade bin. |
| Tag count (mean 24.2, median 5) | Log-transform. Use top-50 genome tags by variance. |
| Rating values | No transform needed — converting to binary 0/1. |

### Step 9 — Save processed files
All outputs saved to `data/processed/` as Parquet with Snappy compression.
`splits_metadata.json` records row counts, date ranges, and SHA256 checksums.

---

## 6. Feature Engineering

Four groups. Each group is a separate file in `src/features/`.

### Group 1 — User features (`user_features.py`)
Computed from train interactions only (no leakage).

| Feature | Type | Skew treatment | Why |
|---|---|---|---|
| `total_ratings` | int | log1p | How much history exists |
| `positive_count` | int | log1p | Strength of positive signal |
| `mean_rating` | float | none | Strict vs generous rater |
| `rating_variance` | float | none | Selective vs broad taste |
| `genre_affinity_*` | float x18 | pct not count | Core taste signal |
| `recent_genre_aff_*` | float x18 | pct, 90-day window | Recent taste shift |
| `days_since_active` | int | log1p | Is user still active? |
| `activity_30d` | int | log1p | Short-term engagement |
| `activity_90d` | int | log1p | Medium-term engagement |
| `user_tier` | str | one-hot | warm / light routing |

### Group 2 — Item features (`item_features.py`)
Computed from train interactions only.

| Feature | Type | Skew treatment | Why |
|---|---|---|---|
| `log_rating_count` | float | already log | Popularity signal |
| `avg_rating` | float | none | Overall quality |
| `rating_variance` | float | none | Polarising vs liked |
| `genre_vector_*` | int x18 | none | Primary content signal |
| `has_genre` | bool | none | Flag for no-genre movies |
| `movie_age` | int | log1p | Era signal |
| `decade_bin_*` | int | one-hot | Era grouping |
| `popularity_pct` | float | none | Rank percentile 0–1 |
| `recent_pop_30d` | int | log1p | Trending signal |
| `pop_trend` | float | clip ±3σ | Rising vs fading |
| `genome_tag_*` | float x50 | none | Semantic content (23% coverage) |
| `has_genome` | bool | none | Coverage flag |
| `is_cold` | bool | none | Cold item flag |
| `log_tag_count` | float | already log | Metadata richness |

Genome features: select top-50 tags by variance across all movies.
Tag IDs saved to `configs/genome_tag_columns.json`.

### Group 3 — User-item interaction features (`interaction_features.py`)
These are typically the strongest signals. Computed at training pair creation time.

| Feature | Description | Why |
|---|---|---|
| `genre_overlap_score` | dot(user_genre_affinity, item_genre_vector) | Taste-content fit |
| `tag_profile_similarity` | cosine(user_tag_profile, item_genome) | Semantic fit |
| `rating_gap` | user_mean_rating - item_avg_rating | Is item above user's bar? |
| `genre_history_count` | # user positives sharing item's primary genre | Genre evidence depth |
| `mf_score` | dot(user_mf_embedding, item_mf_embedding) | CF signal |
| `pop_affinity` | user_activity_level × item_popularity | Active users + popular items |

### Group 4 — Time/context features (`time_features.py`)

| Feature | Description |
|---|---|
| `interaction_month` | Month of simulated request (1–12) |
| `interaction_dayofweek` | Day of week (0–6) |
| `days_since_user_active` | request_time - user's last rating timestamp |
| `days_since_item_rated` | request_time - item's last rating timestamp |

### Feature assembly (`feature_store.py`)
- Joins all four groups into a single training matrix
- Final column list saved to `configs/feature_columns.json`
- API loads `feature_columns.json` from the same MLflow run as the model
- Column order is locked — inference must match training exactly

---

## 7. Model Design

### Stage 1 — Candidate retrieval (`src/candidates/`)

Generate 100–300 candidates per user using three sources, then union and deduplicate:

| Source | Method | Size |
|---|---|---|
| Popularity filter | Top-N globally + top-N per user's top genres | ~100 |
| CF neighbors | Item-item cosine similarity on user's positives | ~100 |
| MF similarity | ANN search on user MF embedding (FAISS or sklearn) | ~100 |

Remove items the user has already rated. Cap at 300 candidates.

### Stage 2 — Learning-to-rank (`src/ranking/`)

Train an XGBoost Ranker or LightGBM Ranker on (user, candidate) pairs.

- **Objective:** `rank:pairwise` (XGBoost) or `lambdarank` (LightGBM)
- **Label:** `is_positive` (0 or 1)
- **Group:** all candidates for a user form one ranking group
- **Features:** all four groups joined (see Feature Engineering)
- **Output:** score per candidate → sort descending → top-10

### Model variants to train (in order)

| Run name | Features included | Purpose |
|---|---|---|
| `popularity_baseline` | none — just rank by popularity | Floor to beat |
| `genre_popularity` | genre only | Improved baseline |
| `cf_baseline` | item-item CF score only | CF-only ceiling |
| `mf_baseline` | MF embedding score only | MF-only ceiling |
| `xgb_user_item` | user + item features | First real ranker |
| `xgb_plus_interaction` | + interaction features | Isolate interaction contribution |
| `xgb_plus_time` | + time features | Isolate time contribution |
| `lgbm_full` | all features, LightGBM | Compare to XGBoost |
| `xgb_full_tuned` | all features, hyperparameter search | Best model |

### Hyperparameter search space (for final run)

```yaml
n_estimators:    [300, 500, 800]
learning_rate:   [0.01, 0.05, 0.1]
max_depth:       [4, 6, 8]
min_child_weight:[1, 5, 10]
subsample:       [0.7, 0.8, 1.0]
colsample:       [0.7, 0.8, 1.0]
```

Use 3-fold cross-validation on train data. Optimise for MAP@10 on val set.

---

## 8. Evaluation Strategy

### Two-track evaluation — mandatory

74.4% of test users are new to training. A single MAP@10 across all test users
would be misleading. Always report two tracks separately.

**Track 1 — Warm user evaluation (primary)**
- Users: 4,051 warm test users with ≥3 positives in the 2018+ window
- Metrics: MAP@10, NDCG@10, Precision@10, Recall@10, MRR@10
- This is the primary number reported in README and MLflow

**Track 2 — Cold user evaluation (secondary)**
- Users: 13,775 new test users served by fallback logic
- Metrics: genre coverage of recommendations, novelty score, diversity
- Shows fallback quality — not model quality

### Ranking metrics explained

| Metric | Formula summary | What it rewards |
|---|---|---|
| MAP@10 | Mean of average precision at each relevant position | Correct ordering within top-10 |
| NDCG@10 | Normalised discounted cumulative gain | Position-aware — penalises burying relevant items |
| Precision@10 | Relevant items in top-10 / 10 | Simple hit rate |
| Recall@10 | Relevant items in top-10 / total relevant | Coverage of user's liked catalog |
| MRR@10 | 1 / rank of first relevant item | First-hit quality |

### Diversity metrics

| Metric | Formula | Why |
|---|---|---|
| Catalog coverage | Unique movies recommended / total catalog | Prevents top-590 collapse |
| Intra-list diversity | Avg pairwise genre dissimilarity within top-10 | Prevents 10 Drama recs |
| Novelty | -log2(popularity) averaged over top-10 | Long-tail exposure |
| Cold-item exposure rate | % recs that are cold items | 58.8% catalog is cold |

### Evaluation filter
- Filter val and test to warm users with ≥3 positives in that window
- Val eval set: 4,069 users · Test eval set: 4,051 users
- Never evaluate on test until final model is selected

---

## 9. Cold-Start Handling

Cold-start in this project has two dimensions: cold users and cold items.

### User routing logic

```
IF user has ≥20 positive interactions in training data:
    → WARM path: full two-stage ranker + all features

ELSE IF user has 1–19 positive interactions:
    → LIGHT path: simplified ranker (user + item features only, no MF)
                  smaller candidate pool (100 items)
                  more weight on content similarity

ELSE (user has 0 interactions OR is brand new at inference time):
    → NEW USER path: onboarding fallback
        1. Collect 3–5 genre preferences from user
        2. Build genre affinity vector from selections
        3. Return top-rated movies within selected genres
           sorted by (avg_rating × log_popularity_score)
```

### Cold item routing logic

```
IF item has ≥10 ratings in training data:
    → WARM item: scored by full ranker with CF + content features

ELSE (item has <10 ratings — 58.8% of catalog):
    → COLD item path:
        1. Score by genre overlap with user's affinity vector
        2. If genome features available (23% of catalog):
           add genome tag cosine similarity score
        3. Add exploration boost: ε = 0.05
           (5% of cold items surfaced regardless of score)

IF item has no genre listed (8.1% of catalog):
    → NO-GENRE fallback:
        1. Assign zero genre vector
        2. Route to global popularity fallback
        3. Never attempt content-based scoring
```

### Cold-start at evaluation time

Since 39.6% of test items are new to training:
- Warm items (in training): evaluated with full model
- Cold items (new to training): evaluated with content fallback
- Report cold-item MAP@10 separately to show fallback quality

### Interview answer for cold-start

> "I handle cold users by routing to an onboarding-based popularity fallback
> that collects genre preferences. Cold items — which are 58.8% of the catalog
> based on our EDA — are scored using genre overlap and genome tag similarity.
> Because 74.4% of test users are new to training, I always report evaluation
> in two tracks: warm users evaluated on the full ranker, new users evaluated
> on the fallback path. This makes the metrics honest."

---

## 10. MLflow Experiment Tracking

### Three experiment groups

**Experiment 1: baselines**
Runs: popularity_global, popularity_by_genre, cf_itemitem, mf_als

**Experiment 2: ranker_iterations**
Runs: xgb_user_item, xgb_plus_interaction, xgb_plus_time, lgbm_full, xgb_full_tuned

**Experiment 3: drift_evaluation**
Runs: best_model_2018q1, best_model_2018q2, best_model_2018q3, best_model_2018q4, best_model_2019

### What is logged per run

**Parameters:**
```
model_type, n_estimators, learning_rate, max_depth,
feature_set_version, train_end_year, relevance_threshold,
cold_user_threshold (20), cold_item_threshold (10),
candidate_pool_size, negative_sample_ratio,
genre_vector_dim (18), genome_tags_used (50)
```

**Metrics (warm user track):**
```
warm_map_at_10, warm_ndcg_at_10, warm_precision_at_10,
warm_recall_at_10, warm_mrr_at_10
```

**Metrics (diversity):**
```
catalog_coverage, intra_list_diversity,
novelty_score, cold_item_exposure_rate
```

**Metrics (system):**
```
inference_latency_p95_ms, training_duration_s, model_size_mb
```

**Artifacts saved per run:**
```
model.pkl                  serialized ranker
feature_columns.json       exact column list in training order
genre_columns.json         18 genre names in vector order
genome_tag_columns.json    50 genome tag IDs used
configs/model_config.yaml  full config snapshot
eval_report.json           full metrics for both tracks
feature_importance.png     top-20 features bar chart
```

### Model registry states

```
Staging    → new candidate awaiting review
Production → model currently loaded by API
Archived   → all previous production models

Rollback: promote any Archived version to Production in MLflow UI.
API picks it up on next container restart. Zero code change.
```

### API integration with MLflow

```python
# At startup
model = mlflow.pyfunc.load_model("models:/recommender/Production")
run_id = mlflow.get_registered_model("recommender") \
               .latest_versions[0].run_id
feature_cols = json.load(open(f"mlruns/.../feature_columns.json"))

# /model-info endpoint returns
{
  "model_name": "recommender",
  "version": "3",
  "run_id": "abc123...",
  "trained_on": "2026-03-01",
  "val_map_at_10": 0.143,
  "feature_count": 87
}
```

---

## 11. A/B Testing

### Why A/B testing matters here

A/B testing is the standard way to compare two recommendation policies under
identical conditions. Without it, you cannot tell whether MAP@10 differences
between models are due to the model itself or differences in evaluation setup.

### Setup — offline simulation with deterministic bucketing

Since there is no live user traffic, the A/B test runs on the held-out test
set using deterministic hash-based assignment. This mirrors exactly how
production A/B testing works — the only difference is that outcomes are
measured from held-out interactions rather than real-time clicks.

### User assignment

```python
def get_ab_bucket(user_id: int) -> str:
    """Deterministic — same user always gets same bucket."""
    bucket = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 100
    return "control" if bucket < 50 else "treatment"
```

- **Control (bucket 0–49):** popularity baseline recommender
- **Treatment (bucket 50–99):** full XGBoost ranker

Assignment is deterministic — the same user always lands in the same bucket
across sessions. This is a core requirement of real A/B systems.

### What is measured per group

For each group, compute per-user metrics on warm test users (2018–2019):

```
map_at_10          per user → compare distributions
ndcg_at_10         per user
precision_at_10    per user
catalog_coverage   across all recommendations in group
diversity          intra-list genre dissimilarity
latency_p95_ms     inference time
```

### Statistical significance test

Use **Mann-Whitney U test** (non-parametric) on per-user MAP@10 scores.
Chosen because MAP@10 distributions are not Gaussian.

```python
from scipy import stats
control_scores   = [map_at_10 per user in control group]
treatment_scores = [map_at_10 per user in treatment group]

stat, p_value = stats.mannwhitneyu(treatment_scores, control_scores,
                                    alternative='greater')
effect_size = (treatment_mean - control_mean) / control_mean * 100
```

Report: p-value, effect size (%), 95% confidence interval on the difference.
Significance threshold: p < 0.05.

### What to log in MLflow

```
Experiment: ab_test
Run: control_vs_treatment_v1
  Parameters: control_model, treatment_model, test_period, n_users_per_group
  Metrics:
    control_map_at_10, treatment_map_at_10
    control_ndcg_at_10, treatment_ndcg_at_10
    p_value, effect_size_pct, is_significant
    control_n_users, treatment_n_users
  Artifacts:
    ab_test_report.json
    per_user_scores.parquet
    map_distribution_plot.png   (side-by-side boxplot)
```

### API integration — A/B routing at request time

```python
@app.post("/recommend/{user_id}")
async def recommend(user_id: int):
    bucket   = get_ab_bucket(user_id)
    model    = treatment_model if bucket == "treatment" else control_model
    recs     = model.predict(user_id)

    # Log which variant served this request
    prometheus_counter.labels(ab_variant=bucket).inc()
    return {"recommendations": recs, "ab_variant": bucket}
```

This means the API actively routes users to different models at inference time
and tracks which variant served each request — real production A/B behaviour.

### Interview answer for A/B testing

> "I implemented offline A/B testing using deterministic hash-based bucketing
> on user ID, so the same user always lands in the same group. Control received
> the popularity baseline, treatment received the full XGBoost ranker. I
> evaluated both on held-out 2018–2019 warm-user interactions, computed per-user
> MAP@10 for each group, and used a Mann-Whitney U test to confirm statistical
> significance. I also wired the bucket assignment into the FastAPI endpoint so
> the API actively routes users to different models at inference time and logs
> which variant served each request — which is exactly how this would work in
> production with real traffic."

---

## 12. Drift Monitoring

### Three drift categories

**Category 1 — Input drift** (`src/monitoring/feature_drift.py`)
Measures whether the distribution of input features is changing over time.

Tracked features:
- Genre distribution across interactions (monthly)
- User activity counts (active users per month)
- Item popularity distribution (top-10% share per month)
- Rating volume per month
- Cold-user share per month

Method: KS test between 2016 training baseline and each monthly window.
Alert threshold: KS statistic > 0.1 on any tracked feature.

**Category 2 — Prediction drift** (`src/monitoring/prediction_drift.py`)
Measures whether the model's output is changing even if inputs seem stable.

Tracked:
- Distribution of recommendation scores (monthly)
- Top-item concentration: % of recs from top-1% of movies per month
- Genre distribution of recommended items per month
- Cold-item exposure rate per month

Method: KS test between val-period baseline and each monthly window.

**Category 3 — Performance drift** (`src/monitoring/performance_drift.py`)
Measures whether ranking quality is degrading over time.

Method: Evaluate best model on monthly slices of test set.
- 2018 Q1, Q2, Q3, Q4
- 2019 (full year or monthly if data is sufficient)

Output: MAP@10-over-time chart — this is a key README artifact.

Retraining trigger: MAP@10 on any monthly window drops >10% from val baseline.

### Drift report output

```
reports/drift/
  feature_drift_report.html     KS stats per feature, trend charts
  prediction_drift_report.html  Score and concentration drift charts
  performance_drift_report.html MAP@10-over-time chart
  drift_summary.json            Machine-readable, used by monitoring alert
```

Reports generated by running `python src/monitoring/reporter.py`.
In production: schedule weekly via cron or Airflow.

---

## 13. API Design

### Endpoints

```
POST /recommend/{user_id}
  Request body: { "n": 10, "exclude_seen": true }
  Response: {
    "user_id": 123,
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
      },
      ...
    ],
    "model_version": "3",
    "latency_ms": 94
  }

GET /health
  Response: { "status": "ok", "model_version": "3", "uptime_s": 3600 }

GET /metrics
  Prometheus text format:
    recommender_requests_total{ab_variant, user_tier}
    recommender_latency_seconds{quantile}
    recommender_errors_total{error_type}
    recommender_cache_hits_total
    recommender_model_version

GET /model-info
  Response: {
    "model_name": "recommender",
    "version": "3",
    "run_id": "abc123",
    "trained_on": "2026-03-01",
    "val_map_at_10": 0.143,
    "feature_count": 87
  }
```

### Request flow

```
1. Receive user_id
2. Determine AB bucket via hash (control / treatment)
3. Determine user tier (warm / light / new) from feature store
4. Look up user features: Redis → PostgreSQL fallback
5. Generate candidates (pre-cached or compute on-the-fly for warm items)
6. Look up item features for candidates: Redis → PostgreSQL
7. Compute interaction features (genre overlap, tag similarity, MF score)
8. Score candidates with loaded model (or fallback if cold user/item)
9. Sort, deduplicate, remove already-rated
10. Attach reason codes to top-10
11. Log to Prometheus: variant, tier, latency, cache hit/miss
12. Return response
```

### Graceful degradation

```
If model fails to load at startup   → FAIL FAST (container restarts)
If model fails during inference     → fall back to popularity ranker, log error
If feature store unreachable        → serve from in-memory emergency cache (60s)
If Redis down                       → go directly to PostgreSQL
If user not found                   → treat as new user, serve onboarding fallback
```

---

## 14. Functional Requirements

| ID | Requirement | Priority | Grounded in |
|---|---|---|---|
| FR-1 | Given userId, return top-10 ranked unseen movies scored ≥ 4.0 threshold | Must | Core problem |
| FR-2 | Use two-stage pipeline: retrieval (100–300) then learning-to-rank | Must | Production design |
| FR-3 | Route users: warm (≥20 pos) / light (1–19) / new (0) | Must | 23.4% light users in data |
| FR-4 | Cold items (<10 ratings): score via genre + genome content features | Must | 58.8% of catalog is cold |
| FR-5 | New API users: onboarding → genre preferences → popularity fallback | Must | Inference-time cold start |
| FR-6 | Evaluate MAP@10 on warm test users (4,051) separately from cold users | Must | 74.4% of test users are new |
| FR-7 | A/B test: deterministic hash bucketing, control vs treatment | Must | Key portfolio differentiator |
| FR-8 | Log all experiments in MLflow: params, metrics (both tracks), artifacts | Must | Reproducibility |
| FR-9 | Register best model in MLflow registry, API loads by registry name | Must | Production model serving |
| FR-10 | Generate drift reports: input, prediction, performance (monthly windows) | Must | Model lifecycle |
| FR-11 | Expose /recommend, /health, /metrics, /model-info endpoints | Must | API contract |
| FR-12 | Response includes movie title, score, reason code, AB variant | Must | Explainability |
| FR-13 | Retraining trigger: MAP@10 drops >10% from val baseline | Should | Automated lifecycle |
| FR-14 | No-genre movies (8.1%) routed to popularity fallback, not content scoring | Should | Data quality |
| FR-15 | Architecture diagram documents AWS equivalent even if not deployed | Should | Portfolio completeness |

---

## 15. Non-Functional Requirements

| ID | Requirement | Target | How measured |
|---|---|---|---|
| NFR-1 | P95 API latency (warm user, cached features) | < 200ms | Prometheus histogram |
| NFR-2 | P95 API latency (cold user, fallback) | < 50ms | Prometheus histogram |
| NFR-3 | API uptime | > 99% | Docker healthcheck |
| NFR-4 | Graceful degradation on ranker failure | Always return 200 | Error rate metric |
| NFR-5 | Batch feature pipeline runtime | < 2 hours | Pipeline timer |
| NFR-6 | Every training run fully reproducible from run ID | 100% | MLflow audit |
| NFR-7 | All random seeds fixed | numpy, sklearn, xgboost | Code review |
| NFR-8 | Docker image builds cleanly from scratch | CI check | docker build |
| NFR-9 | All secrets via environment variables | 0 secrets in code | Code scan |
| NFR-10 | Raw data never modified | File checksums stable | Checksum in metadata |
| NFR-11 | Feature column order locked per model version | feature_columns.json | Inference check |
| NFR-12 | Model loaded from registry — no hardcoded paths | Code review | Registry name only |

---

## 16. Deployment

### Phase 1 — Local Docker stack (do now)

`docker-compose up` starts all 6 services:

```yaml
services:
  api:        FastAPI recommendation service
  postgres:   postgres:15    — feature store
  redis:      redis:7        — feature cache
  mlflow:     mlflow server  — experiment UI + registry
  prometheus: prom/prometheus — metrics scraping
  grafana:    grafana/grafana — dashboards
```

One command. Everything wired. Runs entirely on your laptop.

### Phase 2 — Free live link (after system is stable)

Deploy FastAPI container to **Fly.io** (free tier, Docker-native):
```bash
fly launch --dockerfile Dockerfile
fly deploy
```

MLflow and monitoring stay local — they do not need to be public.
Live link goes in README for portfolio visibility.

### Phase 3 — AWS documentation (optional, no cost)

Document the production equivalent without running it:
- `infra/ecs_task_definition.json` — Fargate task config
- `infra/ecr_push.sh` — image push script
- `infra/prometheus.yml` — scrape config
- `architecture.png` — local stack + AWS equivalent side by side

This shows you understand production deployment without spending money.

### What actually matters for portfolio

1. **Screen-recorded demo** (3–5 min): `docker-compose up` → hit
   `/recommend/123` → Grafana dashboard → MLflow experiment comparison
2. **README results table**: baseline vs CF vs MF vs ranker iterations vs final
3. **Drift chart**: MAP@10 across 2018 Q1 → Q2 → Q3 → Q4 plotted as a line
4. **A/B test result**: control vs treatment MAP@10 with p-value and effect size
5. **Architecture diagram**: local and AWS equivalent

---

## 17. Build Plan

### Phase 1 — Data foundation (Week 1)
- [ ] Set up project repo with full folder structure
- [ ] `src/ingestion/loader.py` — load all 5 files
- [ ] `src/ingestion/cleaner.py` — clean, parse year, build genre vector
- [ ] `src/ingestion/splitter.py` — temporal split with hardcoded dates
- [ ] Verify: save processed parquet files, write splits_metadata.json
- [ ] Set up docker-compose with postgres + redis, populate feature tables

### Phase 2 — Feature engineering (Week 2)
- [ ] `src/features/user_features.py` — all user aggregates with log transforms
- [ ] `src/features/item_features.py` — all item features, genome top-50
- [ ] `src/features/interaction_features.py` — genre overlap, MF score, tag sim
- [ ] `src/features/time_features.py` — temporal context features
- [ ] `src/features/feature_store.py` — assemble full training matrix
- [ ] Verify: no data leakage, feature column list saved to configs/

### Phase 3 — Baselines + ranker (Week 3)
- [ ] `src/candidates/popularity.py`
- [ ] `src/candidates/collaborative.py`
- [ ] `src/candidates/matrix_factorization.py`
- [ ] `src/ranking/baselines.py` — 4 baseline runs logged to MLflow
- [ ] `src/ranking/ranker.py` — XGBoost/LightGBM Ranker training loop
- [ ] `src/evaluation/ranking_metrics.py` — MAP, NDCG, Precision, Recall, MRR
- [ ] `src/evaluation/diversity_metrics.py` — coverage, diversity, novelty
- [ ] Train all 9 model variants, log everything to MLflow
- [ ] Register best model in MLflow registry

### Phase 4 — Cold-start + A/B testing (Week 4)
- [ ] `src/ranking/cold_start.py` — routing logic, content fallback
- [ ] `src/monitoring/ab_testing.py` — hash bucketing, per-group evaluation
- [ ] Run A/B test: control (popularity) vs treatment (best ranker)
- [ ] Mann-Whitney U test, log results to MLflow
- [ ] Verify two-track evaluation produces honest separate numbers

### Phase 5 — API + serving (Week 5)
- [ ] `src/api/main.py` — FastAPI app with lifespan (model load at startup)
- [ ] `src/api/routes.py` — all 4 endpoints
- [ ] `src/api/schemas.py` — Pydantic request/response models
- [ ] `src/api/dependencies.py` — feature store client, model loader
- [ ] Wire A/B routing into /recommend endpoint
- [ ] Cold-start routing in /recommend
- [ ] Prometheus metrics on all endpoints

### Phase 6 — Monitoring + polish (Week 6)
- [ ] `src/monitoring/feature_drift.py` — KS tests on monthly windows
- [ ] `src/monitoring/performance_drift.py` — MAP@10 time series
- [ ] `src/monitoring/reporter.py` — HTML + JSON drift reports
- [ ] Grafana dashboard JSON
- [ ] Dockerfile + docker-compose with all 6 services
- [ ] README: architecture diagram, results table, drift chart, A/B result
- [ ] Screen record demo
- [ ] (Optional) deploy to Fly.io for live link

---

## 18. KPI Targets

### Primary — must beat these or model does not ship

| Metric | Target | Baseline to beat | Track |
|---|---|---|---|
| warm_map_at_10 | ≥ 0.12 | Popularity ~0.081 (+48%) | Warm users |
| warm_ndcg_at_10 | ≥ 0.18 | Popularity ~0.124 | Warm users |
| warm_precision_at_10 | ≥ 0.25 | Popularity ~0.162 | Warm users |
| warm_recall_at_10 | ≥ 0.15 | Popularity ~0.10 | Warm users |

### Diversity — must hit to prevent popularity collapse

| Metric | Target | Why |
|---|---|---|
| catalog_coverage | > 15% of catalog | Top 1% = 47.6% of ratings |
| intra_list_diversity | > 0.6 avg genre dissimilarity | Prevent 10-Drama lists |
| cold_item_exposure_rate | > 5% of recommendations | 58.8% catalog is cold |

### A/B test

| Metric | Target |
|---|---|
| MAP@10 lift (treatment vs control) | > 20% relative improvement |
| p-value | < 0.05 |
| Effect size | Report in README |

### Operational

| Metric | Target |
|---|---|
| P95 latency (warm user) | < 200ms |
| API success rate | > 99% |
| Model retraining trigger | MAP@10 drops > 10% from val baseline |