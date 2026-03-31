"""Comprehensive tests for src/config/*.

Coverage:
- Default values match EDA-locked constants
- Invalid values raise AssertionError with descriptive messages
- Valid configs pass without error
- YAML round-trip: write → load → values preserved
- ExperimentConfig routes to XGBConfig vs LGBMConfig by model_type
- to_mlflow_params() returns only str values, str keys, with correct prefixes
- Boundary values for relevance_threshold and learning_rate
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    FeatureConfig,
    LGBMConfig,
    TrainingConfig,
    XGBConfig,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_yaml(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(content), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# DataConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataConfigDefaults:
    def test_locked_eda_values(self) -> None:
        cfg = DataConfig()
        assert cfg.train_end_year == 2016
        assert cfg.val_year == 2017
        assert cfg.test_start_year == 2018
        assert cfg.relevance_threshold == 4.0
        assert cfg.cold_user_threshold == 20
        assert cfg.cold_item_threshold == 10

    def test_sampling_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.negative_sample_ratio == 4
        assert cfg.random_seed == 42

    def test_path_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.raw_data_dir == "data/raw"
        assert cfg.processed_data_dir == "data/processed"


class TestDataConfigValidation:
    def test_train_must_be_before_val(self) -> None:
        with pytest.raises(AssertionError, match="train_end_year"):
            DataConfig(train_end_year=2017, val_year=2017, test_start_year=2018)

    def test_val_must_be_before_test(self) -> None:
        with pytest.raises(AssertionError, match="val_year"):
            DataConfig(train_end_year=2016, val_year=2018, test_start_year=2018)

    def test_relevance_threshold_too_low(self) -> None:
        with pytest.raises(AssertionError, match="relevance_threshold"):
            DataConfig(relevance_threshold=0.4)

    def test_relevance_threshold_too_high(self) -> None:
        with pytest.raises(AssertionError, match="relevance_threshold"):
            DataConfig(relevance_threshold=5.1)

    def test_relevance_threshold_boundary_low(self) -> None:
        cfg = DataConfig(relevance_threshold=0.5)
        assert cfg.relevance_threshold == 0.5

    def test_relevance_threshold_boundary_high(self) -> None:
        cfg = DataConfig(relevance_threshold=5.0)
        assert cfg.relevance_threshold == 5.0

    def test_cold_user_threshold_zero(self) -> None:
        with pytest.raises(AssertionError, match="cold_user_threshold"):
            DataConfig(cold_user_threshold=0)

    def test_cold_item_threshold_zero(self) -> None:
        with pytest.raises(AssertionError, match="cold_item_threshold"):
            DataConfig(cold_item_threshold=0)

    def test_negative_sample_ratio_zero(self) -> None:
        with pytest.raises(AssertionError, match="negative_sample_ratio"):
            DataConfig(negative_sample_ratio=0)

    def test_valid_config_passes(self) -> None:
        cfg = DataConfig(
            train_end_year=2015,
            val_year=2016,
            test_start_year=2017,
            relevance_threshold=3.5,
            cold_user_threshold=5,
            cold_item_threshold=3,
            negative_sample_ratio=2,
        )
        assert cfg.train_end_year == 2015


class TestDataConfigYamlRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = DataConfig(relevance_threshold=3.5, random_seed=99)
        p = _write_yaml(tmp_path, original.to_dict())
        loaded = DataConfig.from_yaml(p)
        assert loaded.relevance_threshold == 3.5
        assert loaded.random_seed == 99
        assert loaded.train_end_year == 2016

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DataConfig.from_yaml(tmp_path / "nonexistent.yaml")


class TestDataConfigMlflowParams:
    def test_all_strings(self) -> None:
        params = DataConfig().to_mlflow_params()
        for k, v in params.items():
            assert isinstance(k, str), f"key {k!r} is not a string"
            assert isinstance(v, str), f"value for {k!r} is not a string"

    def test_expected_keys_present(self) -> None:
        params = DataConfig().to_mlflow_params()
        assert "train_end_year" in params
        assert "relevance_threshold" in params
        assert params["train_end_year"] == "2016"
        assert params["relevance_threshold"] == "4.0"


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureConfigDefaults:
    def test_locked_genre_dim(self) -> None:
        assert FeatureConfig().genre_vector_dim == 18

    def test_genome_tags(self) -> None:
        assert FeatureConfig().n_genome_tags == 50

    def test_mf_embedding_dim(self) -> None:
        assert FeatureConfig().mf_embedding_dim == 64


class TestFeatureConfigValidation:
    def test_genre_dim_must_be_18(self) -> None:
        with pytest.raises(AssertionError, match="genre_vector_dim"):
            FeatureConfig(genre_vector_dim=20)

    def test_invalid_genome_tags(self) -> None:
        with pytest.raises(AssertionError, match="n_genome_tags"):
            FeatureConfig(n_genome_tags=30)

    def test_valid_genome_tags(self) -> None:
        for v in (20, 50, 100):
            cfg = FeatureConfig(n_genome_tags=v)
            assert cfg.n_genome_tags == v

    def test_invalid_mf_dim(self) -> None:
        with pytest.raises(AssertionError, match="mf_embedding_dim"):
            FeatureConfig(mf_embedding_dim=48)

    def test_valid_mf_dims(self) -> None:
        for v in (32, 64, 128):
            cfg = FeatureConfig(mf_embedding_dim=v)
            assert cfg.mf_embedding_dim == v


# ═══════════════════════════════════════════════════════════════════════════════
# XGBConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestXGBConfigDefaults:
    def test_model_type(self) -> None:
        assert XGBConfig().model_type == "xgboost"

    def test_objective(self) -> None:
        assert XGBConfig().objective == "rank:pairwise"

    def test_n_estimators(self) -> None:
        assert XGBConfig().n_estimators == 300


class TestXGBConfigValidation:
    def test_learning_rate_too_low(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate"):
            XGBConfig(learning_rate=0.0009)

    def test_learning_rate_too_high(self) -> None:
        with pytest.raises(AssertionError, match="learning_rate"):
            XGBConfig(learning_rate=1.1)

    def test_learning_rate_boundary_low(self) -> None:
        cfg = XGBConfig(learning_rate=0.001)
        assert cfg.learning_rate == 0.001

    def test_learning_rate_boundary_high(self) -> None:
        cfg = XGBConfig(learning_rate=1.0)
        assert cfg.learning_rate == 1.0

    def test_n_estimators_zero(self) -> None:
        with pytest.raises(AssertionError, match="n_estimators"):
            XGBConfig(n_estimators=0)

    def test_max_depth_zero(self) -> None:
        with pytest.raises(AssertionError, match="max_depth"):
            XGBConfig(max_depth=0)

    def test_subsample_too_low(self) -> None:
        with pytest.raises(AssertionError, match="subsample"):
            XGBConfig(subsample=0.05)

    def test_colsample_too_high(self) -> None:
        with pytest.raises(AssertionError, match="colsample_bytree"):
            XGBConfig(colsample_bytree=1.1)

    def test_valid_config(self) -> None:
        cfg = XGBConfig(learning_rate=0.01, n_estimators=100, max_depth=4)
        assert cfg.learning_rate == 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# LGBMConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestLGBMConfigDefaults:
    def test_model_type(self) -> None:
        assert LGBMConfig().model_type == "lightgbm"

    def test_n_estimators(self) -> None:
        assert LGBMConfig().n_estimators == 500

    def test_num_leaves(self) -> None:
        assert LGBMConfig().num_leaves == 31


class TestLGBMConfigValidation:
    def test_learning_rate_boundary_low(self) -> None:
        cfg = LGBMConfig(learning_rate=0.001)
        assert cfg.learning_rate == 0.001

    def test_learning_rate_boundary_high(self) -> None:
        cfg = LGBMConfig(learning_rate=1.0)
        assert cfg.learning_rate == 1.0

    def test_num_leaves_zero(self) -> None:
        with pytest.raises(AssertionError, match="num_leaves"):
            LGBMConfig(num_leaves=0)

    def test_subsample_out_of_range(self) -> None:
        with pytest.raises(AssertionError, match="subsample"):
            LGBMConfig(subsample=0.0)

    def test_valid_config(self) -> None:
        cfg = LGBMConfig(num_leaves=63, learning_rate=0.02)
        assert cfg.num_leaves == 63


# ═══════════════════════════════════════════════════════════════════════════════
# TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingConfigDefaults:
    def test_candidate_pool_size(self) -> None:
        assert TrainingConfig().candidate_pool_size == 300

    def test_als_factors(self) -> None:
        assert TrainingConfig().als_factors == 64

    def test_faiss_index_type(self) -> None:
        assert TrainingConfig().faiss_index_type == "IndexFlatIP"

    def test_per_source_candidates_sum(self) -> None:
        cfg = TrainingConfig()
        total = cfg.n_candidates_pop + cfg.n_candidates_cf + cfg.n_candidates_mf
        assert total >= cfg.candidate_pool_size


class TestTrainingConfigValidation:
    def test_candidate_pool_too_small(self) -> None:
        with pytest.raises(AssertionError, match="candidate_pool_size"):
            TrainingConfig(candidate_pool_size=50)

    def test_candidate_pool_too_large(self) -> None:
        with pytest.raises(AssertionError, match="candidate_pool_size"):
            TrainingConfig(candidate_pool_size=501)

    def test_invalid_als_factors(self) -> None:
        with pytest.raises(AssertionError, match="als_factors"):
            TrainingConfig(als_factors=48)

    def test_valid_als_factors(self) -> None:
        for v in (32, 64, 128):
            cfg = TrainingConfig(als_factors=v)
            assert cfg.als_factors == v

    def test_als_iterations_zero(self) -> None:
        with pytest.raises(AssertionError, match="als_iterations"):
            TrainingConfig(als_iterations=0)

    def test_candidates_sum_too_small(self) -> None:
        with pytest.raises(AssertionError):
            TrainingConfig(
                candidate_pool_size=300,
                n_candidates_pop=50,
                n_candidates_cf=50,
                n_candidates_mf=50,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# EvalConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvalConfigDefaults:
    def test_k(self) -> None:
        assert EvalConfig().k == 10

    def test_warm_user_min_positives(self) -> None:
        assert EvalConfig().warm_user_min_positives == 3

    def test_metrics_list(self) -> None:
        metrics = EvalConfig().metrics
        assert "map" in metrics
        assert "ndcg" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "mrr" in metrics

    def test_diversity_metrics_list(self) -> None:
        dm = EvalConfig().diversity_metrics
        assert "catalog_coverage" in dm
        assert "intra_list_diversity" in dm
        assert "novelty" in dm
        assert "cold_item_exposure" in dm


class TestEvalConfigValidation:
    def test_invalid_k(self) -> None:
        with pytest.raises(AssertionError, match=r"k \("):
            EvalConfig(k=15)

    def test_valid_k_values(self) -> None:
        for v in (5, 10, 20):
            assert EvalConfig(k=v).k == v

    def test_warm_user_min_positives_zero(self) -> None:
        with pytest.raises(AssertionError, match="warm_user_min_positives"):
            EvalConfig(warm_user_min_positives=0)

    def test_warm_user_min_positives_one(self) -> None:
        cfg = EvalConfig(warm_user_min_positives=1)
        assert cfg.warm_user_min_positives == 1

    def test_mutable_defaults_are_independent(self) -> None:
        a = EvalConfig()
        b = EvalConfig()
        a.metrics.append("extra")
        assert "extra" not in b.metrics


# ═══════════════════════════════════════════════════════════════════════════════
# ExperimentConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestExperimentConfigDefaults:
    def test_default_sub_configs(self) -> None:
        cfg = ExperimentConfig()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.feature, FeatureConfig)
        assert isinstance(cfg.model, XGBConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.eval, EvalConfig)

    def test_default_model_is_xgb(self) -> None:
        cfg = ExperimentConfig()
        assert cfg.model.model_type == "xgboost"


class TestExperimentConfigRouting:
    def test_routes_to_xgb(self, tmp_path: Path) -> None:
        content = {
            "data": {},
            "feature": {},
            "model": {"model_type": "xgboost"},
            "training": {},
            "eval": {},
        }
        p = _write_yaml(tmp_path, content)
        cfg = ExperimentConfig.from_yaml(p)
        assert isinstance(cfg.model, XGBConfig)

    def test_routes_to_lgbm(self, tmp_path: Path) -> None:
        content = {
            "data": {},
            "feature": {},
            "model": {"model_type": "lightgbm"},
            "training": {},
            "eval": {},
        }
        p = _write_yaml(tmp_path, content)
        cfg = ExperimentConfig.from_yaml(p)
        assert isinstance(cfg.model, LGBMConfig)

    def test_lgbm_specific_field_preserved(self, tmp_path: Path) -> None:
        content = {
            "data": {},
            "feature": {},
            "model": {"model_type": "lightgbm", "num_leaves": 63},
            "training": {},
            "eval": {},
        }
        p = _write_yaml(tmp_path, content)
        cfg = ExperimentConfig.from_yaml(p)
        assert isinstance(cfg.model, LGBMConfig)
        assert cfg.model.num_leaves == 63


class TestExperimentConfigYamlRoundtrip:
    def test_full_roundtrip(self, tmp_path: Path) -> None:
        content = {
            "data": {"relevance_threshold": 3.5, "random_seed": 7},
            "feature": {"n_genome_tags": 20},
            "model": {"model_type": "xgboost", "n_estimators": 200},
            "training": {"als_factors": 32},
            "eval": {"k": 5},
        }
        p = _write_yaml(tmp_path, content)
        cfg = ExperimentConfig.from_yaml(p)

        assert cfg.data.relevance_threshold == 3.5
        assert cfg.data.random_seed == 7
        assert cfg.feature.n_genome_tags == 20
        assert cfg.model.n_estimators == 200
        assert cfg.training.als_factors == 32
        assert cfg.eval.k == 5

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml(tmp_path / "missing.yaml")

    def test_empty_sections_use_defaults(self, tmp_path: Path) -> None:
        content = {"data": {}, "feature": {}, "model": {}, "training": {}, "eval": {}}
        p = _write_yaml(tmp_path, content)
        cfg = ExperimentConfig.from_yaml(p)
        assert cfg.data.train_end_year == 2016
        assert cfg.feature.genre_vector_dim == 18
        assert cfg.eval.k == 10


class TestExperimentConfigMlflowParams:
    def test_all_values_are_strings(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        for k, v in params.items():
            assert isinstance(k, str), f"key {k!r} not a string"
            assert isinstance(v, str), f"value for {k!r} not a string"

    def test_all_keys_are_strings(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        for k in params:
            assert isinstance(k, str)

    def test_prefixed_keys_data(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        assert "data.train_end_year" in params
        assert params["data.train_end_year"] == "2016"

    def test_prefixed_keys_model(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        assert "model.learning_rate" in params
        assert params["model.learning_rate"] == "0.05"

    def test_prefixed_keys_training(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        assert "training.candidate_pool_size" in params

    def test_prefixed_keys_eval(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        assert "eval.k" in params
        assert params["eval.k"] == "10"

    def test_prefixed_keys_feature(self) -> None:
        params = ExperimentConfig().to_mlflow_params()
        assert "feature.genre_vector_dim" in params
        assert params["feature.genre_vector_dim"] == "18"

    def test_lgbm_prefixed_params(self) -> None:
        cfg = ExperimentConfig(model=LGBMConfig())
        params = cfg.to_mlflow_params()
        assert "model.model_type" in params
        assert params["model.model_type"] == "lightgbm"


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-cutting: to_mlflow_params on individual configs
# ═══════════════════════════════════════════════════════════════════════════════

class TestMlflowParamsIndividual:
    @pytest.mark.parametrize("cfg_cls", [DataConfig, FeatureConfig, XGBConfig, LGBMConfig, TrainingConfig, EvalConfig])
    def test_all_strings_no_prefix(self, cfg_cls: type) -> None:
        params = cfg_cls().to_mlflow_params()  # type: ignore[call-arg]
        for k, v in params.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_list_fields_serialized(self) -> None:
        params = EvalConfig().to_mlflow_params()
        assert "metrics" in params
        assert isinstance(params["metrics"], str)
