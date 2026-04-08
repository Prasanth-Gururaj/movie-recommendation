"""Tests for RankingEvaluator and EvalReport."""

from __future__ import annotations

import math
import pytest

from src.evaluation.ranking_metrics import RankingEvaluator
from src.evaluation.evaluator import EvalReport


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ev():
    return RankingEvaluator(k=10)


@pytest.fixture
def ev5():
    return RankingEvaluator(k=5)


# ── MAP@k ─────────────────────────────────────────────────────────────────────

class TestMAP:
    def test_perfect_list(self, ev):
        result = ev.evaluate([[1, 2, 3]], [{1, 2, 3}])
        assert result["map@10"] == pytest.approx(1.0)

    def test_empty_predictions(self, ev):
        result = ev.evaluate([], [])
        assert result["map@10"] == 0.0

    def test_no_relevant_items_skipped(self, ev):
        result = ev.evaluate([[1, 2, 3]], [set()])
        assert result["map@10"] == 0.0

    def test_hand_verified_example(self, ev):
        # pred=[1,2,3,4,5], relevant={1,3}
        # hits at rank 1 (p=1/1) and rank 3 (p=2/3)
        # AP = (1.0 + 2/3) / min(2,10) = 5/6
        result = ev.evaluate([[1, 2, 3, 4, 5]], [{1, 3}])
        expected = (1.0 + 2.0 / 3) / 2
        assert result["map@10"] == pytest.approx(expected, rel=1e-4)

    def test_relevant_at_last_position(self, ev):
        # Single relevant item at position 10 -> AP = (1/10) / 1 = 0.1
        preds = [[99, 98, 97, 96, 95, 94, 93, 92, 91, 1]]
        result = ev.evaluate(preds, [{1}])
        assert result["map@10"] == pytest.approx(0.1, rel=1e-4)

    def test_relevant_outside_k_ignored(self, ev5):
        # item 1 at position 6 but k=5 -> not counted
        result = ev5.evaluate([[99, 98, 97, 96, 95, 1]], [{1}])
        assert result["map@5"] == 0.0

    def test_multiple_users_macro_averaged(self, ev):
        # user1: hit at pos1 (AP=1.0), user2: miss (AP=0.0)
        result = ev.evaluate([[1], [2]], [{1}, {99}])
        assert result["map@10"] == pytest.approx(0.5)

    def test_all_miss(self, ev):
        result = ev.evaluate([[10, 11, 12]], [{1, 2, 3}])
        assert result["map@10"] == 0.0


# ── NDCG@k ────────────────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_list(self, ev):
        result = ev.evaluate([[1, 2, 3]], [{1, 2, 3}])
        assert result["ndcg@10"] == pytest.approx(1.0)

    def test_zero_hits(self, ev):
        result = ev.evaluate([[10, 11]], [{1, 2}])
        assert result["ndcg@10"] == 0.0

    def test_single_relevant_at_position_1(self, ev):
        # DCG = 1/log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
        result = ev.evaluate([[1, 2, 3]], [{1}])
        assert result["ndcg@10"] == pytest.approx(1.0)

    def test_single_relevant_at_position_2(self, ev):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1.0
        result = ev.evaluate([[99, 1, 2]], [{1}])
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert result["ndcg@10"] == pytest.approx(expected, rel=1e-4)

    def test_empty(self, ev):
        assert ev.evaluate([], [])["ndcg@10"] == 0.0


# ── MRR@k ─────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position_mrr_is_1(self, ev):
        result = ev.evaluate([[1, 2, 3]], [{1}])
        assert result["mrr@10"] == pytest.approx(1.0)

    def test_second_position_mrr_is_half(self, ev):
        result = ev.evaluate([[99, 1, 2]], [{1}])
        assert result["mrr@10"] == pytest.approx(0.5)

    def test_third_position_mrr_is_one_third(self, ev):
        result = ev.evaluate([[99, 98, 1]], [{1}])
        assert result["mrr@10"] == pytest.approx(1.0 / 3, rel=1e-4)

    def test_no_hit_mrr_is_zero(self, ev):
        result = ev.evaluate([[99, 98, 97]], [{1}])
        assert result["mrr@10"] == 0.0


# ── Precision & Recall ────────────────────────────────────────────────────────

class TestPrecisionRecall:
    def test_precision_perfect(self, ev5):
        result = ev5.evaluate([[1, 2, 3, 4, 5]], [{1, 2, 3, 4, 5}])
        assert result["precision@5"] == pytest.approx(1.0)

    def test_precision_partial(self, ev5):
        # 3 hits in 5 recs
        result = ev5.evaluate([[1, 99, 2, 98, 3]], [{1, 2, 3}])
        assert result["precision@5"] == pytest.approx(3.0 / 5)

    def test_recall_all_found(self, ev5):
        # 2 relevant, both in top-5
        result = ev5.evaluate([[1, 2, 3, 4, 5]], [{1, 2}])
        assert result["recall@5"] == pytest.approx(1.0)

    def test_recall_partial(self, ev5):
        # 1 hit out of 4 relevant
        result = ev5.evaluate([[1, 99, 98, 97, 96]], [{1, 2, 3, 4}])
        assert result["recall@5"] == pytest.approx(0.25)


# ── EvalReport ────────────────────────────────────────────────────────────────

class TestEvalReport:
    def test_to_mlflow_metrics_returns_only_scalars(self):
        report = EvalReport(
            config_name="test_run",
            ranking={"map@10": 0.25, "ndcg@10": 0.30},
            diversity={"catalog_coverage": 0.12, "intra_list_diversity": 0.65},
            split="val",
            n_users=100,
            extra={"custom_metric": 0.99},
        )
        metrics = report.to_mlflow_metrics()
        assert all(isinstance(v, float) for v in metrics.values())
        assert "map_at_10" in metrics  # @ is sanitized to _at_ for MLflow
        assert "catalog_coverage" in metrics
        assert "custom_metric" in metrics

    def test_to_mlflow_metrics_values_correct(self):
        report = EvalReport(
            config_name="x",
            ranking={"map@10": 0.123},
            diversity={"catalog_coverage": 0.456},
        )
        metrics = report.to_mlflow_metrics()
        assert metrics["map_at_10"] == pytest.approx(0.123)  # @ sanitized to _at_
        assert metrics["catalog_coverage"] == pytest.approx(0.456)

    def test_primary_metric(self):
        report = EvalReport(
            config_name="x",
            ranking={"map@10": 0.123},
            diversity={},
        )
        assert report.primary_metric(10) == pytest.approx(0.123)

    def test_primary_metric_missing_returns_zero(self):
        report = EvalReport(config_name="x", ranking={}, diversity={})
        assert report.primary_metric(10) == 0.0

    def test_config_name_preserved(self):
        report = EvalReport(config_name="xgb_full_tuned", ranking={}, diversity={})
        assert report.config_name == "xgb_full_tuned"

    def test_split_field(self):
        report = EvalReport(config_name="x", ranking={}, diversity={}, split="val")
        assert report.split == "val"
