
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import PROJECT_ROOT, TEST_DATA_DIR
from data_loader import (
    load_aisles,
    load_order_products,
    load_orders,
    load_predictions,
    load_products,
    load_rules,
)
from recommendation_engine import RecommendationEngine

TEST_DATA = TEST_DATA_DIR


# ----------------------------------------------------------------------
# Fixtures: small synthetic test dataframes
# ----------------------------------------------------------------------


@pytest.fixture
def test_aisles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aisle_id": [1, 2, 3, 4, 5],
            "aisle": [
                "prepared soups salads",
                "specialty cheeses",
                "packaged vegetables fruits",
                "yogurt",
                "bread",
            ],
        }
    )


@pytest.fixture
def test_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [100, 101, 102, 103, 104, 105],
            "product_name": [
                "Product A",
                "Product B",
                "Product C",
                "Product D",
                "Product E",
                "Product F",
            ],
            "aisle_id": [1, 2, 3, 4, 5, 1],
            "department_id": [1, 1, 2, 2, 3, 1],
        }
    )


@pytest.fixture
def test_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1001, 1002, 1003, 1004],
            "user_id": [1, 1, 2, 3],
            "eval_set": ["prior", "prior", "prior", "prior"],
            "order_number": [1, 2, 1, 1],
        }
    )


@pytest.fixture
def test_order_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1001, 1001, 1002, 1002, 1003, 1004],
            "product_id": [100, 101, 102, 103, 104, 105],
            "add_to_cart_order": [1, 2, 1, 2, 1, 1],
            "reordered": [1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def test_rules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "antecedent_ids": ["1", "1", "2", "1"],
            "consequent_ids": [4, 2, 4, 5],
            "ve_trai": [
                "prepared soups salads",
                "prepared soups salads",
                "specialty cheeses",
                "prepared soups salads",
            ],
            "ve_phai": [
                "yogurt",
                "specialty cheeses",
                "yogurt",
                "bread",
            ],
            "support": [0.02, 0.015, 0.01, 0.008],
            "confidence": [0.6, 0.5, 0.4, 0.3],
            "lift": [2.0, 1.8, 1.5, 1.2],
            "level": ["aisle", "aisle", "aisle", "aisle"],
        }
    )


@pytest.fixture
def test_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "product_id": [100, 101, 104],
            "product_name": ["Product A", "Product B", "Product E"],
            "probability": [0.95, 0.90, 0.85],
            "predicted": [1, 1, 1],
            "rank_in_user": [1, 2, 1],
            "past_orders_count": [2, 2, 1],
            "trang_thai_goi_y": [
                "Nên gợi ý mua lại",
                "Nên gợi ý mua lại",
                "Nên gợi ý mua lại",
            ],
        }
    )


@pytest.fixture
def engine(test_rules, test_predictions, test_orders, test_order_products, test_products, test_aisles):
    return RecommendationEngine(
        rules=test_rules,
        predictions=test_predictions,
        orders=test_orders,
        order_products=test_order_products,
        products=test_products,
        aisles=test_aisles,
    )


# ----------------------------------------------------------------------
# Test 1: Load rules from real file
# ----------------------------------------------------------------------


def test_load_rules_from_real_file():
    """Real FP-Growth rules file loads with expected columns and rows."""
    rules = load_rules()
    assert not rules.empty
    assert "antecedent_ids" in rules.columns
    assert "consequent_ids" in rules.columns
    assert "lift" in rules.columns
    assert "antecedent_set" in rules.columns
    assert len(rules) > 0


# ----------------------------------------------------------------------
# Test 2: Load predictions from real file
# ----------------------------------------------------------------------


def test_load_predictions_from_real_file():
    """Real ML predictions file loads with expected columns and rows."""
    preds = load_predictions()
    assert not preds.empty
    assert "user_id" in preds.columns
    assert "product_id" in preds.columns
    assert "probability" in preds.columns
    assert len(preds) > 0


# ----------------------------------------------------------------------
# Test 3: Engine with empty data returns empty results
# ----------------------------------------------------------------------


def test_engine_empty_data_returns_empty():
    """Engine constructed with empty DataFrames returns empty DataFrames."""
    empty = pd.DataFrame()
    eng = RecommendationEngine(
        rules=empty,
        predictions=empty,
        orders=empty,
        order_products=empty,
        products=empty,
        aisles=empty,
    )
    ml = eng.ml_recommendations(user_id=1)
    fp = eng.fpgrowth_recommendations(user_id=1)
    combined = eng.combined_recommendations(user_id=1)
    assert ml.empty
    assert fp.empty
    assert combined.empty


# ----------------------------------------------------------------------
# Test 4: ML recommendations for existing user, sorted by probability
# ----------------------------------------------------------------------


def test_ml_recommendations_existing_user(engine):
    """User 1 gets ML recommendations sorted by probability desc."""
    ml = engine.ml_recommendations(user_id=1, top_n=10)
    assert not ml.empty
    assert list(ml["source"]) == ["ML", "ML"]
    assert list(ml["product_id"]) == [100, 101]
    assert list(ml["probability"]) == [0.95, 0.90]
    assert list(ml["rank"]) == [1, 2]


# ----------------------------------------------------------------------
# Test 5: ML recommendations for non-existent user returns empty
# ----------------------------------------------------------------------


def test_ml_recommendations_nonexistent_user(engine):
    """User 999 has no ML predictions, returns empty DataFrame."""
    ml = engine.ml_recommendations(user_id=999, top_n=10)
    assert ml.empty


# ----------------------------------------------------------------------
# Test 6: FP-Growth recommendations when user has matching aisles
# ----------------------------------------------------------------------


def test_fpgrowth_recommendations_matching_aisles(engine):
    """User 1 bought from aisles {1,2,3,4}; rules match -> aisle 5 products."""
    fp = engine.fpgrowth_recommendations(user_id=1, top_n=10)
    assert not fp.empty
    # User 1 already purchased products 100-103, so FP-Growth should
    # recommend products 104 and 105 (aisle 5, rule {1}->5 lift 1.2)
    product_ids = set(fp["product_id"])
    assert 104 in product_ids or 105 in product_ids
    assert all(s == "FP-Growth" for s in fp["source"])


# ----------------------------------------------------------------------
# Test 7: FP-Growth recommendations when no rule matches user's aisles
# ----------------------------------------------------------------------


def test_fpgrowth_recommendations_no_match(engine):
    """User 2 bought only aisle 5; no rule antecedent is subset of {5}."""
    fp = engine.fpgrowth_recommendations(user_id=2, top_n=10)
    assert fp.empty


# ----------------------------------------------------------------------
# Test 8: FP-Growth excludes already-purchased products
# ----------------------------------------------------------------------


def test_fpgrowth_excludes_purchased_products(engine):
    """User 1 purchased product 103 (aisle 4); FP-Growth should not return it."""
    fp = engine.fpgrowth_recommendations(user_id=1, top_n=10)
    if not fp.empty:
        assert 103 not in set(fp["product_id"])


# ----------------------------------------------------------------------
# Test 9: Combined recommendations - ML priority dedup
# ----------------------------------------------------------------------


def test_combined_ml_priority_dedup(engine):
    """When a product appears in both ML and FP-Growth, ML entry wins."""
    combined = engine.combined_recommendations(user_id=1, top_n=20)
    assert not combined.empty
    # ML products 100 and 101 appear first
    ml_rows = combined[combined["source"] == "ML"]
    assert list(ml_rows["product_id"]) == [100, 101]
    # No duplicate product_ids
    assert combined["product_id"].is_unique


# ----------------------------------------------------------------------
# Test 10: Combined recommendations respects top_n limit
# ----------------------------------------------------------------------


def test_combined_top_n_limit(engine):
    """Combined results are capped at top_n."""
    combined = engine.combined_recommendations(user_id=1, top_n=2)
    assert len(combined) <= 2
    assert list(combined["rank"]) == [1, 2]


# ----------------------------------------------------------------------
# Test 11: User summary
# ----------------------------------------------------------------------


def test_user_summary(engine):
    """User summary has expected structure and values."""
    summary = engine.user_summary(user_id=1)
    assert summary["user_id"] == 1
    assert summary["total_orders"] == 2
    assert summary["total_products_purchased"] == 4
    assert summary["has_ml_predictions"] is True
    assert summary["ml_prediction_count"] == 2


def test_user_summary_nonexistent_user(engine):
    """User 999 summary has zero counts."""
    summary = engine.user_summary(user_id=999)
    assert summary["user_id"] == 999
    assert summary["total_orders"] == 0
    assert summary["total_products_purchased"] == 0
    assert summary["has_ml_predictions"] is False


# ----------------------------------------------------------------------
# Test 12: Missing file returns empty DataFrame
# ----------------------------------------------------------------------


def test_missing_file_returns_empty():
    """Loading a non-existent CSV returns an empty DataFrame, not an error."""
    missing = Path(TEST_DATA) / "does_not_exist.csv"
    result = load_rules(path=missing)
    assert result.empty
    result = load_predictions(path=missing)
    assert result.empty
