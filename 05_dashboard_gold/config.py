
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# DATA SOURCE PATHS (team Member 2 / Member 3 / Member 4 deliverables)
# ============================================================

# Member 3: FP-Growth aisle-level association rules
RULES_FILE = PROJECT_ROOT / "03_spark_baskets" / "gold_rules_aisle.csv"

# Member 4: Spark MLlib reorder predictions
PREDICTIONS_FILE = PROJECT_ROOT / "04_spark_ml" / "gold_reorder_predictions.csv"

# Member 2: Silver-layer purchase history
ORDERS_FILE = PROJECT_ROOT / "silver_orders.csv"
ORDER_PRODUCTS_FILE = PROJECT_ROOT / "silver_order_products.csv"

# Product / aisle reference data
PRODUCTS_FILE = PROJECT_ROOT / "data_sample" / "products.csv"
AISLES_FILE = PROJECT_ROOT / "data_sample" / "aisles.csv"

# ============================================================
# WEEK 1 REPORTS (existing dashboard data, preserved)
# ============================================================

WEEK1_DIR = PROJECT_ROOT / "reports" / "week1"
INPUT_DIR = WEEK1_DIR / "input"
OUTPUT_DIR = WEEK1_DIR / "output"

RECONCILIATION_SCRIPT = PROJECT_ROOT / "05_dashboard_gold" / "reconciliation.py"

LAYER_COUNTS_FILE = INPUT_DIR / "layer_counts.csv"
RECONCILIATION_FILE = OUTPUT_DIR / "reconciliation_report.csv"
ISSUES_FILE = OUTPUT_DIR / "validation_issues.csv"

MODEL_METRICS_FILE = INPUT_DIR / "model_metrics.csv"
SUMMARY_METRICS_FILE = INPUT_DIR / "summary_metrics.csv"

EXPECTED_TABLES = [
    "orders",
    "products",
    "aisles",
    "departments",
    "order_products__prior",
    "order_products__train",
]

# ============================================================
# WEEK 2 REPORTS (test data and validation output)
# ============================================================

WEEK2_DIR = PROJECT_ROOT / "reports" / "week2"
TEST_DATA_DIR = WEEK2_DIR / "test_data"
WEEK2_OUTPUT_DIR = WEEK2_DIR / "output"

# ============================================================
# RECOMMENDATION ENGINE SETTINGS
# ============================================================

DEFAULT_TOP_N_ML = 10
DEFAULT_TOP_N_FPGROWTH = 10
DEFAULT_TOP_N_COMBINED = 20

MIN_CONFIDENCE = 0.0
MIN_LIFT = 1.0

EXPECTED_RULES_COLUMNS = {
    "antecedent_ids",
    "consequent_ids",
    "support",
    "confidence",
    "lift",
    "level",
}

EXPECTED_PREDICTIONS_COLUMNS = {
    "user_id",
    "product_id",
    "product_name",
    "probability",
    "predicted",
    "rank_in_user",
}

EXPECTED_ORDERS_COLUMNS = {
    "order_id",
    "user_id",
    "eval_set",
    "order_number",
}

EXPECTED_ORDER_PRODUCTS_COLUMNS = {
    "order_id",
    "product_id",
}

EXPECTED_PRODUCTS_COLUMNS = {
    "product_id",
    "product_name",
    "aisle_id",
}

EXPECTED_AISLES_COLUMNS = {
    "aisle_id",
    "aisle",
}
