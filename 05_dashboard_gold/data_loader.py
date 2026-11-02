
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AISLES_FILE,
    EXPECTED_AISLES_COLUMNS,
    EXPECTED_ORDER_PRODUCTS_COLUMNS,
    EXPECTED_ORDERS_COLUMNS,
    EXPECTED_PREDICTIONS_COLUMNS,
    EXPECTED_PRODUCTS_COLUMNS,
    EXPECTED_RULES_COLUMNS,
    ORDER_PRODUCTS_FILE,
    ORDERS_FILE,
    PREDICTIONS_FILE,
    PRODUCTS_FILE,
    RULES_FILE,
)


def _validate_columns(df: pd.DataFrame, expected: set[str], name: str) -> None:
    """Raise ValueError if required columns are missing."""
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} lacks required columns: {', '.join(sorted(missing))}"
        )


@st.cache_data(show_spinner=False)
def load_rules(path: Path = RULES_FILE) -> pd.DataFrame:
    """Load FP-Growth aisle-level rules and parse antecedent_ids into sets."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, dtype={"antecedent_ids": "string"})
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_RULES_COLUMNS, path.name)

    df = df.copy()
    df["consequent_ids"] = pd.to_numeric(df["consequent_ids"], errors="coerce")
    df["support"] = pd.to_numeric(df["support"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["lift"] = pd.to_numeric(df["lift"], errors="coerce")

    df = df.dropna(subset=["consequent_ids", "support", "confidence", "lift"])

    def parse_antecedent(value):
        if pd.isna(value) or str(value).strip() == "":
            return set()
        parts = str(value).split(",")
        result = set()
        for part in parts:
            part = part.strip()
            if part:
                try:
                    result.add(int(part))
                except ValueError:
                    continue
        return result

    df["antecedent_set"] = df["antecedent_ids"].apply(parse_antecedent)
    df = df[df["antecedent_set"].apply(len) > 0]

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_predictions(path: Path = PREDICTIONS_FILE) -> pd.DataFrame:
    """Load Spark MLlib reorder predictions."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_PREDICTIONS_COLUMNS, path.name)

    df = df.copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["rank_in_user"] = pd.to_numeric(df["rank_in_user"], errors="coerce")

    df = df.dropna(subset=["user_id", "product_id", "probability"])
    df["user_id"] = df["user_id"].astype("int64")
    df["product_id"] = df["product_id"].astype("int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_orders(path: Path = ORDERS_FILE) -> pd.DataFrame:
    """Load Silver-layer orders."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_ORDERS_COLUMNS, path.name)

    df = df.copy()
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["order_number"] = pd.to_numeric(df["order_number"], errors="coerce")

    df = df.dropna(subset=["order_id", "user_id"])
    df["order_id"] = df["order_id"].astype("int64")
    df["user_id"] = df["user_id"].astype("int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_order_products(path: Path = ORDER_PRODUCTS_FILE) -> pd.DataFrame:
    """Load Silver-layer order_products."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_ORDER_PRODUCTS_COLUMNS, path.name)

    df = df.copy()
    df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")

    df = df.dropna(subset=["order_id", "product_id"])
    df["order_id"] = df["order_id"].astype("int64")
    df["product_id"] = df["product_id"].astype("int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_products(path: Path = PRODUCTS_FILE) -> pd.DataFrame:
    """Load product reference data."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_PRODUCTS_COLUMNS, path.name)

    df = df.copy()
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["aisle_id"] = pd.to_numeric(df["aisle_id"], errors="coerce")

    df = df.dropna(subset=["product_id", "aisle_id"])
    df["product_id"] = df["product_id"].astype("int64")
    df["aisle_id"] = df["aisle_id"].astype("int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_aisles(path: Path = AISLES_FILE) -> pd.DataFrame:
    """Load aisle reference data."""
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    _validate_columns(df, EXPECTED_AISLES_COLUMNS, path.name)

    df = df.copy()
    df["aisle_id"] = pd.to_numeric(df["aisle_id"], errors="coerce")
    df = df.dropna(subset=["aisle_id"])
    df["aisle_id"] = df["aisle_id"].astype("int64")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load all data sources and return a dict keyed by name."""
    return {
        "rules": load_rules(),
        "predictions": load_predictions(),
        "orders": load_orders(),
        "order_products": load_order_products(),
        "products": load_products(),
        "aisles": load_aisles(),
    }


def get_user_purchased_aisles(
    user_id: int,
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
) -> set[int]:
    """Return the set of aisle_ids a user has purchased from."""
    if orders.empty or order_products.empty or products.empty:
        return set()

    user_orders = orders[orders["user_id"] == user_id]
    if user_orders.empty:
        return set()

    user_order_ids = set(user_orders["order_id"].astype("int64"))

    user_op = order_products[
        order_products["order_id"].isin(user_order_ids)
    ]
    if user_op.empty:
        return set()

    user_product_ids = set(user_op["product_id"].astype("int64"))

    user_products = products[products["product_id"].isin(user_product_ids)]
    if user_products.empty:
        return set()

    return set(user_products["aisle_id"].dropna().astype("int64"))


def get_user_purchased_products(
    user_id: int,
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
) -> set[int]:
    """Return the set of product_ids a user has already purchased."""
    if orders.empty or order_products.empty:
        return set()

    user_orders = orders[orders["user_id"] == user_id]
    if user_orders.empty:
        return set()

    user_order_ids = set(user_orders["order_id"].astype("int64"))

    user_op = order_products[
        order_products["order_id"].isin(user_order_ids)
    ]
    if user_op.empty:
        return set()

    return set(user_op["product_id"].dropna().astype("int64"))
