
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_TOP_N_COMBINED,
    DEFAULT_TOP_N_FPGROWTH,
    DEFAULT_TOP_N_ML,
    MIN_CONFIDENCE,
    MIN_LIFT,
)
from data_loader import (
    get_user_purchased_aisles,
    get_user_purchased_products,
    load_all_data,
)


class RecommendationEngine:
    """Merge Spark MLlib predictions with FP-Growth aisle rules.

    ML predictions provide user-product-level reorder probabilities.
    FP-Growth rules provide aisle-level association: if a user has bought
    from all antecedent aisles, products in consequent aisles are suggested.

    No probabilities or lift values are fabricated: ML scores come from
    gold_reorder_predictions.csv, rule scores come from
    gold_rules_aisle.csv.
    """

    def __init__(
        self,
        rules: pd.DataFrame | None = None,
        predictions: pd.DataFrame | None = None,
        orders: pd.DataFrame | None = None,
        order_products: pd.DataFrame | None = None,
        products: pd.DataFrame | None = None,
        aisles: pd.DataFrame | None = None,
        data_dir: Path | None = None,
    ):
        if data_dir is not None and any(
            x is not None for x in (rules, predictions, orders, order_products, products, aisles)
        ):
            raise ValueError(
                "Provide either data_dir or individual dataframes, not both."
            )

        if data_dir is not None:
            data = load_all_data()
            self.rules = data["rules"]
            self.predictions = data["predictions"]
            self.orders = data["orders"]
            self.order_products = data["order_products"]
            self.products = data["products"]
            self.aisles = data["aisles"]
        else:
            self.rules = rules if rules is not None else pd.DataFrame()
            self.predictions = (
                predictions if predictions is not None else pd.DataFrame()
            )
            self.orders = orders if orders is not None else pd.DataFrame()
            self.order_products = (
                order_products if order_products is not None else pd.DataFrame()
            )
            self.products = products if products is not None else pd.DataFrame()
            self.aisles = aisles if aisles is not None else pd.DataFrame()

        self._product_aisle_map: dict[int, int] = {}
        self._aisle_name_map: dict[int, str] = {}
        self._product_name_map: dict[int, str] = {}
        self._build_maps()
        self._parse_rules()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_maps(self) -> None:
        """Build lookup maps from products and aisles dataframes."""
        self._product_aisle_map = {}
        self._product_name_map = {}
        self._aisle_name_map = {}

        if not self.products.empty:
            for _, row in self.products.iterrows():
                pid = int(row["product_id"])
                self._product_name_map[pid] = str(row.get("product_name", ""))
                aid = row.get("aisle_id")
                if pd.notna(aid):
                    self._product_aisle_map[pid] = int(aid)

        if not self.aisles.empty:
            for _, row in self.aisles.iterrows():
                aid = int(row["aisle_id"])
                self._aisle_name_map[aid] = str(row.get("aisle", ""))

    def _parse_rules(self) -> None:
        """Parse antecedent_ids into antecedent_set column on self.rules."""
        if self.rules.empty or "antecedent_ids" not in self.rules.columns:
            return

        if "antecedent_set" in self.rules.columns:
            return

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

        self.rules = self.rules.copy()
        self.rules["antecedent_set"] = self.rules["antecedent_ids"].apply(
            parse_antecedent
        )
        self.rules = self.rules[
            self.rules["antecedent_set"].apply(len) > 0
        ].reset_index(drop=True)

    def _aisle_name(self, aisle_id: int) -> str:
        return self._aisle_name_map.get(aisle_id, f"aisle_{aisle_id}")

    def _product_name(self, product_id: int) -> str:
        return self._product_name_map.get(
            product_id, f"product_{product_id}"
        )

    # ------------------------------------------------------------------
    # ML-based recommendations
    # ------------------------------------------------------------------

    def ml_recommendations(
        self, user_id: int, top_n: int = DEFAULT_TOP_N_ML
    ) -> pd.DataFrame:
        """Return ML-based reorder recommendations for a user.

        Columns: user_id, product_id, product_name, probability,
        source, rank.
        """
        if self.predictions.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "probability",
                    "source",
                    "rank",
                ]
            )

        user_preds = self.predictions[
            self.predictions["user_id"] == user_id
        ].copy()

        if user_preds.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "probability",
                    "source",
                    "rank",
                ]
            )

        user_preds = user_preds.sort_values(
            "probability", ascending=False
        ).head(top_n)

        user_preds["product_name"] = user_preds[
            "product_id"
        ].apply(self._product_name)
        user_preds["source"] = "ML"
        user_preds["rank"] = range(1, len(user_preds) + 1)

        return user_preds[
            [
                "user_id",
                "product_id",
                "product_name",
                "probability",
                "source",
                "rank",
            ]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # FP-Growth-based recommendations
    # ------------------------------------------------------------------

    def fpgrowth_recommendations(
        self, user_id: int, top_n: int = DEFAULT_TOP_N_FPGROWTH
    ) -> pd.DataFrame:
        """Return FP-Growth-based recommendations for a user.

        Logic:
        1. Find aisles the user has purchased from.
        2. Find rules whose antecedent aisle set is a subset of the user's
           purchased aisles.
        3. Collect consequent aisle IDs from those rules.
        4. Find products in those consequent aisles (excluding products the
           user has already purchased).
        5. Score each product by the best (max) lift of any matching rule.
        6. Return top_n products sorted by score desc.

        Columns: user_id, product_id, product_name, score, source, rank,
        rule_lift, rule_confidence, consequent_aisle_id, consequent_aisle.
        """
        if (
            self.rules.empty
            or self.products.empty
            or self.order_products.empty
            or self.orders.empty
        ):
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        user_aisles = get_user_purchased_aisles(
            user_id, self.orders, self.order_products, self.products
        )

        if not user_aisles:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        # Filter rules: confidence and lift thresholds
        valid_rules = self.rules[
            (self.rules["confidence"] >= MIN_CONFIDENCE)
            & (self.rules["lift"] >= MIN_LIFT)
        ].copy()

        if valid_rules.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        # Find rules whose antecedent is a subset of user's purchased aisles
        matching_rows = []
        for _, rule in valid_rules.iterrows():
            antecedent = rule["antecedent_set"]
            if antecedent and antecedent.issubset(user_aisles):
                consequent_id = int(rule["consequent_ids"])
                matching_rows.append(
                    {
                        "consequent_aisle_id": consequent_id,
                        "consequent_aisle": self._aisle_name(consequent_id),
                        "rule_lift": float(rule["lift"]),
                        "rule_confidence": float(rule["confidence"]),
                    }
                )

        if not matching_rows:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        matching_df = pd.DataFrame(matching_rows)

        # Find products in consequent aisles
        consequent_aisle_ids = set(matching_df["consequent_aisle_id"])

        candidate_products = self.products[
            self.products["aisle_id"].isin(consequent_aisle_ids)
        ].copy()

        if candidate_products.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        # Exclude products the user has already purchased
        purchased_products = get_user_purchased_products(
            user_id, self.orders, self.order_products
        )
        candidate_products = candidate_products[
            ~candidate_products["product_id"].isin(purchased_products)
        ]

        if candidate_products.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        # Join products with their best rule score
        product_scores = []
        for _, prod in candidate_products.iterrows():
            pid = int(prod["product_id"])
            aid = int(prod["aisle_id"])

            # Best rule for this aisle
            aisle_rules = matching_df[
                matching_df["consequent_aisle_id"] == aid
            ]
            if aisle_rules.empty:
                continue

            best = aisle_rules.loc[aisle_rules["rule_lift"].idxmax()]
            product_scores.append(
                {
                    "product_id": pid,
                    "product_name": str(prod.get("product_name", "")),
                    "score": float(best["rule_lift"]),
                    "rule_lift": float(best["rule_lift"]),
                    "rule_confidence": float(best["rule_confidence"]),
                    "consequent_aisle_id": aid,
                    "consequent_aisle": str(best["consequent_aisle"]),
                }
            )

        if not product_scores:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                ]
            )

        result = pd.DataFrame(product_scores)
        result = result.sort_values(
            ["score", "product_id"], ascending=[False, True]
        ).head(top_n)

        result["user_id"] = user_id
        result["source"] = "FP-Growth"
        result["rank"] = range(1, len(result) + 1)

        return result[
            [
                "user_id",
                "product_id",
                "product_name",
                "score",
                "source",
                "rank",
                "rule_lift",
                "rule_confidence",
                "consequent_aisle_id",
                "consequent_aisle",
            ]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Combined recommendations
    # ------------------------------------------------------------------

    def combined_recommendations(
        self, user_id: int, top_n: int = DEFAULT_TOP_N_COMBINED
    ) -> pd.DataFrame:
        """Combine ML and FP-Growth recommendations.

        ML recommendations come first (sorted by probability desc), then
        FP-Growth recommendations (sorted by lift desc). Products are
        deduplicated by product_id: if a product appears in both, the ML
        entry wins.

        Columns: user_id, product_id, product_name, score, source, rank,
        rule_lift, rule_confidence, consequent_aisle_id, consequent_aisle,
        probability.
        """
        ml_df = self.ml_recommendations(user_id, top_n=top_n)
        fp_df = self.fpgrowth_recommendations(user_id, top_n=top_n)

        if ml_df.empty and fp_df.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                    "probability",
                ]
            )

        # Normalize ML columns to combined schema
        if not ml_df.empty:
            ml_norm = ml_df.copy()
            ml_norm["score"] = ml_norm["probability"]
            ml_norm["rule_lift"] = pd.NA
            ml_norm["rule_confidence"] = pd.NA
            ml_norm["consequent_aisle_id"] = pd.NA
            ml_norm["consequent_aisle"] = ""
            ml_norm["probability"] = ml_norm["probability"]
        else:
            ml_norm = pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                    "probability",
                ]
            )

        # Normalize FP-Growth columns to combined schema
        if not fp_df.empty:
            fp_norm = fp_df.copy()
            fp_norm["probability"] = pd.NA
        else:
            fp_norm = pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "product_name",
                    "score",
                    "source",
                    "rank",
                    "rule_lift",
                    "rule_confidence",
                    "consequent_aisle_id",
                    "consequent_aisle",
                    "probability",
                ]
            )

        combined_cols = [
            "user_id",
            "product_id",
            "product_name",
            "score",
            "source",
            "rank",
            "rule_lift",
            "rule_confidence",
            "consequent_aisle_id",
            "consequent_aisle",
            "probability",
        ]

        ml_norm = ml_norm[combined_cols]
        fp_norm = fp_norm[combined_cols]

        # Concatenate, ML first so it takes priority in dedup
        combined = pd.concat([ml_norm, fp_norm], ignore_index=True)

        # Deduplicate by product_id, keeping first occurrence (ML priority)
        combined = combined.drop_duplicates(
            subset=["product_id"], keep="first"
        )

        # Re-rank
        combined = combined.sort_values(
            ["source", "score"], ascending=[True, False]
        )
        # ML first (source="ML" < "FP-Growth" alphabetically), then by score
        combined["rank"] = range(1, len(combined) + 1)
        combined = combined.head(top_n)

        return combined[combined_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # User summary
    # ------------------------------------------------------------------

    def user_summary(self, user_id: int) -> dict:
        """Return a summary dict for a user: purchase history and stats."""
        summary = {
            "user_id": user_id,
            "total_orders": 0,
            "total_products_purchased": 0,
            "unique_aisles": 0,
            "aisles": [],
            "has_ml_predictions": False,
            "ml_prediction_count": 0,
        }

        if not self.orders.empty:
            user_orders = self.orders[self.orders["user_id"] == user_id]
            summary["total_orders"] = int(len(user_orders))

        if not self.orders.empty and not self.order_products.empty:
            user_orders = self.orders[self.orders["user_id"] == user_id]
            if not user_orders.empty:
                user_order_ids = set(user_orders["order_id"].astype("int64"))
                user_op = self.order_products[
                    self.order_products["order_id"].isin(user_order_ids)
                ]
                summary["total_products_purchased"] = int(len(user_op))

        if (
            not self.orders.empty
            and not self.order_products.empty
            and not self.products.empty
        ):
            user_aisles = get_user_purchased_aisles(
                user_id, self.orders, self.order_products, self.products
            )
            summary["unique_aisles"] = len(user_aisles)
            summary["aisles"] = sorted(
                self._aisle_name(aid) for aid in user_aisles
            )

        if not self.predictions.empty:
            user_preds = self.predictions[
                self.predictions["user_id"] == user_id
            ]
            summary["has_ml_predictions"] = not user_preds.empty
            summary["ml_prediction_count"] = int(len(user_preds))

        return summary
