
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT / "reports" / "week1" / "input" / "layer_counts.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "week1" / "output"
)

EXPECTED_TABLES = [
    "orders",
    "products",
    "aisles",
    "departments",
    "order_products__prior",
    "order_products__train",
]

EXPECTED_LAYERS = ["Source", "Bronze", "Silver"]

REQUIRED_COLUMNS = {"table_name", "layer", "row_count"}


# ============================================================
# INPUT VALIDATION
# ============================================================

def load_layer_counts(input_path: Path) -> pd.DataFrame:
    """Read and validate the team's layer-count CSV."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file đầu vào: {input_path}\n"
            "Hãy tạo reports/week1/input/layer_counts.csv "
            "theo mẫu đã thống nhất với nhóm."
        )

    df = pd.read_csv(input_path, dtype={"table_name": "string"})

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "File layer_counts.csv thiếu cột bắt buộc: "
            + ", ".join(sorted(missing_columns))
        )

    # Keep only the required fields plus optional notes.
    if "notes" not in df.columns:
        df["notes"] = ""

    df["table_name"] = df["table_name"].astype("string").str.strip()
    df["layer"] = df["layer"].astype("string").str.strip().str.lower()
    df["notes"] = df["notes"].fillna("").astype(str).str.strip()

    layer_mapping = {
        "source": "Source",
        "mariadb": "Source",
        "bronze": "Bronze",
        "hdfs": "Bronze",
        "silver": "Silver",
    }
    df["layer"] = df["layer"].map(layer_mapping).fillna(df["layer"])

    df["row_count"] = pd.to_numeric(df["row_count"], errors="coerce")

    if df["table_name"].isna().any() or (df["table_name"] == "").any():
        raise ValueError("Có table_name rỗng trong file đầu vào.")

    if df["row_count"].isna().any():
        bad_rows = df[df["row_count"].isna()].index.tolist()
        raise ValueError(
            f"row_count không hợp lệ tại các dòng CSV: {bad_rows}. "
            "Hãy dùng số nguyên không âm."
        )

    # Counts should be whole, non-negative numbers.
    if (df["row_count"] < 0).any():
        raise ValueError("row_count không được là số âm.")

    if ((df["row_count"] % 1) != 0).any():
        raise ValueError("row_count phải là số nguyên.")

    df["row_count"] = df["row_count"].astype("int64")

    invalid_layers = sorted(set(df["layer"]) - set(EXPECTED_LAYERS))
    if invalid_layers:
        raise ValueError(
            "Giá trị layer không được hỗ trợ: "
            + ", ".join(invalid_layers)
            + ". Dùng Source, Bronze hoặc Silver."
        )

    # Duplicate rows make reconciliation ambiguous.
    duplicate_mask = df.duplicated(
        subset=["table_name", "layer"], keep=False
    )
    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask, ["table_name", "layer"]
        ].to_dict("records")
        raise ValueError(
            "Có nhiều dòng cho cùng một table_name + layer. "
            f"Hãy xử lý các dòng trùng trước: {duplicates}"
        )

    return df


# ============================================================
# RECONCILIATION
# ============================================================

def reconcile_counts(
    df: pd.DataFrame,
    expected_tables: list[str],
) -> pd.DataFrame:
    """Compare Source, Bronze and Silver counts per table."""

    indexed = df.set_index(["table_name", "layer"])["row_count"]

    def get_count(table: str, layer: str):
        return indexed.get((table, layer), pd.NA)

    results = []

    for table in expected_tables:
        source = get_count(table, "Source")
        bronze = get_count(table, "Bronze")
        silver = get_count(table, "Silver")

        missing = [
            layer
            for layer, value in [
                ("Source", source),
                ("Bronze", bronze),
                ("Silver", silver),
            ]
            if pd.isna(value)
        ]

        source_bronze_diff = (
            int(bronze) - int(source)
            if pd.notna(source) and pd.notna(bronze)
            else pd.NA
        )

        bronze_silver_diff = (
            int(silver) - int(bronze)
            if pd.notna(bronze) and pd.notna(silver)
            else pd.NA
        )

        if missing:
            status = "MISSING_DATA"
            explanation = "Thiếu số liệu: " + ", ".join(missing)

        elif int(source) != int(bronze):
            status = "BRONZE_MISMATCH"
            explanation = (
                "Số dòng Bronze không khớp Source. "
                "Kiểm tra Sqoop, phạm vi dữ liệu và cách đếm."
            )

        elif int(bronze) != int(silver):
            status = "SILVER_CHANGED_REVIEW"
            explanation = (
                "Silver có số dòng khác Bronze. "
                "Cần đối chiếu quy tắc làm sạch và ghi rõ nguyên nhân."
            )

        else:
            status = "MATCH_ALL"
            explanation = "Số dòng bằng nhau ở cả ba tầng."

        results.append(
            {
                "table_name": table,
                "source_rows": source,
                "bronze_rows": bronze,
                "silver_rows": silver,
                "source_to_bronze_diff": source_bronze_diff,
                "bronze_to_silver_diff": bronze_silver_diff,
                "status": status,
                "explanation": explanation,
            }
        )

    result = pd.DataFrame(results)

    count_columns = [
        "source_rows",
        "bronze_rows",
        "silver_rows",
        "source_to_bronze_diff",
        "bronze_to_silver_diff",
    ]

    for column in count_columns:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        ).astype("Int64")

    return result


def build_input_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Report unexpected table names and absent expected layers."""

    issues = []

    for table in sorted(set(df["table_name"].dropna().astype(str))):
        if table not in EXPECTED_TABLES:
            issues.append(
                {
                    "severity": "WARNING",
                    "table_name": table,
                    "issue": "UNEXPECTED_TABLE",
                    "details": "Bảng chưa nằm trong danh sách 6 bảng chuẩn.",
                }
            )

    for table in EXPECTED_TABLES:
        table_rows = df[df["table_name"] == table]
        existing = set(table_rows["layer"])

        for layer in EXPECTED_LAYERS:
            if layer not in existing:
                issues.append(
                    {
                        "severity": "WARNING",
                        "table_name": table,
                        "issue": "MISSING_LAYER_COUNT",
                        "details": f"Chưa có số dòng cho tầng {layer}.",
                    }
                )

    return pd.DataFrame(
        issues,
        columns=["severity", "table_name", "issue", "details"],
    )


# ============================================================
# REPORT OUTPUT
# ============================================================

def save_reports(
    source_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    reconciliation_path = output_dir / "reconciliation_report.csv"
    issues_path = output_dir / "validation_issues.csv"
    summary_path = output_dir / "reconciliation_summary.json"

    reconciliation_df.to_csv(
        reconciliation_path, index=False, encoding="utf-8-sig"
    )
    issues_df.to_csv(
        issues_path, index=False, encoding="utf-8-sig"
    )

    total_tables = len(reconciliation_df)
    match_count = int(
        (reconciliation_df["status"] == "MATCH_ALL").sum()
    )
    bronze_mismatch_count = int(
        (reconciliation_df["status"] == "BRONZE_MISMATCH").sum()
    )
    silver_review_count = int(
        (
            reconciliation_df["status"]
            == "SILVER_CHANGED_REVIEW"
        ).sum()
    )
    missing_count = int(
        (reconciliation_df["status"] == "MISSING_DATA").sum()
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(source_df)),
        "expected_table_count": total_tables,
        "match_all_count": match_count,
        "bronze_mismatch_count": bronze_mismatch_count,
        "silver_changed_review_count": silver_review_count,
        "missing_data_count": missing_count,
        "validation_issue_count": int(len(issues_df)),
        "reports": {
            "reconciliation": reconciliation_path.name,
            "validation_issues": issues_path.name,
        },
        "note": (
            "Silver count differences require investigation and an "
            "explanation of cleaning rules; they are not automatically "
            "proof of an error."
        ),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RECONCILIATION SUMMARY ===")
    print(f"Input rows:               {len(source_df)}")
    print(f"Expected tables:          {total_tables}")
    print(f"All layers match:         {match_count}")
    print(f"Bronze mismatch:          {bronze_mismatch_count}")
    print(f"Silver requires review:   {silver_review_count}")
    print(f"Missing data:             {missing_count}")
    print(f"Validation issues:        {len(issues_df)}")
    print(f"\nReport: {reconciliation_path}")
    print(f"Issues: {issues_path}")
    print(f"Summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Đối chiếu số dòng Source, Bronze và Silver "
            "cho dự án Instacart."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Đường dẫn tới layer_counts.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu báo cáo đối chiếu",
    )
    args = parser.parse_args()

    try:
        input_df = load_layer_counts(args.input)
        reconciliation_df = reconcile_counts(
            input_df, EXPECTED_TABLES
        )
        issues_df = build_input_issues(input_df)

        save_reports(
            input_df,
            reconciliation_df,
            issues_df,
            args.output_dir,
        )

        # Exit code 0 means the script completed successfully.
        # It does not mean every table passed validation.
        return 0

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
