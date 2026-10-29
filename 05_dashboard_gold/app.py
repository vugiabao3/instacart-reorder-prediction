
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEEK1_DIR = PROJECT_ROOT / "reports" / "week1"
INPUT_DIR = WEEK1_DIR / "input"
OUTPUT_DIR = WEEK1_DIR / "output"

RECONCILIATION_SCRIPT = (
    PROJECT_ROOT / "05_dashboard_gold" / "reconciliation.py"
)

LAYER_COUNTS_FILE = INPUT_DIR / "layer_counts.csv"
RECONCILIATION_FILE = OUTPUT_DIR / "reconciliation_report.csv"
ISSUES_FILE = OUTPUT_DIR / "validation_issues.csv"

RULES_FILE = INPUT_DIR / "association_rules.csv"
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Instacart | NIC Big Data",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Instacart Market Basket Analysis")
st.caption(
    "NIC Big Data | Thành viên 5 — Dashboard và kiểm soát chất lượng dữ liệu"
)


# ============================================================
# COMMON HELPERS
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV safely. Return an empty DataFrame if unavailable."""

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Không đọc được {path.name}: {exc}")
        return pd.DataFrame()


def format_number(value) -> str:
    """Format numeric values without inventing missing values."""

    if pd.isna(value):
        return "Chưa có"

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def show_missing_data(filename: str) -> None:
    st.info(
        f"Chưa có dữ liệu **{filename}**. "
        "Hãy yêu cầu thành viên phụ trách xuất file theo "
        "hợp đồng dữ liệu của nhóm."
    )


def normalize_layer_name(value: str) -> str:
    mapping = {
        "source": "Source",
        "mariadb": "Source",
        "bronze": "Bronze",
        "hdfs": "Bronze",
        "silver": "Silver",
    }
    return mapping.get(str(value).strip().lower(), str(value).strip())


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Điều hướng")
    st.markdown(
        """
        **Phạm vi tuần 1**
        - Kiểm tra dữ liệu Source/Bronze/Silver
        - Tổng hợp lỗi dữ liệu
        - Chuẩn bị khu vực hiển thị FP-Growth
        - Chuẩn bị khu vực hiển thị kết quả ML
        """
    )

    st.divider()

    st.subheader("Tệp đầu vào")
    st.caption(f"Layer counts: {LAYER_COUNTS_FILE.name}")
    st.caption(f"FP-Growth: {RULES_FILE.name}")
    st.caption(f"ML metrics: {MODEL_METRICS_FILE.name}")

    if st.button("Chạy lại đối chiếu dữ liệu", use_container_width=True):
        if not RECONCILIATION_SCRIPT.exists():
            st.error("Không tìm thấy reconciliation.py.")
        elif not LAYER_COUNTS_FILE.exists():
            st.error(
                "Chưa có layer_counts.csv. "
                "Hãy nhận số liệu từ nhóm trước khi chạy."
            )
        else:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(RECONCILIATION_SCRIPT),
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )

                if result.returncode == 0:
                    st.success("Đã chạy xong script đối chiếu.")
                    if result.stdout:
                        st.code(result.stdout)
                    st.rerun()
                else:
                    st.error("Script đối chiếu chạy thất bại.")
                    if result.stderr:
                        st.code(result.stderr)
                    if result.stdout:
                        st.code(result.stdout)

            except subprocess.TimeoutExpired:
                st.error("Script chạy quá 120 giây.")
            except OSError as exc:
                st.error(f"Không chạy được script: {exc}")


# ============================================================
# LOAD DATA
# ============================================================

layer_counts = load_csv(LAYER_COUNTS_FILE)
reconciliation = load_csv(RECONCILIATION_FILE)
issues = load_csv(ISSUES_FILE)
rules = load_csv(RULES_FILE)
model_metrics = load_csv(MODEL_METRICS_FILE)
summary_metrics = load_csv(SUMMARY_METRICS_FILE)


# ============================================================
# TABS
# ============================================================

overview_tab, reconciliation_tab, quality_tab, rules_tab, ml_tab = (
    st.tabs(
        [
            "Tổng quan",
            "Đối chiếu dữ liệu",
            "Chất lượng dữ liệu",
            "FP-Growth",
            "Dự đoán mua lại",
        ]
    )
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with overview_tab:
    st.subheader("Tổng quan tiến độ dữ liệu")

    if summary_metrics.empty:
        st.info(
            "Chưa có summary_metrics.csv. "
            "Khi các thành viên bàn giao số liệu thật, "
            "bạn có thể hiển thị các chỉ số tổng quan tại đây."
        )
    else:
        required = {"metric", "value"}

        if required.issubset(summary_metrics.columns):
            metrics = summary_metrics.dropna(subset=["metric"])
            metrics = metrics.head(4)

            if not metrics.empty:
                columns = st.columns(len(metrics))

                for index, (_, row) in enumerate(metrics.iterrows()):
                    with columns[index]:
                        st.metric(
                            label=str(row["metric"]),
                            value=format_number(row["value"]),
                        )

            st.dataframe(
                summary_metrics,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.error(
                "summary_metrics.csv cần hai cột: metric, value."
            )

    st.markdown("### Trạng thái đối chiếu")

    if reconciliation.empty:
        show_missing_data("reconciliation_report.csv")
        st.caption(
            "Tạo layer_counts.csv rồi bấm "
            "'Chạy lại đối chiếu dữ liệu' ở thanh bên."
        )
    elif "status" not in reconciliation.columns:
        st.error("reconciliation_report.csv thiếu cột status.")
    else:
        status_counts = (
            reconciliation["status"]
            .fillna("UNKNOWN")
            .value_counts()
            .rename_axis("status")
            .reset_index(name="table_count")
        )

        columns = st.columns(4)
        status_values = reconciliation["status"].fillna("UNKNOWN")

        columns[0].metric(
            "Tổng bảng",
            format_number(len(reconciliation)),
        )
        columns[1].metric(
            "Khớp cả 3 tầng",
            format_number((status_values == "MATCH_ALL").sum()),
        )
        columns[2].metric(
            "Bronze lệch nguồn",
            format_number(
                (status_values == "BRONZE_MISMATCH").sum()
            ),
        )
        columns[3].metric(
            "Cần kiểm tra",
            format_number(
                (
                    status_values.isin(
                        [
                            "SILVER_CHANGED_REVIEW",
                            "MISSING_DATA",
                        ]
                    )
                ).sum()
            ),
        )

        fig = px.bar(
            status_counts,
            x="status",
            y="table_count",
            title="Số bảng theo trạng thái đối chiếu",
            labels={
                "status": "Trạng thái",
                "table_count": "Số bảng",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Số dòng theo tầng")

    if layer_counts.empty:
        show_missing_data("layer_counts.csv")
    else:
        required = {"table_name", "layer", "row_count"}

        if required.issubset(layer_counts.columns):
            plot_data = layer_counts.copy()
            plot_data["layer"] = plot_data["layer"].map(
                normalize_layer_name
            )
            plot_data["row_count"] = pd.to_numeric(
                plot_data["row_count"], errors="coerce"
            )
            plot_data = plot_data.dropna(subset=["row_count"])

            if not plot_data.empty:
                fig = px.bar(
                    plot_data,
                    x="table_name",
                    y="row_count",
                    color="layer",
                    barmode="group",
                    title="So sánh số dòng Source, Bronze và Silver",
                    labels={
                        "table_name": "Bảng",
                        "row_count": "Số dòng",
                        "layer": "Tầng dữ liệu",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Chưa có số dòng hợp lệ để vẽ biểu đồ.")
        else:
            st.error(
                "layer_counts.csv cần các cột "
                "table_name, layer, row_count."
            )


# ============================================================
# TAB 2: RECONCILIATION
# ============================================================

with reconciliation_tab:
    st.subheader("Đối chiếu Source → Bronze → Silver")

    st.markdown(
        """
        - **Source:** số dòng trong MariaDB.
        - **Bronze:** số dòng đã chuyển sang HDFS.
        - **Silver:** số dòng sau quy trình làm sạch.
        """
    )

    if reconciliation.empty:
        show_missing_data("reconciliation_report.csv")
    else:
        display_columns = [
            "table_name",
            "source_rows",
            "bronze_rows",
            "silver_rows",
            "source_to_bronze_diff",
            "bronze_to_silver_diff",
            "status",
            "explanation",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in reconciliation.columns
        ]

        st.dataframe(
            reconciliation[available_columns],
            use_container_width=True,
            hide_index=True,
        )

        if "status" in reconciliation.columns:
            review_rows = reconciliation[
                reconciliation["status"].isin(
                    [
                        "BRONZE_MISMATCH",
                        "SILVER_CHANGED_REVIEW",
                        "MISSING_DATA",
                    ]
                )
            ]

            if not review_rows.empty:
                st.warning(
                    f"Có {len(review_rows)} bảng cần kiểm tra. "
                    "Đọc cột explanation và trao đổi với "
                    "thành viên phụ trách tầng dữ liệu."
                )
            else:
                st.success(
                    "Không có bảng nào mang trạng thái "
                    "lệch Bronze, Silver cần xem xét hoặc thiếu số liệu."
                )

        csv_bytes = reconciliation.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Tải báo cáo đối chiếu CSV",
            data=csv_bytes,
            file_name="reconciliation_report.csv",
            mime="text/csv",
        )

    st.caption(
        "Số dòng Silver khác Bronze cần được giải thích bằng "
        "quy tắc làm sạch; không tự động kết luận là lỗi."
    )


# ============================================================
# TAB 3: DATA QUALITY
# ============================================================

with quality_tab:
    st.subheader("Chất lượng dữ liệu")

    st.markdown(
        """
        Khu vực này hiển thị các vấn đề được script phát hiện.
        Những kiểm tra như khóa ngoại, NULL và khóa trùng cần được
        các thành viên phụ trách nguồn/làm sạch cung cấp kết quả.
        """
    )

    if issues.empty:
        show_missing_data("validation_issues.csv")
    else:
        st.metric("Số vấn đề được ghi nhận", format_number(len(issues)))

        if "severity" in issues.columns:
            severity_counts = (
                issues["severity"]
                .fillna("UNKNOWN")
                .value_counts()
                .rename_axis("severity")
                .reset_index(name="issue_count")
            )

            fig = px.bar(
                severity_counts,
                x="severity",
                y="issue_count",
                title="Vấn đề theo mức độ",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            issues,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Số liệu kiểm tra từ các thành viên")

    quality_checks_path = INPUT_DIR / "quality_checks.csv"
    quality_checks = load_csv(quality_checks_path)

    if quality_checks.empty:
        show_missing_data("quality_checks.csv")
        st.caption(
            "Có thể dùng file này để tổng hợp số NULL, "
            "khóa trùng và khóa ngoại không hợp lệ."
        )
    else:
        st.dataframe(
            quality_checks,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 4: ASSOCIATION RULES
# ============================================================

with rules_tab:
    st.subheader("Luật kết hợp sản phẩm — Spark FP-Growth")

    if rules.empty:
        show_missing_data("association_rules.csv")
        st.caption(
            "Thành viên 3 cần xuất luật kết hợp từ Spark "
            "và bàn giao theo định dạng đã thống nhất."
        )
    else:
        required = {
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
        }

        if not required.issubset(rules.columns):
            st.error(
                "association_rules.csv cần các cột: "
                "antecedent, consequent, support, confidence, lift."
            )
        else:
            rules = rules.copy()

            for column in ["support", "confidence", "lift"]:
                rules[column] = pd.to_numeric(
                    rules[column], errors="coerce"
                )

            rules = rules.dropna(
                subset=["support", "confidence", "lift"]
            )

            if rules.empty:
                st.warning("Không có luật với các chỉ số hợp lệ.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    min_confidence = st.slider(
                        "Confidence tối thiểu",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                    )

                with col2:
                    min_lift = st.slider(
                        "Lift tối thiểu",
                        min_value=0.0,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                    )

                filtered = rules[
                    (rules["confidence"] >= min_confidence)
                    & (rules["lift"] >= min_lift)
                ].copy()

                st.metric("Số luật sau lọc", format_number(len(filtered)))

                st.dataframe(
                    filtered.sort_values(
                        "lift", ascending=False
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if not filtered.empty:
                    fig = px.scatter(
                        filtered,
                        x="support",
                        y="confidence",
                        size="lift",
                        hover_data=["antecedent", "consequent"],
                        title="Support, Confidence và Lift",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Các chỉ số phải được tính từ kết quả Spark thật. "
                    "Lift > 1 gợi ý mối liên hệ dương, nhưng không "
                    "chứng minh quan hệ nhân quả."
                )


# ============================================================
# TAB 5: MACHINE LEARNING
# ============================================================

with ml_tab:
    st.subheader("Dự đoán mua lại — Spark MLlib")

    if model_metrics.empty:
        show_missing_data("model_metrics.csv")
        st.caption(
            "Thành viên 4 cần bàn giao chỉ số đánh giá trên "
            "tập kiểm thử, kèm mô tả cách chia train/test."
        )
    else:
        required = {"metric", "value"}

        if not required.issubset(model_metrics.columns):
            st.error(
                "model_metrics.csv cần các cột metric và value."
            )
        else:
            metrics = model_metrics.dropna(subset=["metric"]).copy()

            # Display F1/AUC when available; retain other metrics too.
            metric_lookup = {
                str(row["metric"]).strip().upper(): row["value"]
                for _, row in metrics.iterrows()
            }

            f1_col, auc_col = st.columns(2)

            with f1_col:
                st.metric(
                    "F1",
                    str(metric_lookup.get("F1", "Chưa có")),
                )

            with auc_col:
                st.metric(
                    "AUC",
                    str(metric_lookup.get("AUC", "Chưa có")),
                )

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True,
            )

    st.info(
        "F1 và AUC chỉ có ý nghĩa khi đi kèm cách chia tập dữ liệu, "
        "định nghĩa nhãn và phương pháp đánh giá. Không so sánh "
        "các kết quả được tính trên các tập khác nhau mà không "
        "nêu rõ điều kiện."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Instacart Reorder Prediction | NIC Big Data | Tuần 1. "
    "Dashboard chỉ trình bày dữ liệu nhóm đã bàn giao và kiểm chứng; "
    "không tự sinh số liệu thực nghiệm."
)
