
# -*- coding: utf-8 -*-
"""
step1_build_features.py

Nhiệm vụ của Thành viên 4:
- Đọc dữ liệu Silver từ CSV.
- Tính các nhóm đặc trưng User, Product, User-Product.
- Gán nhãn label = 1/0 từ tập train.
- Ghi gold_features Parquet bằng PyArrow để tránh lỗi winutils.exe
  khi Spark ghi file trên Windows.
- Kiểm tra số dòng, phân bố nhãn và Data Lineage.

Đầu vào:
    02_hive_cleaning/silver_orders.csv
    02_hive_cleaning/silver_order_products.csv

Đầu ra:
    04_spark_ml/storage/gold/gold_features/part-00000.parquet
"""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORDERS_CSV = (
    PROJECT_ROOT
    / "02_hive_cleaning"
    / "silver_orders.csv"
)

ORDER_PRODUCTS_CSV = (
    PROJECT_ROOT
    / "02_hive_cleaning"
    / "silver_order_products.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "04_spark_ml"
    / "storage"
    / "gold"
    / "gold_features"
)

BATCH_SIZE = 10000


# ============================================================
# 2. KHỞI TẠO SPARK
# ============================================================

def create_spark_session():
    return (
        SparkSession.builder
        .appName("TV4_Step1_Build_Features")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# ============================================================
# 3. GHI SPARK DATAFRAME SANG PARQUET BẰNG PYARROW
# ============================================================

def save_parquet_with_pyarrow(dataframe, output_dir, batch_size=10000):
    """
    Ghi Spark DataFrame thành một file Parquet bằng PyArrow.

    Không gọi toPandas() trên toàn bộ DataFrame.
    Dữ liệu được lấy về theo từng lô để hạn chế RAM.

    Lưu ý:
    toLocalIterator() vẫn chuyển dữ liệu từ Spark về Python driver.
    """

    output_dir = Path(output_dir)
    output_file = output_dir / "part-00000.parquet"

    print(f"-> Thư mục đầu ra: {output_dir}")
    print(f"-> Kích thước mỗi lô: {batch_size:,} dòng")

    # Xóa kết quả cũ để tránh giữ dữ liệu từ lần chạy trước.
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    batch_rows = []
    total_rows = 0
    writer = None

    try:
        for row in dataframe.toLocalIterator():
            batch_rows.append(row.asDict(recursive=True))

            if len(batch_rows) >= batch_size:
                table = pa.Table.from_pylist(batch_rows)

                if writer is None:
                    writer = pq.ParquetWriter(
                        str(output_file),
                        table.schema,
                        compression="snappy"
                    )

                # Bảo đảm schema giữa các lô thống nhất.
                if not table.schema.equals(
                    writer.schema,
                    check_metadata=False
                ):
                    raise ValueError(
                        "Schema giữa các lô dữ liệu không đồng nhất."
                    )

                writer.write_table(table)

                total_rows += len(batch_rows)
                batch_rows.clear()

                print(f"   Đã ghi {total_rows:,} dòng...")

        # Ghi phần dữ liệu còn lại.
        if batch_rows:
            table = pa.Table.from_pylist(batch_rows)

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_file),
                    table.schema,
                    compression="snappy"
                )

            if not table.schema.equals(
                writer.schema,
                check_metadata=False
            ):
                raise ValueError(
                    "Schema giữa các lô dữ liệu không đồng nhất."
                )

            writer.write_table(table)
            total_rows += len(batch_rows)
            batch_rows.clear()

    finally:
        if writer is not None:
            writer.close()

    # Nếu không có dữ liệu, không tạo file Parquet rỗng
    # không có schema.
    if not output_file.exists():
        raise RuntimeError(
            "gold_features không có dòng dữ liệu nào. "
            "Hãy kiểm tra dữ liệu đầu vào và các phép join."
        )

    print(f"-> Đã ghi tổng cộng {total_rows:,} dòng.")
    print(f"-> File Parquet: {output_file}")

    return output_file, total_rows


# ============================================================
# 4. KIỂM TRA DATA LINEAGE BẰNG PYARROW
# ============================================================

def validate_gold_features(parquet_file, expected_rows):
    """
    Kiểm tra file Parquet đã ghi:
    - Tổng số dòng.
    - Các cột đặc trưng quan trọng.
    - Phân bố label = 0/1.
    """

    print("\n=== KIỂM TRA DATA LINEAGE GOLD_FEATURES ===")

    parquet_file = Path(parquet_file)

    parquet_metadata = pq.read_metadata(parquet_file)
    total_rows = parquet_metadata.num_rows

    print(f"-> Tổng số cặp ứng viên: {total_rows:,}")

    if total_rows != expected_rows:
        raise RuntimeError(
            f"Số dòng không khớp: dự kiến {expected_rows:,}, "
            f"đọc lại được {total_rows:,}."
        )

    # Kiểm tra các cột bắt buộc.
    schema = pq.read_schema(parquet_file)
    columns = set(schema.names)

    required_columns = {
        "user_id",
        "product_id",
        "label",
        "u_total_orders",
        "p_total_purchases",
        "up_buy_count"
    }

    missing_columns = required_columns - columns

    if missing_columns:
        raise RuntimeError(
            "Thiếu các cột bắt buộc: "
            + ", ".join(sorted(missing_columns))
        )

    print("-> Kiểm tra schema: ĐẠT")

    # Chỉ đọc cột label, không tải toàn bộ bảng vào RAM.
    label_table = pq.read_table(
        str(parquet_file),
        columns=["label"]
    )

    label_counts_table = (
        label_table
        .group_by("label")
        .aggregate([("label", "count")])
    )

    label_counts = label_counts_table.to_pylist()

    print("-> Phân bố nhãn:")

    for item in label_counts:
        label = item["label"]
        count = item["label_count"]
        pct = count / total_rows * 100 if total_rows else 0

        print(
            f"   + label = {label}: "
            f"{count:,} dòng ({pct:.2f}%)"
        )

    print("-> Kiểm tra số dòng: ĐẠT")
    print("-> Kiểm tra Data Lineage: HOÀN TẤT")


# ============================================================
# 5. LUỒNG XỬ LÝ CHÍNH
# ============================================================

def main():
    spark = None
    prior_df = None

    try:
        # ----------------------------------------------------
        # BƯỚC 1: KIỂM TRA ĐẦU VÀO
        # ----------------------------------------------------

        print("=" * 65)
        print("STEP 1 - XÂY DỰNG GOLD_FEATURES")
        print("=" * 65)

        for file_path in [ORDERS_CSV, ORDER_PRODUCTS_CSV]:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy file đầu vào: {file_path}"
                )

        print(f"-> Orders: {ORDERS_CSV}")
        print(f"-> Order products: {ORDER_PRODUCTS_CSV}")

        # ----------------------------------------------------
        # BƯỚC 2: ĐỌC SILVER CSV BẰNG SPARK
        # ----------------------------------------------------

        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        print("\nĐang đọc dữ liệu Silver CSV...")

        orders = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(ORDERS_CSV))
        )

        order_products = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(ORDER_PRODUCTS_CSV))
        )

        print("-> Đã đọc xong hai bảng Silver.")

        # ----------------------------------------------------
        # BƯỚC 3: TÁCH PRIOR VÀ TRAIN
        # ----------------------------------------------------

        print("\nĐang tách dữ liệu prior và train...")

        prior_orders = orders.filter(
            F.col("eval_set") == "prior"
        )

        train_orders = orders.filter(
            F.col("eval_set") == "train"
        )

        prior_df = (
            prior_orders
            .join(
                order_products,
                on="order_id",
                how="inner"
            )
            .cache()
        )

        train_df = train_orders.join(
            order_products,
            on="order_id",
            how="inner"
        )

        # ----------------------------------------------------
        # NHÓM 1: ĐẶC TRƯNG USER
        # ----------------------------------------------------

        print("\n1. Đang tính đặc trưng User...")

        user_order_stats = (
            prior_orders
            .groupBy("user_id")
            .agg(
                F.max("order_number").alias("u_total_orders"),
                F.avg("days_since_prior_order").alias(
                    "u_avg_days_between_orders"
                )
            )
        )

        user_item_stats = (
            prior_df
            .groupBy("user_id")
            .agg(
                F.count("product_id").alias("u_total_items"),
                F.avg("reordered").alias("u_reorder_ratio")
            )
        )

        user_features = (
            user_order_stats
            .join(
                user_item_stats,
                on="user_id",
                how="inner"
            )
            .withColumn(
                "u_avg_basket_size",
                F.col("u_total_items")
                / F.when(
                    F.col("u_total_orders") != 0,
                    F.col("u_total_orders")
                ).otherwise(F.lit(None))
            )
            .fillna({
                "u_avg_days_between_orders": 0.0,
                "u_reorder_ratio": 0.0,
                "u_avg_basket_size": 0.0
            })
        )

        # ----------------------------------------------------
        # NHÓM 2: ĐẶC TRƯNG PRODUCT
        # ----------------------------------------------------

        print("2. Đang tính đặc trưng Product...")

        product_features = (
            prior_df
            .groupBy("product_id")
            .agg(
                F.count("*").alias("p_total_purchases"),
                F.avg("reordered").alias("p_reorder_ratio"),
                F.countDistinct("user_id").alias("p_unique_users"),
                F.avg("add_to_cart_order").alias(
                    "p_avg_add_to_cart"
                )
            )
            .fillna({
                "p_reorder_ratio": 0.0
            })
        )

        # ----------------------------------------------------
        # NHÓM 3: ĐẶC TRƯNG USER - PRODUCT
        # ----------------------------------------------------

        print("3. Đang tính đặc trưng User-Product...")

        w_user_order = (
            Window
            .partitionBy("user_id")
            .orderBy("order_number")
            .rowsBetween(
                Window.unboundedPreceding,
                Window.currentRow
            )
        )

        orders_with_cum_days = (
            prior_orders
            .withColumn(
                "days_clean",
                F.coalesce(
                    F.col("days_since_prior_order"),
                    F.lit(0.0)
                )
            )
            .withColumn(
                "cum_days",
                F.sum("days_clean").over(w_user_order)
            )
        )

        user_max_state = (
            orders_with_cum_days
            .groupBy("user_id")
            .agg(
                F.max("order_number").alias("u_max_order_num"),
                F.max("cum_days").alias("u_max_cum_days")
            )
        )

        prior_with_days = prior_df.join(
            orders_with_cum_days.select(
                "order_id",
                "cum_days"
            ),
            on="order_id",
            how="inner"
        )

        up_basic = (
            prior_with_days
            .groupBy("user_id", "product_id")
            .agg(
                F.count("*").alias("up_buy_count"),
                F.min("order_number").alias(
                    "up_first_order_num"
                ),
                F.max("order_number").alias(
                    "up_last_order_num"
                ),
                F.max("cum_days").alias(
                    "up_last_buy_cum_days"
                ),
                F.avg("add_to_cart_order").alias(
                    "up_avg_add_to_cart"
                )
            )
        )

        # Đếm chuỗi mua liên tiếp tính từ đơn prior cuối cùng.
        w_up_desc = (
            Window
            .partitionBy("user_id", "product_id")
            .orderBy(F.col("order_number").desc())
        )

        up_streak_df = (
            prior_df
            .withColumn("rn", F.row_number().over(w_up_desc))
            .withColumn(
                "diff",
                F.col("order_number") + F.col("rn")
            )
        )

        max_order_per_up = (
            prior_df
            .groupBy("user_id", "product_id")
            .agg(
                F.max("order_number").alias("max_o")
            )
        )

        up_streak = (
            up_streak_df
            .join(
                max_order_per_up,
                on=["user_id", "product_id"],
                how="inner"
            )
            .filter(
                F.col("diff") == F.col("max_o") + 1
            )
            .groupBy("user_id", "product_id")
            .agg(
                F.count("*").alias("up_streak")
            )
        )

        up_features = (
            up_basic
            .join(
                user_max_state,
                on="user_id",
                how="inner"
            )
            .join(
                up_streak,
                on=["user_id", "product_id"],
                how="left"
            )
            .withColumn(
                "up_orders_since_last_buy",
                F.col("u_max_order_num")
                - F.col("up_last_order_num")
            )
            .withColumn(
                "up_days_since_last_buy",
                F.col("u_max_cum_days")
                - F.col("up_last_buy_cum_days")
            )
            .withColumn(
                "up_buy_rate_since_first",
                F.col("up_buy_count")
                / (
                    F.col("u_max_order_num")
                    - F.col("up_first_order_num")
                    + F.lit(1.0)
                )
            )
            .drop(
                "u_max_order_num",
                "u_max_cum_days",
                "up_last_buy_cum_days"
            )
            .fillna({
                "up_streak": 1
            })
        )

        # ----------------------------------------------------
        # BƯỚC 4: GÁN NHÃN TỪ TẬP TRAIN
        # ----------------------------------------------------

        print("4. Đang gán nhãn label từ tập train...")

        train_users = (
            train_orders
            .select("user_id")
            .distinct()
        )

        train_labels = (
            train_df
            .filter(F.col("reordered") == 1)
            .select("user_id", "product_id")
            .distinct()
            .withColumn(
                "label",
                F.lit(1)
            )
        )

        gold_features = (
            up_features
            .join(
                train_users,
                on="user_id",
                how="inner"
            )
            .join(
                user_features,
                on="user_id",
                how="inner"
            )
            .join(
                F.broadcast(product_features),
                on="product_id",
                how="inner"
            )
            .join(
                train_labels,
                on=["user_id", "product_id"],
                how="left"
            )
            .fillna({
                "label": 0
            })
        )

        # ----------------------------------------------------
        # BƯỚC 5: GHI PARQUET BẰNG PYARROW
        # ----------------------------------------------------

        print("\n5. Đang ghi gold_features bằng PyArrow...")

        output_file, total_rows = save_parquet_with_pyarrow(
            gold_features,
            OUTPUT_DIR,
            batch_size=BATCH_SIZE
        )

        # ----------------------------------------------------
        # BƯỚC 6: KIỂM TRA DATA LINEAGE
        # ----------------------------------------------------

        validate_gold_features(
            output_file,
            expected_rows=total_rows
        )

        print("\n" + "=" * 65)
        print("HOÀN THÀNH STEP 1!")
        print(f"Đầu ra: {output_file}")
        print("=" * 65)

    finally:
        if prior_df is not None:
            prior_df.unpersist()

        if spark is not None:
            spark.stop()


# ============================================================
# 6. ĐIỂM BẮT ĐẦU CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()
