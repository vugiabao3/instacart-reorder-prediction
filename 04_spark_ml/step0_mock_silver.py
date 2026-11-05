# -*- coding: utf-8 -*-
"""
step0_mock_silver.py
Nhiệm vụ: Đọc dữ liệu sạch tầng Silver từ Thành viên 2 (02_hive_cleaning/silver_*_parquet)
bổ sung tập train nếu thiếu và đồng bộ sang storage/silver của TV4 dưới dạng Parquet local.
"""

import os
import platform
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField

# Hỗ trợ tự động winutils.exe nếu chạy trên Windows (theo yêu cầu của Lead)
if platform.system() == "Windows" and os.path.exists("C:\\hadoop"):
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = os.environ.get("PATH", "") + ";C:\\hadoop\\bin"


def create_spark_session():
    return (
        SparkSession.builder.appName("TV4_Step0_Sync_TV2_Silver_Parquet")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    base_dir = Path(__file__).resolve().parent.parent
    tv2_dir = base_dir / "02_hive_cleaning"
    sample_dir = base_dir / "data_sample"
    silver_out_dir = base_dir / "04_spark_ml" / "storage" / "silver"
    silver_out_dir.mkdir(parents=True, exist_ok=True)

    # Đường dẫn Parquet mới của TV2 trên nhánh main
    silver_orders_pq = tv2_dir / "silver_orders_parquet"
    silver_op_pq = tv2_dir / "silver_order_products_parquet"

    # Đường dẫn CSV dự phòng
    silver_orders_csv = tv2_dir / "silver_orders.csv"
    silver_op_csv = tv2_dir / "silver_order_products.csv"
    raw_train_op_path = sample_dir / "order_products__train.csv"

    print("=== [STEP 0] ĐỒNG BỘ DỮ LIỆU PARQUET TẦNG SILVER TỪ THÀNH VIÊN 2 ===")

    orders_schema = StructType([
        StructField("order_id", IntegerType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("eval_set", StringType(), True),
        StructField("order_number", IntegerType(), True),
        StructField("order_dow", IntegerType(), True),
        StructField("order_hour_of_day", IntegerType(), True),
        StructField("days_since_prior_order", DoubleType(), True),
        StructField("is_first_order", IntegerType(), True),
    ])

    op_schema = StructType([
        StructField("order_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("add_to_cart_order", IntegerType(), True),
        StructField("reordered", IntegerType(), True),
    ])

    # 1. Đọc bảng silver_orders từ Parquet của TV2
    if silver_orders_pq.exists():
        print(f"-> Đang đọc Parquet từ TV2: {silver_orders_pq}")
        silver_orders = spark.read.parquet(silver_orders_pq.as_uri())
        if "is_first_order" not in silver_orders.columns:
            silver_orders = silver_orders.withColumn(
                "is_first_order", F.when(F.col("days_since_prior_order").isNull(), 1).otherwise(0)
            )
        silver_orders = silver_orders.select(
            F.col("order_id").cast(IntegerType()),
            F.col("user_id").cast(IntegerType()),
            F.col("eval_set").cast(StringType()),
            F.col("order_number").cast(IntegerType()),
            F.col("order_dow").cast(IntegerType()),
            F.col("order_hour_of_day").cast(IntegerType()),
            F.col("days_since_prior_order").cast(DoubleType()),
            F.col("is_first_order").cast(IntegerType()),
        )
    else:
        print(f"-> Đang đọc CSV dự phòng từ TV2: {silver_orders_csv}")
        silver_orders = spark.read.csv(silver_orders_csv.as_uri(), header=True, schema=orders_schema)

    # 2. Đọc bảng silver_order_products từ Parquet của TV2
    if silver_op_pq.exists():
        print(f"-> Đang đọc Parquet từ TV2: {silver_op_pq}")
        silver_op = spark.read.parquet(silver_op_pq.as_uri()).select(
            F.col("order_id").cast(IntegerType()),
            F.col("product_id").cast(IntegerType()),
            F.col("add_to_cart_order").cast(IntegerType()),
            F.col("reordered").cast(IntegerType()),
        )
    else:
        print(f"-> Đang đọc CSV dự phòng từ TV2: {silver_op_csv}")
        silver_op = spark.read.csv(silver_op_csv.as_uri(), header=True, schema=op_schema)

    # 3. Kiểm tra và bổ sung tập train nếu thiếu
    train_orders_ids = silver_orders.filter(F.col("eval_set") == "train").select("order_id")
    train_in_op_count = silver_op.join(train_orders_ids, on="order_id", how="inner").limit(1).count()

    if train_in_op_count == 0 and raw_train_op_path.exists():
        print("-> Đang ghép thêm đơn 'train' từ data_sample theo chuẩn của TV2...")
        train_op_raw = spark.read.csv(raw_train_op_path.as_uri(), header=True, schema=op_schema)
        train_op_clean = (
            train_op_raw.dropna(subset=["order_id", "product_id"])
            .dropDuplicates()
            .join(train_orders_ids, on="order_id", how="inner")
            .select("order_id", "product_id", "add_to_cart_order", "reordered")
        )
        silver_op = silver_op.unionByName(train_op_clean)

    # 4. Lưu trực tiếp dưới dạng Parquet local cho Step 1
    out_orders_uri = (silver_out_dir / "silver_orders").as_uri()
    out_op_uri = (silver_out_dir / "silver_order_products").as_uri()

    silver_orders.write.mode("overwrite").parquet(out_orders_uri)
    silver_op.write.mode("overwrite").parquet(out_op_uri)

    print("=== HOÀN TẤT STEP 0 ===")
    print(f"Tổng số dòng silver_orders        : {silver_orders.count():,}")
    print(f"Tổng số dòng silver_order_products: {silver_op.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()
