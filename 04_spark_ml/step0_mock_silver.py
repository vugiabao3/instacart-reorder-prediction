# -*- coding: utf-8 -*-
"""
step0_mock_silver.py
Nhiệm vụ: Đọc dữ liệu sạch tầng Silver từ Thành viên 2 (02_hive_cleaning/silver_*.csv)
và đồng bộ sang định dạng Parquet (kèm bổ sung chi tiết đơn train để gán nhãn cho TV4).
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, StringType, StructType, StructField


def create_spark_session():
    return (
        SparkSession.builder.appName("TV4_Step0_Sync_TV2_Silver")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Xác định đường dẫn gốc của dự án
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tv2_dir = os.path.join(base_dir, "02_hive_cleaning")
    sample_dir = os.path.join(base_dir, "data_sample")
    silver_out_dir = os.path.join(base_dir, "04_spark_ml", "storage", "silver")

    silver_orders_path = os.path.join(tv2_dir, "silver_orders.csv")
    silver_op_path = os.path.join(tv2_dir, "silver_order_products.csv")
    raw_train_op_path = os.path.join(sample_dir, "order_products__train.csv")

    print("=== [STEP 0] ĐỒNG BỘ DỮ LIỆU TẦNG SILVER TỪ THÀNH VIÊN 2 ===")
    print(f"-> Đọc silver_orders từ: {silver_orders_path}")
    print(f"-> Đọc silver_order_products từ: {silver_op_path}")

    # 1. Đọc silver_orders.csv của Thành viên 2
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

    silver_orders = spark.read.csv(
        f"file://{silver_orders_path}", header=True, schema=orders_schema
    )

    # 2. Đọc silver_order_products.csv của Thành viên 2
    op_schema = StructType([
        StructField("order_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("add_to_cart_order", IntegerType(), True),
        StructField("reordered", IntegerType(), True),
    ])

    silver_op = spark.read.csv(
        f"file://{silver_op_path}", header=True, schema=op_schema
    )

    # 3. Kiểm tra xem silver_order_products của TV2 đã có các đơn 'train' chưa
    train_orders_ids = silver_orders.filter(F.col("eval_set") == "train").select("order_id")
    train_in_op_count = silver_op.join(train_orders_ids, on="order_id", how="inner").limit(1).count()

    if train_in_op_count == 0 and os.path.exists(raw_train_op_path):
        print("-> Phát hiện silver_order_products của TV2 chỉ chứa đơn 'prior'.")
        print(f"-> Đang làm sạch và ghép thêm đơn 'train' từ {raw_train_op_path} theo chuẩn của TV2...")
        train_op_raw = spark.read.csv(
            f"file://{raw_train_op_path}", header=True, schema=op_schema
        )
        # Áp dụng đúng quy tắc làm sạch của TV2: bỏ null, bỏ trùng, chỉ giữ order_id có trong silver_orders
        train_op_clean = (
            train_op_raw.dropna(subset=["order_id", "product_id"])
            .dropDuplicates()
            .join(train_orders_ids, on="order_id", how="inner")
            .select("order_id", "product_id", "add_to_cart_order", "reordered")
        )
        silver_op = silver_op.unionByName(train_op_clean)

    # 4. Lưu xuống định dạng Parquet chuẩn để Step 1 chạy tốc độ cao
    out_orders_parquet = f"file://{os.path.join(silver_out_dir, 'silver_orders')}"
    out_op_parquet = f"file://{os.path.join(silver_out_dir, 'silver_order_products')}"

    silver_orders.write.mode("overwrite").parquet(out_orders_parquet)
    silver_op.write.mode("overwrite").parquet(out_op_parquet)

    print("=== HOÀN TẤT STEP 0 (ĐỐI CHIẾU SỐ DÒNG SILVER) ===")
    print(f"Tổng số dòng silver_orders        : {silver_orders.count():,}")
    print(f"Tổng số dòng silver_order_products: {silver_op.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()