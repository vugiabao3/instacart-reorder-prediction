# -*- coding: utf-8 -*-
"""
step1_build_features.py
Nhiệm vụ của Thành viên 4:
- Đọc dữ liệu tầng Silver dưới dạng file Parquet (đã đồng bộ từ step0).
- Tính 3 nhóm đặc trưng: User (5 biến), Product (4 biến), User-Product (8 biến).
- Gán nhãn (label = 1/0) từ đơn hàng cuối (eval_set == 'train'), tránh Data Leakage.
- Xuất bảng gold_features (Parquet) và kiểm tra Data Lineage.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_spark_session():
    return (
        SparkSession.builder.appName("TV4_Step1_Build_Features")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    silver_storage = os.path.join(base_dir, "04_spark_ml", "storage", "silver")
    gold_out_dir = os.path.join(base_dir, "04_spark_ml", "storage", "gold", "gold_features")

    orders_parquet = os.path.join(silver_storage, "silver_orders")
    op_parquet = os.path.join(silver_storage, "silver_order_products")

    print("=== [STEP 1] XÂY DỰNG 3 NHÓM ĐẶC TRƯNG (GOLD_FEATURES) ===")
    print(f"-> Đọc silver_orders từ Parquet: {orders_parquet}")
    print(f"-> Đọc silver_order_products từ Parquet: {op_parquet}")

    # Đọc trực tiếp file Parquet được sinh ra từ step0_mock_silver.py
    orders = spark.read.parquet(f"file://{orders_parquet}")
    order_products = spark.read.parquet(f"file://{op_parquet}")

    # 1. Tách dữ liệu theo thời gian: prior dùng tính đặc trưng, train dùng gán nhãn
    prior_orders = orders.filter(F.col("eval_set") == "prior")
    train_orders = orders.filter(F.col("eval_set") == "train")

    prior_df = prior_orders.join(order_products, on="order_id", how="inner").cache()
    train_df = train_orders.join(order_products, on="order_id", how="inner")

    # ---------------------------------------------------------
    # NHÓM 1: ĐẶC TRƯNG USER (u_*)
    # ---------------------------------------------------------
    print("1. Đang tính nhóm đặc trưng User...")
    user_order_stats = prior_orders.groupBy("user_id").agg(
        F.max("order_number").alias("u_total_orders"),
        F.avg("days_since_prior_order").alias("u_avg_days_between_orders"),
    )

    user_item_stats = prior_df.groupBy("user_id").agg(
        F.count("product_id").alias("u_total_items"),
        F.avg("reordered").alias("u_reorder_ratio"),
    )

    user_features = (
        user_order_stats.join(user_item_stats, on="user_id", how="inner")
        .withColumn("u_avg_basket_size", F.col("u_total_items") / F.col("u_total_orders"))
        .fillna({"u_avg_days_between_orders": 0.0, "u_reorder_ratio": 0.0})
    )

    # ---------------------------------------------------------
    # NHÓM 2: ĐẶC TRƯNG PRODUCT (p_*)
    # ---------------------------------------------------------
    print("2. Đang tính nhóm đặc trưng Product...")
    product_features = (
        prior_df.groupBy("product_id")
        .agg(
            F.count("*").alias("p_total_purchases"),
            F.avg("reordered").alias("p_reorder_ratio"),
            F.countDistinct("user_id").alias("p_unique_users"),
            F.avg("add_to_cart_order").alias("p_avg_add_to_cart"),
        )
        .fillna({"p_reorder_ratio": 0.0})
    )

    # ---------------------------------------------------------
    # NHÓM 3: ĐẶC TRƯNG TƯƠNG TÁC USER - PRODUCT (up_*)
    # ---------------------------------------------------------
    print("3. Đang tính nhóm đặc trưng User-Product (chuỗi mua & ngày tích lũy)...")
    w_user_order = (
        Window.partitionBy("user_id")
        .orderBy("order_number")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    orders_with_cum_days = prior_orders.withColumn(
        "days_clean", F.coalesce(F.col("days_since_prior_order"), F.lit(0.0))
    ).withColumn("cum_days", F.sum("days_clean").over(w_user_order))

    user_max_state = orders_with_cum_days.groupBy("user_id").agg(
        F.max("order_number").alias("u_max_order_num"),
        F.max("cum_days").alias("u_max_cum_days"),
    )

    prior_with_days = prior_df.join(
        orders_with_cum_days.select("order_id", "cum_days"), on="order_id", how="inner"
    )

    up_basic = prior_with_days.groupBy("user_id", "product_id").agg(
        F.count("*").alias("up_buy_count"),
        F.min("order_number").alias("up_first_order_num"),
        F.max("order_number").alias("up_last_order_num"),
        F.max("cum_days").alias("up_last_buy_cum_days"),
        F.avg("add_to_cart_order").alias("up_avg_add_to_cart"),
    )

    w_up_desc = Window.partitionBy("user_id", "product_id").orderBy(F.col("order_number").desc())
    up_streak_df = prior_df.withColumn("rn", F.row_number().over(w_up_desc)).withColumn(
        "diff", F.col("order_number") + F.col("rn")
    )

    max_order_per_up = prior_df.groupBy("user_id", "product_id").agg(
        F.max("order_number").alias("max_o")
    )
    up_streak = (
        up_streak_df.join(max_order_per_up, on=["user_id", "product_id"], how="inner")
        .filter(F.col("diff") == (F.col("max_o") + 1))
        .groupBy("user_id", "product_id")
        .agg(F.count("*").alias("up_streak"))
    )

    up_features = (
        up_basic.join(user_max_state, on="user_id", how="inner")
        .join(up_streak, on=["user_id", "product_id"], how="left")
        .withColumn("up_orders_since_last_buy", F.col("u_max_order_num") - F.col("up_last_order_num"))
        .withColumn("up_days_since_last_buy", F.col("u_max_cum_days") - F.col("up_last_buy_cum_days"))
        .withColumn(
            "up_buy_rate_since_first",
            F.col("up_buy_count") / (F.col("u_max_order_num") - F.col("up_first_order_num") + 1.0),
        )
        .drop("u_max_order_num", "u_max_cum_days", "up_last_buy_cum_days")
        .fillna({"up_streak": 1})
    )

    # ---------------------------------------------------------
    # GÁN NHÃN VÀ LƯU BẢNG GOLD_FEATURES
    # ---------------------------------------------------------
    print("4. Đang gán nhãn từ tập train và lưu bảng gold_features...")
    train_users = train_orders.select("user_id").distinct()

    train_labels = (
        train_df.filter(F.col("reordered") == 1)
        .select("user_id", "product_id")
        .distinct()
        .withColumn("label", F.lit(1))
    )

    gold_features = (
        up_features.join(train_users, on="user_id", how="inner")
        .join(user_features, on="user_id", how="inner")
        .join(F.broadcast(product_features), on="product_id", how="inner")
        .join(train_labels, on=["user_id", "product_id"], how="left")
        .fillna({"label": 0})
    )

    out_path = f"file://{gold_out_dir}"
    gold_features.write.mode("overwrite").parquet(out_path)

    # Kiểm tra Data Lineage
    df_check = spark.read.parquet(out_path)
    total_rows = df_check.count()
    label_counts = df_check.groupBy("label").count().collect()

    print("=== HOÀN TẤT STEP 1: KIỂM TRA DATA LINEAGE GOLD_FEATURES ===")
    print(f"-> Tổng số cặp ứng viên: {total_rows:,}")
    for row in label_counts:
        pct = (row["count"] / total_rows) * 100 if total_rows > 0 else 0
        print(f"   + Nhãn label = {row['label']}: {row['count']:,} dòng ({pct:.2f}%)")

    prior_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()


