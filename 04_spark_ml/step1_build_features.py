# -*- coding: utf-8 -*-
"""
step1_build_features.py
Nhiệm vụ của Thành viên 4:
- Đọc dữ liệu tầng Silver dưới dạng file Parquet (từ Step 0 / Parquet của TV2).
- Tính 3 nhóm đặc trưng: User (5 biến), Product (4 biến), User-Product (8 biến).
- Gán nhãn (label = 1/0) từ đơn hàng cuối (eval_set == 'train'), tránh Data Leakage.
- Xuất bảng gold_features (Parquet local + CSV trong 04_spark_ml) và kiểm tra Data Lineage.
"""

import os
import platform
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

if platform.system() == "Windows" and os.path.exists("C:\\hadoop"):
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = os.environ.get("PATH", "") + ";C:\\hadoop\\bin"


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

    base_dir = Path(__file__).resolve().parent.parent
    silver_storage = base_dir / "04_spark_ml" / "storage" / "silver"
    tv2_dir = base_dir / "02_hive_cleaning"
    gold_out_dir = base_dir / "04_spark_ml" / "storage" / "gold" / "gold_features"
    github_features_csv = base_dir / "04_spark_ml" / "gold_features.csv"
    gold_out_dir.parent.mkdir(parents=True, exist_ok=True)

    orders_parquet = silver_storage / "silver_orders"
    op_parquet = silver_storage / "silver_order_products"

    print("=== [STEP 1] XÂY DỰNG 3 NHÓM ĐẶC TRƯNG (GOLD_FEATURES) ===")

    if orders_parquet.exists() and op_parquet.exists():
        print(f"-> Đọc silver_orders từ Parquet: {orders_parquet}")
        print(f"-> Đọc silver_order_products từ Parquet: {op_parquet}")
        orders = spark.read.parquet(orders_parquet.as_uri())
        order_products = spark.read.parquet(op_parquet.as_uri())
    else:
        orders_tv2_pq = tv2_dir / "silver_orders_parquet"
        op_tv2_pq = tv2_dir / "silver_order_products_parquet"
        print(f"-> Đọc trực tiếp Parquet từ TV2: {orders_tv2_pq} & {op_tv2_pq}")
        orders = spark.read.parquet(orders_tv2_pq.as_uri())
        order_products = spark.read.parquet(op_tv2_pq.as_uri())

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

    gold_features.write.mode("overwrite").parquet(gold_out_dir.as_uri())

    # Kiểm tra Data Lineage và xuất CSV vào 04_spark_ml/gold_features.csv
    df_check = spark.read.parquet(gold_out_dir.as_uri())
    total_rows = df_check.count()
    distinct_pairs = df_check.select("user_id", "product_id").distinct().count()
    label_counts = df_check.groupBy("label").count().collect()

    pdf_gf = df_check.orderBy("user_id", "product_id").limit(2000).toPandas()
    pdf_gf.to_csv(github_features_csv, index=False, encoding="utf-8")

    print("=== HOÀN TẤT STEP 1: KIỂM TRA DATA LINEAGE GOLD_FEATURES ===")
    print(f"-> Đã lưu Parquet tại: {gold_out_dir}")
    print(f"-> Đã xuất CSV tại   : {github_features_csv} ({len(pdf_gf):,} dòng)")
    print(f"-> Tổng số dòng: {total_rows:,} | Số cặp (user_id, product_id) duy nhất: {distinct_pairs:,}")
    for row in label_counts:
        pct = (row["count"] / total_rows) * 100 if total_rows > 0 else 0
        print(f"   + Nhãn label = {row['label']}: {row['count']:,} dòng ({pct:.2f}%)")

    prior_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
