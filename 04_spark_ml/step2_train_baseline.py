# -*- coding: utf-8 -*-
"""
step2_train_baseline.py
Nhiệm vụ của Thành viên 4:
- Đọc bảng gold_features (từ step1_build_features.py).
- Chia tập Train/Test theo user_id (80/20, seed=42) để tránh rò rỉ dữ liệu.
- Huấn luyện mô hình Baseline Logistic Regression (Spark MLlib).
- Tối ưu ngưỡng quyết định (Threshold tuning) và đánh giá: AUC-ROC, PR-AUC, F1, Precision@K, Recall@K.
- Xuất bảng gold_reorder_predictions:
  + Bản Parquet chuẩn hợp đồng dữ liệu.
  + Bản CSV (8 cột trực quan, 2.000 dòng) để hiển thị bảng Preview trên GitHub và cho Thành viên 5 làm Dashboard.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def create_spark_session():
    return (
        SparkSession.builder.appName("TV4_Step2_Train_Baseline_LR")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def evaluate_precision_recall_at_k(pred_df, k=10):
    """Tính Precision@K và Recall@K trung bình trên từng khách hàng."""
    w_user = Window.partitionBy("user_id").orderBy(F.col("probability").desc())
    ranked = pred_df.withColumn("rank", F.row_number().over(w_user))

    top_k = ranked.filter(F.col("rank") <= k).groupBy("user_id").agg(
        F.sum("label").alias("hits_at_k"),
        F.count("*").alias("k_items"),
    )

    actual_positives = (
        pred_df.groupBy("user_id")
        .agg(F.sum("label").alias("total_actual"))
        .filter(F.col("total_actual") > 0)
    )

    metrics_df = top_k.join(actual_positives, on="user_id", how="inner").select(
        (F.col("hits_at_k") / F.lit(float(k))).alias("precision_at_k"),
        (F.col("hits_at_k") / F.col("total_actual")).alias("recall_at_k"),
    )

    res = metrics_df.agg(
        F.avg("precision_at_k").alias("avg_p_at_k"),
        F.avg("recall_at_k").alias("avg_r_at_k"),
    ).collect()[0]

    return res["avg_p_at_k"] or 0.0, res["avg_r_at_k"] or 0.0


def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Xác định các đường dẫn trong dự án
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    features_dir = os.path.join(base_dir, "04_spark_ml", "storage", "gold", "gold_features")
    pred_parquet_dir = os.path.join(base_dir, "04_spark_ml", "storage", "gold", "gold_reorder_predictions")
    github_csv_out = os.path.join(base_dir, "04_spark_ml", "gold_reorder_predictions.csv")
    sample_csv_out = os.path.join(base_dir, "data_sample", "gold_reorder_predictions.csv")
    products_csv_path = os.path.join(base_dir, "data_sample", "products.csv")

    print("=== [STEP 2] HUẤN LUYỆN BASELINE LOGISTIC REGRESSION ===")
    df_ml = spark.read.parquet(f"file://{features_dir}").cache()

    feature_cols = [
        # Nhóm User (5 đặc trưng)
        "u_total_orders",
        "u_avg_days_between_orders",
        "u_total_items",
        "u_reorder_ratio",
        "u_avg_basket_size",
        # Nhóm Product (4 đặc trưng)
        "p_total_purchases",
        "p_reorder_ratio",
        "p_unique_users",
        "p_avg_add_to_cart",
        # Nhóm User-Product (8 đặc trưng)
        "up_buy_count",
        "up_first_order_num",
        "up_last_order_num",
        "up_avg_add_to_cart",
        "up_streak",
        "up_orders_since_last_buy",
        "up_days_since_last_buy",
        "up_buy_rate_since_first",
    ]

    # 1. Chia tập Train / Test theo user_id (80% - 20%, cố định seed=42)
    users_split = df_ml.select("user_id").distinct().randomSplit([0.8, 0.2], seed=42)
    train_users, test_users = users_split[0], users_split[1]

    train_data = df_ml.join(train_users, on="user_id", how="inner")
    test_data = df_ml.join(test_users, on="user_id", how="inner")

    # 2. Vector hóa và Chuẩn hóa đặc trưng
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="keep")
    train_assembled = assembler.transform(train_data)
    test_assembled = assembler.transform(test_data)

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    scaler_model = scaler.fit(train_assembled)

    train_vec = scaler_model.transform(train_assembled).cache()
    test_vec = scaler_model.transform(test_assembled).cache()

    # 3. Huấn luyện mô hình Logistic Regression (kèm trọng số lớp để xử lý mất cân bằng nhãn)
    pos_count = train_vec.filter(F.col("label") == 1).count()
    total_train = train_vec.count()
    balancing_ratio = (total_train - pos_count) / float(total_train) if pos_count > 0 else 1.0

    train_weighted = train_vec.withColumn(
        "classWeightCol",
        F.when(F.col("label") == 1, balancing_ratio).otherwise(1.0 - balancing_ratio),
    )

    print("-> Đang huấn luyện mô hình Logistic Regression trên tập Train...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeightCol",
        maxIter=30,
        regParam=0.01,
    )
    lr_model = lr.fit(train_weighted)

    # 4. Dự đoán trên tập Test và trích xuất xác suất mua lại (lớp 1)
    raw_preds = lr_model.transform(test_vec).withColumn(
        "probability_val", vector_to_array("probability")[1]
    ).cache()

    eval_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    eval_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="pred_custom", metricName="f1"
    )

    auc_roc = eval_roc.evaluate(raw_preds)
    pr_auc = eval_pr.evaluate(raw_preds)

    # 5. Quét tìm ngưỡng quyết định (Threshold) tối ưu cho F1-Score
    best_th = 0.50
    best_f1 = 0.0
    for th in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        temp_pred = raw_preds.withColumn(
            "pred_custom",
            F.when(F.col("probability_val") >= th, F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        f1_val = eval_f1.evaluate(temp_pred)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_th = th

    # Áp dụng ngưỡng tối ưu để tính Precision@10 và Recall@10
    final_test_preds = raw_preds.select(
        F.col("user_id"),
        F.col("product_id"),
        F.round(F.col("probability_val"), 4).alias("probability"),
        F.when(F.col("probability_val") >= best_th, F.lit(1)).otherwise(F.lit(0)).alias("predicted"),
        F.col("label").cast("double").alias("label"),
    )

    p_at_10, r_at_10 = evaluate_precision_recall_at_k(final_test_preds, k=10)

    print("\n============================================================")
    print("   KẾT QUẢ ĐÁNH GIÁ BASELINE LOGISTIC REGRESSION (TUẦN 1)   ")
    print("============================================================")
    print(f" + AUC-ROC               : {auc_roc:.4f}")
    print(f" + PR-AUC                : {pr_auc:.4f}")
    print(f" + Ngưỡng tối ưu (Best T): {best_th:.2f}")
    print(f" + F1-Score (tại Best T) : {best_f1:.4f}")
    print(f" + Precision@10          : {p_at_10:.4f}")
    print(f" + Recall@10             : {r_at_10:.4f}")
    print("============================================================\n")

    # 6. Lưu bản Parquet chuẩn theo Hợp đồng dữ liệu (Data Contract)
    gold_reorder_predictions = final_test_preds.select(
        "user_id", "product_id", "probability", "predicted"
    )
    gold_reorder_predictions.write.mode("overwrite").parquet(f"file://{pred_parquet_dir}")
    print(f"-> Đã lưu bản Parquet chuẩn tại: {pred_parquet_dir}")

    # 7. Tạo bảng hiển thị trực quan (8 cột) để xuất ra CSV cho GitHub & Thành viên 5
    if os.path.exists(products_csv_path):
        products_df = (
            spark.read.csv(f"file://{products_csv_path}", header=True, inferSchema=True)
            .select("product_id", "product_name")
        )
        preds_with_name = raw_preds.join(F.broadcast(products_df), on="product_id", how="left")
    else:
        preds_with_name = raw_preds.withColumn(
            "product_name", F.concat(F.lit("Product #"), F.col("product_id"))
        )

    w_user_rank = Window.partitionBy("user_id").orderBy(F.col("probability_val").desc())

    gold_reorder_display = (
        preds_with_name
        .withColumn("probability", F.round(F.col("probability_val"), 4))
        .withColumn(
            "predicted",
            F.when(F.col("probability_val") >= best_th, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("rank_in_user", F.row_number().over(w_user_rank))
        .withColumn("past_orders_count", F.col("up_buy_count").cast("int"))
        .withColumn(
            "trang_thai_goi_y",
            F.when(F.col("predicted") == 1, F.lit("Nên gợi ý mua lại")).otherwise(
                F.lit("Không gợi ý")
            ),
        )
        .select(
            "user_id",
            "product_id",
            "product_name",
            "probability",
            "predicted",
            "rank_in_user",
            "past_orders_count",
            "trang_thai_goi_y",
        )
    )

    # Lấy 2.000 dòng tiêu biểu (~50 KB) để GitHub hiển thị bảng Preview mượt mà
    pdf_out = (
        gold_reorder_display
        .orderBy(F.col("user_id").asc(), F.col("rank_in_user").asc())
        .limit(2000)
        .toPandas()
    )

    pdf_out.to_csv(github_csv_out, index=False, encoding="utf-8")
    pdf_out.to_csv(sample_csv_out, index=False, encoding="utf-8")

    print(f"-> Đã xuất CSV hiển thị bảng GitHub tại: {github_csv_out} ({len(pdf_out):,} dòng)")
    print(f"-> Đã cập nhật CSV cho Thành viên 5 tại: {sample_csv_out}")

    # Giải phóng bộ nhớ
    raw_preds.unpersist()
    train_vec.unpersist()
    test_vec.unpersist()
    df_ml.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()