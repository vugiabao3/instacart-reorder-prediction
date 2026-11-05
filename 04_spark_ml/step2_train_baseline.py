# -*- coding: utf-8 -*-
import os
import csv
import time
import platform
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.functions import vector_to_array

if platform.system() == "Windows" and os.path.exists("C:\\hadoop"):
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = os.environ.get("PATH", "") + ";C:\\hadoop\\bin"


def create_spark_session():
    return (
        SparkSession.builder.appName("TV4_Step2_Train_Compare_Models")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


def evaluate_precision_recall_at_k(pred_df, k=10):
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

    base_dir = Path(__file__).resolve().parent.parent
    gold_dir = base_dir / "04_spark_ml" / "storage" / "gold"
    features_dir = gold_dir / "gold_features"
    pred_parquet_dir = gold_dir / "gold_reorder_predictions"
    pred_parquet_dir.parent.mkdir(parents=True, exist_ok=True)

    github_csv_out = base_dir / "04_spark_ml" / "gold_reorder_predictions.csv"
    sample_csv_out = base_dir / "data_sample" / "gold_reorder_predictions.csv"
    github_metrics_out = base_dir / "04_spark_ml" / "model_metrics.csv"
    sample_metrics_out = base_dir / "data_sample" / "model_metrics.csv"
    products_csv_path = base_dir / "data_sample" / "products.csv"

    print("=== [STEP 2] HUẤN LUYỆN & SO SÁNH 3 MÔ HÌNH (LR, RF, GBT) ===")
    print(f"-> Đang đọc dữ liệu từ Parquet local: {features_dir}")
    df_ml = spark.read.parquet(features_dir.as_uri()).na.fill(0).cache()

    feature_cols = [
        "u_total_orders",
        "u_avg_days_between_orders",
        "u_total_items",
        "u_reorder_ratio",
        "u_avg_basket_size",
        "p_total_purchases",
        "p_reorder_ratio",
        "p_unique_users",
        "p_avg_add_to_cart",
        "up_buy_count",
        "up_first_order_num",
        "up_last_order_num",
        "up_avg_add_to_cart",
        "up_streak",
        "up_orders_since_last_buy",
        "up_days_since_last_buy",
        "up_buy_rate_since_first",
    ]
    available_cols = [c for c in feature_cols if c in df_ml.columns]

    users_split = df_ml.select("user_id").distinct().randomSplit([0.8, 0.2], seed=42)
    train_users, test_users = users_split[0], users_split[1]

    train_data = df_ml.join(train_users, on="user_id", how="inner")
    test_data = df_ml.join(test_users, on="user_id", how="inner")

    assembler = VectorAssembler(
        inputCols=available_cols, outputCol="raw_features", handleInvalid="keep"
    )
    train_assembled = assembler.transform(train_data)
    test_assembled = assembler.transform(test_data)

    scaler = StandardScaler(
        inputCol="raw_features", outputCol="features", withStd=True, withMean=False
    )
    scaler_model = scaler.fit(train_assembled)

    train_vec = scaler_model.transform(train_assembled).withColumn(
        "label", F.col("label").cast("double")
    )
    test_vec = (
        scaler_model.transform(test_assembled)
        .withColumn("label", F.col("label").cast("double"))
        .cache()
    )

    pos_count = train_vec.filter(F.col("label") == 1.0).count()
    total_train = train_vec.count()
    ratio = (total_train - pos_count) / float(total_train) if pos_count > 0 else 1.0

    train_weighted = train_vec.withColumn(
        "classWeightCol",
        F.when(F.col("label") == 1.0, ratio).otherwise(1.0 - ratio),
    ).cache()

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="classWeightCol",
            maxIter=30,
            regParam=0.01,
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="classWeightCol",
            numTrees=30,
            maxDepth=6,
            seed=42,
        ),
        "GBT Classifier": GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="classWeightCol",
            maxIter=20,
            maxDepth=5,
            seed=42,
        ),
    }

    eval_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    eval_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="pred_custom", metricName="f1"
    )

    metrics_rows = []
    best_overall_f1 = -1.0
    best_model_name = ""
    best_raw_preds = None
    best_th_overall = 0.50

    for model_name, clf in models.items():
        print(f"\n-> Đang huấn luyện mô hình: {model_name}...")
        t0 = time.time()
        fitted_model = clf.fit(train_weighted)
        train_sec = round(time.time() - t0, 2)

        raw_preds = (
            fitted_model.transform(test_vec)
            .withColumn("probability_val", vector_to_array("probability")[1])
            .cache()
        )

        auc_roc = eval_roc.evaluate(raw_preds)
        pr_auc = eval_pr.evaluate(raw_preds)

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

        temp_test_preds = raw_preds.select(
            F.col("user_id"),
            F.col("product_id"),
            F.round(F.col("probability_val"), 4).alias("probability"),
            F.when(F.col("probability_val") >= best_th, F.lit(1))
            .otherwise(F.lit(0))
            .alias("predicted"),
            F.col("label"),
        )

        p_at_10, r_at_10 = evaluate_precision_recall_at_k(temp_test_preds, k=10)

        msg = (
            f"   [{model_name}] AUC: {auc_roc:.4f} | PR: {pr_auc:.4f} | "
            f"Th: {best_th:.2f} | F1: {best_f1:.4f} | "
            f"P@10: {p_at_10:.4f} | R@10: {r_at_10:.4f} ({train_sec}s)"
        )
        print(msg)

        metrics_rows.append([
            model_name,
            round(auc_roc, 4),
            round(pr_auc, 4),
            round(best_th, 2),
            round(best_f1, 4),
            round(p_at_10, 4),
            round(r_at_10, 4),
        ])

        if best_f1 > best_overall_f1:
            if best_raw_preds is not None:
                best_raw_preds.unpersist()
            best_overall_f1 = best_f1
            best_model_name = model_name
            best_raw_preds = raw_preds
            best_th_overall = best_th
        else:
            raw_preds.unpersist()

    line_sep = "=" * 78
    print("\n" + line_sep)
    print("           BẢNG TỔNG HỢP SO SÁNH 3 MÔ HÌNH HỌC MÁY (TUẦN 2)")
    print(line_sep)
    for r in metrics_rows:
        row_str = (
            f"{r[0]:<30} | AUC:{r[1]:.4f} | PR:{r[2]:.4f} | "
            f"Th:{r[3]:.2f} | F1:{r[4]:.4f} | P10:{r[5]:.4f} | R10:{r[6]:.4f}"
        )
        print(row_str)
    print(line_sep)
    print(f"-> Mô hình tốt nhất: {best_model_name} (F1 = {best_overall_f1:.4f})\n")

    metrics_headers = [
        "Model",
        "AUC_ROC",
        "PR_AUC",
        "Best_Threshold",
        "F1_Score",
        "Precision_at_10",
        "Recall_at_10",
    ]
    for m_path in [github_metrics_out, sample_metrics_out]:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(metrics_headers)
            writer.writerows(metrics_rows)

    print(f"-> Đã lưu bảng so sánh 3 mô hình tại: {github_metrics_out}")
    print(f"-> Đã lưu bảng so sánh 3 mô hình tại: {sample_metrics_out}")

    gold_reorder_predictions = best_raw_preds.select(
        F.col("user_id"),
        F.col("product_id"),
        F.round(F.col("probability_val"), 4).alias("probability"),
        F.when(F.col("probability_val") >= best_th_overall, F.lit(1))
        .otherwise(F.lit(0))
        .alias("predicted"),
    )
    gold_reorder_predictions.write.mode("overwrite").parquet(pred_parquet_dir.as_uri())
    print(f"-> Đã lưu bản Parquet chuẩn tại: {pred_parquet_dir}")

    if products_csv_path.exists():
        products_df = (
            spark.read.csv(products_csv_path.as_uri(), header=True, inferSchema=True)
            .select("product_id", "product_name")
        )
        preds_with_name = best_raw_preds.join(
            F.broadcast(products_df), on="product_id", how="left"
        )
    else:
        preds_with_name = best_raw_preds.withColumn(
            "product_name", F.concat(F.lit("Product #"), F.col("product_id"))
        )

    w_user_rank = Window.partitionBy("user_id").orderBy(F.col("probability_val").desc())

    gold_reorder_display = (
        preds_with_name.withColumn("probability", F.round(F.col("probability_val"), 4))
        .withColumn(
            "predicted",
            F.when(F.col("probability_val") >= best_th_overall, F.lit(1)).otherwise(
                F.lit(0)
            ),
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

    pdf_out = (
        gold_reorder_display.orderBy(
            F.col("user_id").asc(), F.col("rank_in_user").asc()
        )
        .limit(2000)
        .toPandas()
    )

    pdf_out.to_csv(github_csv_out, index=False, encoding="utf-8")
    pdf_out.to_csv(sample_csv_out, index=False, encoding="utf-8")

    print(f"-> Đã xuất CSV GitHub tại: {github_csv_out} ({len(pdf_out):,} dòng)")
    print(f"-> Đã cập nhật CSV cho TV5 tại: {sample_csv_out}")

    best_raw_preds.unpersist()
    train_weighted.unpersist()
    test_vec.unpersist()
    df_ml.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
