# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Buoc 1: Tao gio hang (baskets) tu du lieu don hang.

Dau vao : order_products__prior, products  (CSV mau 5% hoac bang Silver)
Dau ra  : 2 bang Parquet
            - gold_baskets_product : order_id, items (mang product_id)
            - gold_baskets_aisle   : order_id, aisle_items (mang aisle_id)

Cach chay (tu thu muc goc cua repo):
    python 03_spark_baskets/step1_build_baskets.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, size

# ------------------------------------------------------------------
# CAU HINH - chi sua o day
# ------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data_sample")          # thu muc chua CSV mau
OUT_DIR = os.environ.get("OUT_DIR", "03_spark_baskets/output")  # noi luu ket qua

spark = (SparkSession.builder.appName("tv3_build_baskets")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())

# ------------------------------------------------------------------
# 1) DOC DU LIEU
#    Hien tai doc tu CSV mau. Khi Thanh vien 2 xong bang Silver, thay bang:
#      prior    = spark.table("silver_order_products")
#      products = spark.table("silver_products")
# ------------------------------------------------------------------
prior = spark.read.csv(f"{DATA_DIR}/order_products__prior.csv", header=True, inferSchema=True)
products = spark.read.csv(f"{DATA_DIR}/products.csv", header=True, inferSchema=True)

# ------------------------------------------------------------------
# 2) GIO HANG MUC SAN PHAM
#    collect_set: gom cac san pham cua moi don thanh 1 mang, bo trung
# ------------------------------------------------------------------
baskets_product = (prior.groupBy("order_id")
                   .agg(collect_set("product_id").alias("items")))

# ------------------------------------------------------------------
# 3) GIO HANG MUC AISLE (nhom hang)
#    Noi voi bang products de biet moi san pham thuoc aisle nao
# ------------------------------------------------------------------
prior_aisle = prior.join(products.select("product_id", "aisle_id"),
                         on="product_id", how="left")
baskets_aisle = (prior_aisle.groupBy("order_id")
                 .agg(collect_set("aisle_id").alias("aisle_items")))

# ------------------------------------------------------------------
# 4) KIEM TRA (theo yeu cau cua de: so gio hang = so don co san pham)
# ------------------------------------------------------------------
n_orders = prior.select("order_id").distinct().count()
n_bp = baskets_product.count()
n_ba = baskets_aisle.count()
print(f"So don co san pham      : {n_orders}")
print(f"So gio hang muc san pham: {n_bp}")
print(f"So gio hang muc aisle   : {n_ba}")
assert n_orders == n_bp == n_ba, "SO GIO HANG KHONG KHOP SO DON - kiem tra lai phep noi bang!"

# ------------------------------------------------------------------
# 5) LUU KET QUA
# ------------------------------------------------------------------
baskets_product.write.mode("overwrite").parquet(f"{OUT_DIR}/gold_baskets_product")
baskets_aisle.write.mode("overwrite").parquet(f"{OUT_DIR}/gold_baskets_aisle")
print("Da luu vao:", OUT_DIR)

spark.stop()
