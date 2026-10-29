
# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Buoc 1: Tao gio hang (baskets).

Dau vao:
    data_sample/order_products__prior.csv
    data_sample/products.csv

Dau ra:
    03_spark_baskets/output/gold_baskets_product/
    03_spark_baskets/output/gold_baskets_aisle/

Spark xu ly va ghi Parquet truc tiep.
Windows su dung winutils.exe qua HADOOP_HOME.
"""

import os
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set


# ============================================================
# 1. CAU HINH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HADOOP_HOME = Path(os.environ.get("HADOOP_HOME", r"C:\hadoop"))
WINUTILS = HADOOP_HOME / "bin" / "winutils.exe"


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


DATA_DIR = resolve_path(os.environ.get("DATA_DIR", "data_sample"))
OUT_DIR = resolve_path(
    os.environ.get("OUT_DIR", "03_spark_baskets/output")
)

PRIOR_FILE = DATA_DIR / "order_products__prior.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"


# ============================================================
# 2. KIEM TRA WINDOWS HADOOP
# ============================================================

if os.name == "nt":
    if not WINUTILS.is_file():
        raise FileNotFoundError(
            f"Khong tim thay winutils.exe: {WINUTILS}\n"
            "Hay cai winutils.exe tuong thich voi Hadoop cua Spark "
            "vao thu muc HADOOP_HOME\\bin."
        )

    os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
    os.environ["hadoop.home.dir"] = str(HADOOP_HOME)
    os.environ["PATH"] = (
        str(HADOOP_HOME / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

for file_path in [PRIOR_FILE, PRODUCTS_FILE]:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file dau vao: {file_path}"
        )


# ============================================================
# 3. KHOI TAO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("tv3_build_baskets")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


def save_baskets(dataframe, output_path):
    """Ghi Spark DataFrame truc tiep thanh thu muc Parquet."""

    output_path = Path(output_path)
    print(f"Dang ghi Parquet: {output_path}")

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spark tu tao thu muc va cac file part-*.parquet.
    dataframe.write.mode("overwrite").parquet(str(output_path))

    print(f"Da ghi xong: {output_path}")


try:
    # --------------------------------------------------------
    # 4. DOC CSV
    # --------------------------------------------------------

    print("Dang doc du lieu CSV...")

    prior = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(PRIOR_FILE))
    )

    products = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(PRODUCTS_FILE))
    )

    # --------------------------------------------------------
    # 5. GIO HANG THEO SAN PHAM
    # --------------------------------------------------------

    print("Dang tao gio hang muc san pham...")

    baskets_product = (
        prior
        .groupBy("order_id")
        .agg(collect_set("product_id").alias("items"))
    )

    # --------------------------------------------------------
    # 6. GIO HANG THEO AISLE
    # --------------------------------------------------------

    print("Dang tao gio hang muc aisle...")

    prior_aisle = prior.join(
        products.select("product_id", "aisle_id"),
        on="product_id",
        how="left"
    )

    baskets_aisle = (
        prior_aisle
        .groupBy("order_id")
        .agg(collect_set("aisle_id").alias("aisle_items"))
    )

    # --------------------------------------------------------
    # 7. KIEM TRA SO LUONG
    # --------------------------------------------------------

    n_orders = prior.select("order_id").distinct().count()
    n_product = baskets_product.count()
    n_aisle = baskets_aisle.count()

    print(f"So don hang co san pham : {n_orders:,}")
    print(f"So gio hang product     : {n_product:,}")
    print(f"So gio hang aisle       : {n_aisle:,}")

    assert n_orders == n_product == n_aisle, (
        "So luong gio hang khong khop voi so don hang."
    )

    # --------------------------------------------------------
    # 8. GHI PARQUET BANG SPARK
    # --------------------------------------------------------

    save_baskets(
        baskets_product,
        OUT_DIR / "gold_baskets_product"
    )

    save_baskets(
        baskets_aisle,
        OUT_DIR / "gold_baskets_aisle"
    )

    print("\nHOAN THANH STEP 1!")
    print(f"Thu muc ket qua: {OUT_DIR}")

finally:
    spark.stop()
