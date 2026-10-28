
# -*- coding: utf-8 -*-
"""
Buoc 2: Chay FP-Growth tren gio hang aisle.
Doc va xu ly Parquet truc tiep bang Spark.
"""

import os
from pathlib import Path

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth


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

MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.3
MIN_LIFT = 1.0

INPUT_PATH = OUT_DIR / "gold_baskets_aisle"
OUTPUT_PATH = OUT_DIR / "gold_rules_aisle.csv"
AISLES_FILE = DATA_DIR / "aisles.csv"


# ============================================================
# 2. KIEM TRA DAU VAO
# ============================================================

if os.name == "nt":
    if not WINUTILS.is_file():
        raise FileNotFoundError(
            f"Khong tim thay winutils.exe: {WINUTILS}"
        )

    os.environ["HADOOP_HOME"] = str(HADOOP_HOME)
    os.environ["hadoop.home.dir"] = str(HADOOP_HOME)
    os.environ["PATH"] = (
        str(HADOOP_HOME / "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )

if not INPUT_PATH.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay thu muc Parquet: {INPUT_PATH}. "
        "Hay chay Step 1 truoc."
    )

if not AISLES_FILE.is_file():
    raise FileNotFoundError(
        f"Khong tim thay file: {AISLES_FILE}"
    )


# ============================================================
# 3. KHOI TAO SPARK VA DOC PARQUET
# ============================================================

spark = (
    SparkSession.builder
    .appName("tv3_fpgrowth_aisle")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("Dang doc baskets Parquet bang Spark...")

    baskets = spark.read.parquet(str(INPUT_PATH)).cache()
    n_baskets = baskets.count()

    print(f"So gio hang doc duoc: {n_baskets:,}")

    if n_baskets == 0:
        raise ValueError("Du lieu baskets rong.")

    # --------------------------------------------------------
    # 4. DOC TEN AISLE
    # --------------------------------------------------------

    aisles = pd.read_csv(AISLES_FILE)
    aisle_map = dict(zip(aisles["aisle_id"], aisles["aisle"]))

    # --------------------------------------------------------
    # 5. CHAY FP-GROWTH
    # --------------------------------------------------------

    print("Dang chay FP-Growth...")

    model = FPGrowth(
        itemsCol="aisle_items",
        minSupport=MIN_SUPPORT,
        minConfidence=MIN_CONFIDENCE
    ).fit(baskets)

    # So luat thuong nho hon nhieu so dong baskets.
    rules = model.associationRules.toPandas()

    print(f"So luat truoc khi loc Lift: {len(rules):,}")

    rules = rules[rules["lift"] > MIN_LIFT].copy()

    print(f"So luat sau khi loc Lift > {MIN_LIFT}: {len(rules):,}")

    # --------------------------------------------------------
    # 6. DOI ID THANH TEN AISLE
    # --------------------------------------------------------

    def to_ids(ids):
        return ",".join(str(int(i)) for i in ids)

    def to_names(ids):
        return " + ".join(
            str(aisle_map.get(int(i), i))
            for i in ids
        )

    rules["antecedent_ids"] = rules["antecedent"].apply(to_ids)
    rules["consequent_ids"] = rules["consequent"].apply(to_ids)
    rules["ve_trai"] = rules["antecedent"].apply(to_names)
    rules["ve_phai"] = rules["consequent"].apply(to_names)
    rules["level"] = "aisle"

    columns = [
        "antecedent_ids",
        "consequent_ids",
        "ve_trai",
        "ve_phai",
        "support",
        "confidence",
        "lift",
        "level"
    ]

    rules = rules.sort_values("lift", ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules[columns].to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDa luu {len(rules):,} luat: {OUTPUT_PATH}")
    print("\n10 luat co Lift cao nhat:")

    short_rules = rules[rules["antecedent"].apply(len) <= 2]

    print(
        short_rules[
            ["ve_trai", "ve_phai", "support", "confidence", "lift"]
        ].head(10).to_string(index=False)
    )

finally:
    if "baskets" in locals():
        baskets.unpersist()
    spark.stop()
