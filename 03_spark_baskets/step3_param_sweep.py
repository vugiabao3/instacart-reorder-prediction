
# -*- coding: utf-8 -*-
"""
Buoc 3: Khao sat tham so FP-Growth.
Doc Parquet truc tiep bang Spark.
"""

import os
import time
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


OUT_DIR = resolve_path(
    os.environ.get("OUT_DIR", "03_spark_baskets/output")
)

QUICK = os.environ.get("QUICK", "0") == "1"

INPUT_PATH = OUT_DIR / "gold_baskets_aisle"
OUTPUT_PATH = OUT_DIR / "bang_tham_so_fpgrowth.csv"

DS_SUPPORT = [0.01, 0.02, 0.05]
DS_CONFIDENCE = [0.2, 0.3, 0.5]

if QUICK:
    DS_SUPPORT = [0.05]
    DS_CONFIDENCE = [0.3, 0.5]


# ============================================================
# 2. KIEM TRA WINDOWS HADOOP VA DAU VAO
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


# ============================================================
# 3. DOC DU LIEU VA KHOI TAO SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("tv3_param_sweep")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    print("Dang doc baskets Parquet bang Spark...")

    baskets = spark.read.parquet(str(INPUT_PATH)).cache()
    n_baskets = baskets.count()

    print(f"So gio hang: {n_baskets:,}")

    if n_baskets == 0:
        raise ValueError("Du lieu baskets rong.")

    ket_qua = []

    # --------------------------------------------------------
    # 4. THU CAC CAP THAM SO
    # --------------------------------------------------------

    for support in DS_SUPPORT:
        for confidence in DS_CONFIDENCE:
            print(
                f"\nDang thu support={support}, "
                f"confidence={confidence}..."
            )

            start_time = time.time()

            model = FPGrowth(
                itemsCol="aisle_items",
                minSupport=support,
                minConfidence=confidence
            ).fit(baskets)

            n_itemsets = model.freqItemsets.count()
            n_rules = model.associationRules.count()
            elapsed = round(time.time() - start_time, 1)

            ket_qua.append({
                "minSupport": support,
                "minConfidence": confidence,
                "so_tap_pho_bien": n_itemsets,
                "so_luat": n_rules,
                "thoi_gian_giay": elapsed
            })

            print(
                f"support={support}, confidence={confidence}: "
                f"{n_itemsets:,} tap pho bien, "
                f"{n_rules:,} luat, {elapsed}s"
            )

    # --------------------------------------------------------
    # 5. LUU KET QUA
    # --------------------------------------------------------

    bang = pd.DataFrame(ket_qua)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bang.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nDa luu: {OUTPUT_PATH}")
    print(bang.to_string(index=False))

finally:
    if "baskets" in locals():
        baskets.unpersist()
    spark.stop()
