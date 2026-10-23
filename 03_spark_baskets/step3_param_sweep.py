# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Buoc 3: Thu nhieu cap tham so FP-Growth, do so luat va thoi gian.

Dau vao : gold_baskets_aisle (tu step1)
Dau ra  : bang_tham_so_fpgrowth.csv

Luu y: chay day du 9 cap tren Colab mat khoang 45 phut.
Muon chay thu nhanh, dat bien moi truong QUICK=1 (chi chay 2 cap).

Cach chay:
    python 03_spark_baskets/step3_param_sweep.py
"""
import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

OUT_DIR = os.environ.get("OUT_DIR", "03_spark_baskets/output")
QUICK = os.environ.get("QUICK", "0") == "1"

DS_SUPPORT = [0.01, 0.02, 0.05]
DS_CONFIDENCE = [0.2, 0.3, 0.5]
if QUICK:
    DS_SUPPORT, DS_CONFIDENCE = [0.05], [0.3, 0.5]

spark = (SparkSession.builder.appName("tv3_param_sweep")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())

baskets = spark.read.parquet(f"{OUT_DIR}/gold_baskets_aisle")
baskets.cache()
baskets.count()   # nap vao bo nho truoc de do thoi gian cong bang

ket_qua = []
for s in DS_SUPPORT:
    for c in DS_CONFIDENCE:
        t0 = time.time()
        m = FPGrowth(itemsCol="aisle_items", minSupport=s, minConfidence=c).fit(baskets)
        n_tap = m.freqItemsets.count()
        n_luat = m.associationRules.count()
        giay = round(time.time() - t0, 1)
        ket_qua.append({"minSupport": s, "minConfidence": c,
                        "so_tap_pho_bien": n_tap, "so_luat": n_luat,
                        "thoi_gian_giay": giay})
        print(f"support={s}, confidence={c}: {n_luat} luat, {giay}s")

bang = pd.DataFrame(ket_qua)
out_path = f"{OUT_DIR}/bang_tham_so_fpgrowth.csv"
bang.to_csv(out_path, index=False)
print("\nDa luu:", out_path)
print(bang.to_string(index=False))

spark.stop()
