# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Buoc 2: Chay FP-Growth o muc aisle, tao bang luat ket hop.

Dau vao : gold_baskets_aisle (tu step1) + aisles.csv (de lay ten nhom hang)
Dau ra  : gold_rules_aisle.csv  voi cac cot:
            antecedent_ids, consequent_ids, ve_trai, ve_phai,
            support, confidence, lift, level

Cach chay:
    python 03_spark_baskets/step2_fpgrowth_aisle.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

# ------------------------------------------------------------------
# CAU HINH
# ------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "data_sample")
OUT_DIR = os.environ.get("OUT_DIR", "03_spark_baskets/output")
MIN_SUPPORT = 0.02      # nhom hang phai xuat hien trong it nhat 2% don
MIN_CONFIDENCE = 0.3    # luat phai dung it nhat 30% truong hop ve trai xuat hien
MIN_LIFT = 1.0          # chi giu luat co Lift > 1 (moi that su di kem nhau)

spark = (SparkSession.builder.appName("tv3_fpgrowth_aisle")
         .config("spark.sql.shuffle.partitions", "16")
         .getOrCreate())

baskets = spark.read.parquet(f"{OUT_DIR}/gold_baskets_aisle")
aisles = spark.read.csv(f"{DATA_DIR}/aisles.csv", header=True, inferSchema=True).toPandas()
aisle_map = dict(zip(aisles["aisle_id"], aisles["aisle"]))

# ------------------------------------------------------------------
# CHAY FP-GROWTH
# ------------------------------------------------------------------
model = FPGrowth(itemsCol="aisle_items",
                 minSupport=MIN_SUPPORT,
                 minConfidence=MIN_CONFIDENCE).fit(baskets)

rules = model.associationRules.toPandas()   # bang luat nho (~1000 dong) nen chuyen sang pandas
print(f"So luat truoc khi loc Lift: {len(rules)}")

# Giu luat co Lift > MIN_LIFT
rules = rules[rules["lift"] > MIN_LIFT].copy()
print(f"So luat sau khi loc Lift > {MIN_LIFT}: {len(rules)}")

# ------------------------------------------------------------------
# GHEP TEN NHOM HANG VA DINH DANG CAC COT
# ------------------------------------------------------------------
def to_ids(ids):
    return ",".join(str(int(i)) for i in ids)

def to_names(ids):
    return " + ".join(aisle_map.get(int(i), str(i)) for i in ids)

rules["antecedent_ids"] = rules["antecedent"].apply(to_ids)
rules["consequent_ids"] = rules["consequent"].apply(to_ids)
rules["ve_trai"] = rules["antecedent"].apply(to_names)
rules["ve_phai"] = rules["consequent"].apply(to_names)
rules["level"] = "aisle"

cot = ["antecedent_ids", "consequent_ids", "ve_trai", "ve_phai",
       "support", "confidence", "lift", "level"]
rules = rules.sort_values("lift", ascending=False)

out_path = f"{OUT_DIR}/gold_rules_aisle.csv"
rules[cot].to_csv(out_path, index=False)
print("Da luu:", out_path)

# In 10 luat ngan (ve trai toi da 2 nhom) co Lift cao nhat de xem nhanh
ngan = rules[rules["antecedent"].apply(len) <= 2]
print("\n10 luat ngan co Lift cao nhat:")
print(ngan[["ve_trai", "ve_phai", "support", "confidence", "lift"]].head(10).to_string(index=False))

spark.stop()
