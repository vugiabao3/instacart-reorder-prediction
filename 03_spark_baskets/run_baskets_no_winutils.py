# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Ban KHONG CAN winutils.exe (gop step1 + step2 + step3).

Y tuong: Spark chi DOC CSV va tinh toan trong bo nho. Moi thao tac GHI file
(ket qua luat, bang tham so) deu do pandas (Python) lam, nen khong di qua
phan ghi file cua Hadoop - noi can winutils.exe tren Windows.

Viec lam:
  1) Doc order_products__prior.csv + products.csv
  2) Tao gio hang muc san pham va muc aisle (trong bo nho), kiem tra so luong
  3) Chay FP-Growth muc aisle, loc Lift > 1, ghep ten nhom hang
  4) Luu gold_rules_aisle.csv  (cac cot: antecedent_ids, consequent_ids,
     ve_trai, ve_phai, support, confidence, lift, level)
  5) (tuy chon) SWEEP=1: thu nhieu cap tham so -> bang_tham_so_fpgrowth.csv
  6) (tuy chon) SAVE_BASKETS=1: luu gio hang muc aisle bang pandas

Cach chay (Windows, tu thu muc goc cua repo):
    set OUT_DIR=C:\\instacart_output
    python 03_spark_baskets\\run_baskets_no_winutils.py

Bien moi truong tuy chon:
    DATA_DIR        thu muc chua CSV              (mac dinh: data_sample)
    OUT_DIR         thu muc luu ket qua           (mac dinh: 03_spark_baskets/output)
    SPARK_MEM       bo nho cho Spark              (mac dinh: 4g)
    MIN_SUPPORT     mac dinh 0.02
    MIN_CONFIDENCE  mac dinh 0.3
    SWEEP=1         chay them bang thu tham so (QUICK=1 de chi chay 2 cap)
    SAVE_BASKETS=1  luu gio hang muc aisle ra Parquet bang pandas (can pyarrow)
    READ_WITH_PANDAS=1  doc CSV bang pandas thay cho Spark (du phong, chi hop voi mau nho)
"""
import os
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
from pyspark.ml.fpm import FPGrowth

# ------------------------------------------------------------------
# CAU HINH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_DIR = resolve_path(os.environ.get("DATA_DIR", "data_sample"))
OUT_DIR = resolve_path(os.environ.get("OUT_DIR", "03_spark_baskets/output"))
MIN_SUPPORT = float(os.environ.get("MIN_SUPPORT", "0.02"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.3"))
MIN_LIFT = 1.0
SWEEP = os.environ.get("SWEEP", "0") == "1"
QUICK = os.environ.get("QUICK", "0") == "1"
SAVE_BASKETS = os.environ.get("SAVE_BASKETS", "0") == "1"
READ_WITH_PANDAS = os.environ.get("READ_WITH_PANDAS", "0") == "1"

PRIOR_FILE = DATA_DIR / "order_products__prior.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
AISLES_FILE = DATA_DIR / "aisles.csv"

for f in (PRIOR_FILE, PRODUCTS_FILE, AISLES_FILE):
    if not f.is_file():
        raise FileNotFoundError(f"Khong tim thay file dau vao: {f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# KHOI TAO SPARK
# ------------------------------------------------------------------
spark = (SparkSession.builder.appName("tv3_baskets_no_winutils")
         .config("spark.driver.memory", os.environ.get("SPARK_MEM", "4g"))
         .config("spark.sql.shuffle.partitions", os.environ.get("SHUFFLE_PARTS", "16"))
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")  # an bot canh bao winutils/Hadoop


def read_csv(path):
    """Doc CSV bang Spark; neu loi (hoac READ_WITH_PANDAS=1) thi doc bang pandas."""
    if not READ_WITH_PANDAS:
        try:
            df = spark.read.csv(path.as_uri(), header=True, inferSchema=True)
            df.limit(1).collect()   # ep doc thu de phat hien loi som
            return df
        except Exception as e:
            print(f"[CANH BAO] Spark khong doc duoc {path.name}: {str(e)[:150]}")
            print("           Chuyen sang doc bang pandas (phu hop voi du lieu mau).")
    return spark.createDataFrame(pd.read_csv(path))


try:
    # --------------------------------------------------------------
    # 1) DOC DU LIEU
    # --------------------------------------------------------------
    print("Dang doc du lieu CSV...")
    prior = read_csv(PRIOR_FILE)
    products = read_csv(PRODUCTS_FILE)
    aisles = pd.read_csv(AISLES_FILE)
    aisle_map = dict(zip(aisles["aisle_id"], aisles["aisle"]))

    # --------------------------------------------------------------
    # 2) GIO HANG (trong bo nho, khong ghi file)
    # --------------------------------------------------------------
    print("Dang tao gio hang...")
    baskets_product = prior.groupBy("order_id").agg(
        collect_set("product_id").alias("items"))

    prior_aisle = prior.join(products.select("product_id", "aisle_id"),
                             on="product_id", how="left")
    baskets_aisle = prior_aisle.groupBy("order_id").agg(
        collect_set("aisle_id").alias("aisle_items"))
    baskets_aisle.cache()

    n_orders = prior.select("order_id").distinct().count()
    n_product = baskets_product.count()
    n_aisle = baskets_aisle.count()
    print(f"So don hang co san pham : {n_orders:,}")
    print(f"So gio hang product     : {n_product:,}")
    print(f"So gio hang aisle       : {n_aisle:,}")
    assert n_orders == n_product == n_aisle, "So gio hang khong khop so don hang."

    if SAVE_BASKETS:
        print("Dang luu gio hang muc aisle bang pandas...")
        pdf = baskets_aisle.toPandas()
        pdf["aisle_items"] = pdf["aisle_items"].apply(list)
        try:
            pdf.to_parquet(OUT_DIR / "gold_baskets_aisle.parquet", index=False)
            print("Da luu:", OUT_DIR / "gold_baskets_aisle.parquet")
        except ImportError:
            # Thieu pyarrow: luu CSV, cac aisle_id cach nhau bang dau phay
            pdf["aisle_items"] = pdf["aisle_items"].apply(lambda x: ",".join(map(str, x)))
            pdf.to_csv(OUT_DIR / "gold_baskets_aisle.csv", index=False)
            print("Khong co pyarrow -> da luu CSV:", OUT_DIR / "gold_baskets_aisle.csv")

    # --------------------------------------------------------------
    # 3) FP-GROWTH MUC AISLE
    # --------------------------------------------------------------
    print(f"Dang chay FP-Growth (minSupport={MIN_SUPPORT}, minConfidence={MIN_CONFIDENCE})...")
    model = FPGrowth(itemsCol="aisle_items", minSupport=MIN_SUPPORT,
                     minConfidence=MIN_CONFIDENCE).fit(baskets_aisle)
    rules = model.associationRules.toPandas()
    print(f"So luat truoc khi loc Lift: {len(rules)}")
    rules = rules[rules["lift"] > MIN_LIFT].copy()
    print(f"So luat sau khi loc Lift > {MIN_LIFT}: {len(rules)}")

    def to_ids(ids):
        return ",".join(str(int(i)) for i in ids)

    def to_names(ids):
        return " + ".join(aisle_map.get(int(i), str(i)) for i in ids)

    rules["antecedent_ids"] = rules["antecedent"].apply(to_ids)
    rules["consequent_ids"] = rules["consequent"].apply(to_ids)
    rules["ve_trai"] = rules["antecedent"].apply(to_names)
    rules["ve_phai"] = rules["consequent"].apply(to_names)
    rules["level"] = "aisle"
    rules = rules.sort_values("lift", ascending=False)

    cot = ["antecedent_ids", "consequent_ids", "ve_trai", "ve_phai",
           "support", "confidence", "lift", "level"]
    out_rules = OUT_DIR / "gold_rules_aisle.csv"
    rules[cot].to_csv(out_rules, index=False)
    print("Da luu:", out_rules)

    ngan = rules[rules["antecedent"].apply(len) <= 2]
    print("\n10 luat ngan (ve trai <= 2 nhom) co Lift cao nhat:")
    print(ngan[["ve_trai", "ve_phai", "support", "confidence", "lift"]]
          .head(10).to_string(index=False))

    # --------------------------------------------------------------
    # 4) (TUY CHON) THU NHIEU CAP THAM SO
    # --------------------------------------------------------------
    if SWEEP:
        ds_support = [0.05] if QUICK else [0.01, 0.02, 0.05]
        ds_conf = [0.3, 0.5] if QUICK else [0.2, 0.3, 0.5]
        ket_qua = []
        for s in ds_support:
            for c in ds_conf:
                t0 = time.time()
                m = FPGrowth(itemsCol="aisle_items", minSupport=s,
                             minConfidence=c).fit(baskets_aisle)
                n_tap = m.freqItemsets.count()
                n_luat = m.associationRules.count()
                giay = round(time.time() - t0, 1)
                ket_qua.append({"minSupport": s, "minConfidence": c,
                                "so_tap_pho_bien": n_tap, "so_luat": n_luat,
                                "thoi_gian_giay": giay})
                print(f"support={s}, confidence={c}: {n_luat} luat, {giay}s")
        bang = pd.DataFrame(ket_qua)
        out_sweep = OUT_DIR / "bang_tham_so_fpgrowth.csv"
        bang.to_csv(out_sweep, index=False)
        print("Da luu:", out_sweep)
        print(bang.to_string(index=False))

    print("\nHOAN THANH!")
    print(f"Thu muc ket qua: {OUT_DIR}")

finally:
    spark.stop()
