# -*- coding: utf-8 -*-
"""
Thanh vien 3 - Muc 3.3: Word2Vec (item2vec) va so sanh voi luat ket hop FP-Growth.

Y tuong: coi moi gio hang la mot "cau", moi san pham la mot "tu". Word2Vec hoc
vector cho tung san pham sao cho cac san pham hay xuat hien cung gio hang co
vector gan nhau. Sau do:
  (1) Tim san pham gan nhat (tuong dong cosine) voi mot so san pham vi du.
  (2) So sanh voi luat ket hop o MUC AISLE: voi moi luat "1 nhom -> 1 nhom",
      xem nhom ve phai cua luat co nam trong top-K nhom gan nhat (theo vector)
      cua nhom ve trai hay khong.

Dau vao : order_products__prior.csv, products.csv, aisles.csv (thu muc DATA_DIR)
          gold_rules_aisle.csv (do run_baskets_fpgrowth.py tao ra, trong OUT_DIR)
Dau ra  : (trong OUT_DIR)
          gold_product_embeddings.csv  (product_id, vector)
          word2vec_neighbors.csv       (san pham vi du va 5 san pham gan nhat)
          word2vec_vs_rules.csv        (thu hang nhom ve phai theo vector cho tung luat)

Cach chay (Windows, tu thu muc goc cua repo):
    set OUT_DIR=C:\\instacart_output
    python 03_spark_baskets\\run_word2vec_item2vec.py

Bien moi truong tuy chon:
    DATA_DIR, OUT_DIR, SPARK_MEM, SHUFFLE_PARTS  (nhu cac script khac)
    RULES        duong dan file luat   (mac dinh: OUT_DIR/gold_rules_aisle.csv)
    VECTOR_SIZE  so chieu vector       (mac dinh 50)
    MIN_COUNT    san pham xuat hien it hon so nay bi bo qua (mac dinh 5)
    WINDOW       cua so ngu canh       (mac dinh 10, gan bang do dai gio hang trung binh)
    MAX_ITER     so vong lap huan luyen (mac dinh 5)
    NUM_PARTITIONS  so phan chia khi huan luyen (mac dinh 4; cao hon thi nhanh hon
                    nhung do chinh xac giam nhe; dat 1 de chinh xac nhat nhung cham)
    LIMIT_BASKETS   chi dung N gio hang dau de chay thu nhanh (mac dinh 0 = dung het)
    SEED         hat giong ngau nhien  (mac dinh 42, de ket qua lap lai duoc)
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set, col
from pyspark.ml.feature import Word2Vec

# ------------------------------------------------------------------
# CAU HINH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value):
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_DIR = resolve_path(os.environ.get("DATA_DIR", "data_sample"))
OUT_DIR = resolve_path(os.environ.get("OUT_DIR", "03_spark_baskets/output"))
RULES_FILE = resolve_path(os.environ.get("RULES", str(OUT_DIR / "gold_rules_aisle.csv")))
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "50"))
MIN_COUNT = int(os.environ.get("MIN_COUNT", "5"))
WINDOW = int(os.environ.get("WINDOW", "10"))
MAX_ITER = int(os.environ.get("MAX_ITER", "5"))
SEED = int(os.environ.get("SEED", "42"))
NUM_PARTITIONS = int(os.environ.get("NUM_PARTITIONS", "4"))
LIMIT_BASKETS = int(os.environ.get("LIMIT_BASKETS", "0"))
TOP_NEIGHBORS = 5

PRIOR_FILE = DATA_DIR / "order_products__prior.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
AISLES_FILE = DATA_DIR / "aisles.csv"
for f in (PRIOR_FILE, PRODUCTS_FILE, AISLES_FILE):
    if not f.is_file():
        raise FileNotFoundError(f"Khong tim thay file dau vao: {f}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

spark = (SparkSession.builder.appName("tv3_word2vec")
         .config("spark.driver.memory", os.environ.get("SPARK_MEM", "4g"))
         .config("spark.sql.shuffle.partitions", os.environ.get("SHUFFLE_PARTS", "16"))
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

try:
    # --------------------------------------------------------------
    # 1) GIO HANG MUC SAN PHAM (moi san pham la 1 "tu" dang chuoi)
    # --------------------------------------------------------------
    print("Dang doc du lieu va tao gio hang...")
    prior = spark.read.csv(PRIOR_FILE.as_uri(), header=True, inferSchema=True)
    baskets = prior.groupBy("order_id").agg(
        collect_set(col("product_id").cast("string")).alias("items"))
    if LIMIT_BASKETS > 0:
        baskets = baskets.limit(LIMIT_BASKETS)
        print(f"[CHAY THU] Chi dung {LIMIT_BASKETS:,} gio hang dau tien.")
    baskets.cache()
    print(f"So gio hang: {baskets.count():,}")

    # --------------------------------------------------------------
    # 2) HUAN LUYEN WORD2VEC
    # --------------------------------------------------------------
    print(f"Dang huan luyen Word2Vec (vectorSize={VECTOR_SIZE}, minCount={MIN_COUNT}, "
          f"window={WINDOW}, maxIter={MAX_ITER}, partitions={NUM_PARTITIONS}, seed={SEED})...")
    t0 = time.time()
    w2v = Word2Vec(vectorSize=VECTOR_SIZE, minCount=MIN_COUNT, windowSize=WINDOW,
                   maxIter=MAX_ITER, seed=SEED, numPartitions=NUM_PARTITIONS,
                   inputCol="items", outputCol="vec")
    model = w2v.fit(baskets)
    vec_df = model.getVectors().toPandas()          # cot: word, vector
    print(f"Huan luyen xong sau {time.time() - t0:.0f} giay. "
          f"So san pham co vector: {len(vec_df):,}")

    # Dem so lan mua cua tung san pham (de chon vi du)
    counts = prior.groupBy("product_id").count().toPandas().set_index("product_id")["count"]
finally:
    spark.stop()

# ------------------------------------------------------------------
# 3) CHUAN BI MA TRAN VECTOR (tu day chi dung pandas/numpy)
# ------------------------------------------------------------------
products = pd.read_csv(PRODUCTS_FILE)
aisles = pd.read_csv(AISLES_FILE)
name_of = dict(zip(products["product_id"], products["product_name"]))
aisle_of = dict(zip(products["product_id"], products["aisle_id"]))
aisle_name = dict(zip(aisles["aisle_id"], aisles["aisle"]))

pids = vec_df["word"].astype(int).to_numpy()
V = np.array([v.toArray() for v in vec_df["vector"]])
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)          # chuan hoa de dung cosine

# Luu vector (hop dong du lieu: gold_product_embeddings gom product_id, vector)
emb = pd.DataFrame({"product_id": pids,
                    "vector": [" ".join(f"{x:.5f}" for x in row) for row in V]})
emb.to_csv(OUT_DIR / "gold_product_embeddings.csv", index=False)
print("Da luu:", OUT_DIR / "gold_product_embeddings.csv")

# ------------------------------------------------------------------
# 4) SAN PHAM GAN NHAU (vi du dinh tinh cho bao cao)
# ------------------------------------------------------------------
rng = np.random.default_rng(SEED)
pop = counts.reindex(pids).fillna(0)
top500 = pop.sort_values(ascending=False).head(500).index.to_numpy()
examples = list(pop.sort_values(ascending=False).head(3).index) + \
           list(rng.choice(top500[3:], size=5, replace=False))
idx_of = {p: i for i, p in enumerate(pids)}

rows = []
print("\nSan pham vi du va 5 san pham gan nhat (cosine):")
for p in examples:
    sims = Vn @ Vn[idx_of[p]]
    order = np.argsort(-sims)
    nb = [i for i in order if pids[i] != p][:TOP_NEIGHBORS]
    print(f"  {name_of[p]}  ->  " + "; ".join(f"{name_of[pids[i]]} ({sims[i]:.2f})" for i in nb))
    rows.append({"product_id": p, "product_name": name_of[p],
                 "neighbors": " | ".join(f"{name_of[pids[i]]} ({sims[i]:.2f})" for i in nb)})
pd.DataFrame(rows).to_csv(OUT_DIR / "word2vec_neighbors.csv", index=False)

# ------------------------------------------------------------------
# 5) SO SANH VOI LUAT KET HOP FP-GROWTH (muc aisle)
# ------------------------------------------------------------------
# Vector cua mot nhom hang = trung binh vector da chuan hoa cua cac san pham trong nhom
aisle_ids = sorted({aisle_of[p] for p in pids if p in aisle_of})
A = np.zeros((len(aisle_ids), V.shape[1]))
for k, a in enumerate(aisle_ids):
    members = [i for i, p in enumerate(pids) if aisle_of.get(p) == a]
    A[k] = Vn[members].mean(axis=0)
A = A / np.linalg.norm(A, axis=1, keepdims=True)
S = A @ A.T
np.fill_diagonal(S, -np.inf)
pos = {a: k for k, a in enumerate(aisle_ids)}

if not RULES_FILE.is_file():
    print(f"\n[CANH BAO] Khong thay file luat {RULES_FILE}; bo qua phan so sanh.")
else:
    rules = pd.read_csv(RULES_FILE)
    ant_col = "antecedent_ids" if "antecedent_ids" in rules.columns else "antecedent"
    con_col = "consequent_ids" if "consequent_ids" in rules.columns else "consequent"
    one = rules[(~rules[ant_col].astype(str).str.contains(",")) &
                (~rules[con_col].astype(str).str.contains(","))].copy()
    one["a"] = one[ant_col].astype(int)
    one["c"] = one[con_col].astype(int)
    one = one[one["a"].isin(pos) & one["c"].isin(pos)].copy()

    ranks = []
    for a, c in zip(one["a"], one["c"]):
        sims = S[pos[a]]
        ranks.append(1 + int((sims > sims[pos[c]]).sum()))   # 1 = gan nhat
    one["hang_theo_vector"] = ranks
    one["ve_trai"] = one["a"].map(aisle_name)
    one["ve_phai"] = one["c"].map(aisle_name)
    n_aisle = len(aisle_ids)
    one[["ve_trai", "ve_phai", "support", "confidence", "lift", "hang_theo_vector"]] \
        .to_csv(OUT_DIR / "word2vec_vs_rules.csv", index=False)

    print(f"\nSo sanh voi {len(one)} luat 1 nhom -> 1 nhom (tong {n_aisle} nhom hang):")
    for k in (1, 3, 5, 10):
        hit = (one["hang_theo_vector"] <= k).mean()
        base = k / (n_aisle - 1)
        print(f"  Nhom ve phai nam trong top-{k:<2} nhom gan nhat theo vector: "
              f"{hit:6.1%}   (ngau nhien: {base:5.1%})")
    print(f"  Hang trung vi cua nhom ve phai: {one['hang_theo_vector'].median():.0f} / {n_aisle - 1}")

    print("\n5 luat 1->1 co Lift cao nhat: nhom ve phai xep hang may theo vector?")
    for _, r in one.sort_values("lift", ascending=False).head(5).iterrows():
        near = np.argsort(-S[pos[r["a"]]])[:TOP_NEIGHBORS]
        print(f"  {r['ve_trai']} -> {r['ve_phai']} (Lift {r['lift']:.2f}): "
              f"hang {int(r['hang_theo_vector'])}; top-{TOP_NEIGHBORS} theo vector: "
              + ", ".join(aisle_name[aisle_ids[i]] for i in near))
    print("Da luu:", OUT_DIR / "word2vec_vs_rules.csv")

print("\nHOAN THANH!")
