from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

spark = SparkSession.builder \
    .appName("M4_Step0_Mock_Silver") \
    .config("spark.sql.warehouse.dir", os.path.abspath("./spark-warehouse")) \
    .enableHiveSupport() \
    .getOrCreate()

SAMPLE_DIR = "./data_sample"
print("=== [STEP 0] KHỞI TẠO BẢNG SILVER GIẢ LẬP TỪ DATA_SAMPLE (5%) ===")

# 1. Tạo bảng silver_orders (giữ NULL ở đơn đầu tiên và thêm cờ is_first_order)
orders_df = spark.read.csv(f"{SAMPLE_DIR}/orders.csv", header=True, inferSchema=True)
silver_orders = orders_df.withColumn(
    "is_first_order", F.when(F.col("order_number") == 1, 1).otherwise(0)
).select(
    "order_id", "user_id", "order_number", "order_dow",
    "order_hour_of_day", "days_since_prior_order", "is_first_order", "eval_set"
)

silver_orders.write.mode("overwrite").format("parquet").saveAsTable("silver_orders")
print(f" -> Đã tạo bảng silver_orders: {silver_orders.count():,} dòng.")

# 2. Tạo bảng silver_order_products (gộp prior và train)
op_prior = spark.read.csv(f"{SAMPLE_DIR}/order_products__prior.csv", header=True, inferSchema=True)
op_train = spark.read.csv(f"{SAMPLE_DIR}/order_products__train.csv", header=True, inferSchema=True)

silver_op = op_prior.unionByName(op_train).select(
    "order_id", "product_id", "add_to_cart_order", "reordered"
)

silver_op.write.mode("overwrite").format("parquet").saveAsTable("silver_order_products")
print(f" -> Đã tạo bảng silver_order_products: {silver_op.count():,} dòng.")

spark.stop()
