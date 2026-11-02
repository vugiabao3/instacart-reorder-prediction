import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# 1. CẤU HÌNH ĐƯỜNG DẪN (Đã comment lại vì chạy trên máy ảo Linux)
# os.environ['HADOOP_HOME'] = 'C:\\hadoop'
# os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
# os.environ['PATH'] = os.environ['PATH'] + ';C:\\hadoop\\bin'

# 2. KHỞI TẠO SPARK
print("Đang khởi tạo Spark ")
spark = SparkSession.builder \
    .appName("Instacart_TV2_Week2") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 3. ĐỌC DỮ LIỆU TỪ HDFS (Đã sửa lỗi mất header do Sqoop)
print("Đang đọc dữ liệu từ HDFS")
base_hdfs = "hdfs://localhost:9000/user/student/instacart"

# Đọc dữ liệu với header=False
df_orders = spark.read.csv(f"{base_hdfs}/orders", header=False, inferSchema=True)
df_prior = spark.read.csv(f"{base_hdfs}/order_products__prior", header=False, inferSchema=True)
df_products = spark.read.csv(f"{base_hdfs}/products", header=False, inferSchema=True)
df_departments = spark.read.csv(f"{base_hdfs}/departments", header=False, inferSchema=True)

# Gán lại tên cột chuẩn cho từng bảng
df_orders = df_orders.toDF("order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set")
df_prior = df_prior.toDF("order_id", "product_id", "add_to_cart_order", "reordered")
df_products = df_products.toDF("product_id", "product_name", "aisle_id", "department_id")
df_departments = df_departments.toDF("department_id", "department")
# 4. LÀM SẠCH DỮ LIỆU (TẠO TẦNG SILVER)
print("Đang làm sạch dữ liệu")
df_orders_clean = df_orders.withColumn(
    "is_first_order", 
    when(col("days_since_prior_order").isNull(), 1).otherwise(0)
).dropna(subset=["user_id", "order_id"])

df_prior_clean = df_prior.dropna(subset=["order_id", "product_id"]).dropDuplicates()

# Lưu Parquet lên thẳng HDFS
df_orders_clean.write.mode("overwrite").parquet(f"{base_hdfs}/silver/silver_orders_parquet")
df_prior_clean.write.mode("overwrite").parquet(f"{base_hdfs}/silver/silver_order_products_parquet")
print("-> Đã lưu 2 bảng Parquet (Silver) thành công trên HDFS!")

# 5. VẼ CÁC BIỂU ĐỒ EDA NÂNG CAO
print("Đang vẽ biểu đồ")
sns.set_theme(style="whitegrid")

# BIỂU ĐỒ 4: Khoảng cách giữa các đơn
print("Đang vẽ biểu đồ khoảng cách...")
df_distance = df_orders_clean.filter(col("is_first_order") == 0).select("days_since_prior_order").toPandas()
plt.figure(figsize=(10,5))
sns.histplot(df_distance['days_since_prior_order'], bins=30, color='orange')
plt.title('Phân bố khoảng cách giữa các đơn hàng (Đỉnh 7 & 30 ngày)')
plt.savefig('bieu_do_khoang_cach.png')
plt.close()

# BIỂU ĐỒ 5: Cỡ giỏ trung bình
print("Đang vẽ biểu đồ cỡ giỏ...")
df_basket = df_prior_clean.groupBy("order_id").count().toPandas()
plt.figure(figsize=(10,5))
sns.histplot(df_basket['count'], bins=50, color='purple')
plt.title('Phân bố số lượng sản phẩm trong một đơn hàng (Cỡ giỏ)')
plt.xlim(0, 50) 
plt.savefig('bieu_do_co_gio.png')
plt.close()

# BIỂU ĐỒ 6: Tỷ lệ mua lại theo Department
print("Đang vẽ biểu đồ tỷ lệ mua lại...")
df_reorder = df_prior_clean.join(df_products, on="product_id") \
                           .join(df_departments, on="department_id")
df_dept_reorder = df_reorder.groupBy("department") \
    .agg((sum("reordered") / count("reordered")).alias("reorder_rate")) \
    .orderBy(col("reorder_rate").desc()).limit(10).toPandas()

plt.figure(figsize=(12,6))
sns.barplot(data=df_dept_reorder, x='reorder_rate', y='department', palette='viridis')
plt.title('Top 10 Ngành hàng có tỷ lệ mua lại cao nhất')
plt.savefig('bieu_do_ty_le_mua_lai.png')
plt.close()

# 6. K-MEANS PHÂN KHÚC KHÁCH HÀNG THEO TẦN SUẤT MUA
print("Đang chạy phân khúc khách hàng (K-Means)")
df_user_freq = df_orders_clean.groupBy("user_id").count().withColumnRenamed("count", "frequency")
assembler = VectorAssembler(inputCols=["frequency"], outputCol="features")
df_rfm = assembler.transform(df_user_freq)

kmeans = KMeans(k=4, seed=42)
model = kmeans.fit(df_rfm)
df_segments = model.transform(df_rfm)

# Lưu bảng Gold lên HDFS
df_segments.select("user_id", "frequency", "prediction").write.mode("overwrite").parquet(f"{base_hdfs}/gold/gold_user_segments_parquet")
print("Đã phân khúc khách hàng thành 4 nhóm")
print("ĐÃ CHẠY XONG TOÀN BỘ QUY TRÌNH TUẦN 2")
spark.stop()