import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# 1. CẤU HÌNH ĐƯỜNG DẪN (Bắt buộc cho Windows)
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'  # Sửa lại nếu máy em dùng JDK khác
os.environ['PATH'] = os.environ['PATH'] + ';C:\\hadoop\\bin'

# 2. KHỞI TẠO SPARK
print("--- Đang khởi tạo Spark ---")
spark = SparkSession.builder \
    .appName("Instacart_TV2_Final") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 3. ĐỌC DỮ LIỆU (TV1 sẽ sửa đường dẫn này thành HDFS khi chạy full data)

# TẠM THỜI DÙNG ĐƯỜNG DẪN LOCAL (5% data) ĐỂ TEST
# TV1 ĐỔI THÀNH: "hdfs://IP_CUA_TV1:9000/duong_dan_tren_HDFS/orders.csv"
# df_orders = spark.read.csv("../data_sample/orders.csv", header=True, inferSchema=True)
# df_prior = spark.read.csv("../data_sample/order_products__prior.csv", header=True, inferSchema=True)
# df_products = spark.read.csv("../data_sample/products.csv", header=True, inferSchema=True)
# df_departments = spark.read.csv("../data_sample/departments.csv", header=True, inferSchema=True)
print("--- Đang đọc dữ liệu từ HDFS (Parquet) ---")
# Đọc dữ liệu từ thư mục HDFS dạng Parquet (Bỏ header và inferSchema vì Parquet đã lưu sẵn)
df_orders = spark.read.parquet("hdfs://192.168.1.36:9000/user/student/instacart/bronze/orders/")
df_prior = spark.read.parquet("hdfs://192.168.1.36:9000/user/student/instacart/bronze/order_products__prior/")
df_products = spark.read.parquet("hdfs://192.168.1.36:9000/user/student/instacart/bronze/products/")
df_departments = spark.read.parquet("hdfs://192.168.1.36:9000/user/student/instacart/bronze/departments/")

# 4. LÀM SẠCH DỮ LIỆU (TẠO TẦNG SILVER)

print("--- Đang làm sạch dữ liệu ---")
df_orders_clean = df_orders.withColumn(
    "is_first_order", 
    when(col("days_since_prior_order").isNull(), 1).otherwise(0)
).dropna(subset=["user_id", "order_id"])

df_prior_clean = df_prior.dropna(subset=["order_id", "product_id"]).dropDuplicates()

# Lưu Parquet (Chỉ lưu khi chạy test, TV1 chạy full sẽ lưu ra HDFS)
df_orders_clean.write.mode("overwrite").parquet("silver_orders_parquet")
df_prior_clean.write.mode("overwrite").parquet("silver_order_products_parquet")
print("-> Đã lưu 2 bảng Parquet (Silver) thành công!")

# 5. VẼ 6 BIỂU ĐỒ EDA

print("--- Đang vẽ 6 biểu đồ EDA ---")
sns.set_theme(style="whitegrid")

# BIỂU ĐỒ 1: Đơn hàng theo giờ
print("1/6: Đang vẽ biểu đồ giờ...")
df_hourly = df_orders_clean.groupBy("order_hour_of_day").count().orderBy("order_hour_of_day").toPandas()
plt.figure(figsize=(10,5))
sns.barplot(data=df_hourly, x='order_hour_of_day', y='count', color='#4C72B0')
plt.title('Phân bố đơn hàng theo giờ trong ngày')
plt.savefig('bieu_do_gio.png')
plt.close()

# BIỂU ĐỒ 2: Đơn hàng theo ngày trong tuần
print("2/6: Đang vẽ biểu đồ ngày...")
df_daily = df_orders_clean.groupBy("order_dow").count().orderBy("order_dow").toPandas()
plt.figure(figsize=(10,5))
sns.barplot(data=df_daily, x='order_dow', y='count', color='#55A868')
plt.title('Phân bố đơn hàng theo ngày trong tuần (0=Chủ Nhật)')
plt.savefig('bieu_do_ngay.png')
plt.close()

# BIỂU ĐỒ 3: Top 10 sản phẩm bán chạy
print("3/6: Đang vẽ biểu đồ Top sản phẩm...")
df_top = df_prior_clean.groupBy("product_id").count().orderBy(col("count").desc()).limit(10)
df_top_named = df_top.join(df_products, on="product_id").select("product_name", "count").toPandas()
plt.figure(figsize=(12,6))
sns.barplot(data=df_top_named, x='count', y='product_name', palette='viridis', hue='product_name', legend=False)
plt.title('Top 10 sản phẩm bán chạy nhất')
plt.savefig('bieu_do_top_san_pham.png')
plt.close()

# BIỂU ĐỒ 4: Khoảng cách giữa các đơn
print("4/6: Đang vẽ biểu đồ khoảng cách...")
df_distance = df_orders_clean.filter(col("is_first_order") == 0).select("days_since_prior_order").toPandas()
plt.figure(figsize=(10,5))
sns.histplot(df_distance['days_since_prior_order'], bins=30, color='orange')
plt.title('Phân bố khoảng cách giữa các đơn hàng')
plt.savefig('bieu_do_khoang_cach.png')
plt.close()

# BIỂU ĐỒ 5: Cỡ giỏ trung bình
print("5/6: Đang vẽ biểu đồ cỡ giỏ...")
df_basket = df_prior_clean.groupBy("order_id").count().toPandas()
plt.figure(figsize=(10,5))
sns.histplot(df_basket['count'], bins=50, color='purple')
plt.title('Phân bố số lượng sản phẩm trong một đơn hàng')
plt.xlim(0, 50) 
plt.savefig('bieu_do_co_gio.png')
plt.close()

# BIỂU ĐỒ 6: Tỷ lệ mua lại theo Department
print("6/6: Đang vẽ biểu đồ tỷ lệ mua lại...")
df_reorder = df_prior_clean.join(df_products, on="product_id").join(df_departments, on="department_id")
df_dept_reorder = df_reorder.groupBy("department") \
    .agg((sum("reordered") / count("reordered")).alias("reorder_rate")) \
    .orderBy(col("reorder_rate").desc()).limit(10).toPandas()
plt.figure(figsize=(12,6))
sns.barplot(data=df_dept_reorder, x='reorder_rate', y='department', palette='viridis', hue='department', legend=False)
plt.title('Top 10 Ngành hàng có tỷ lệ mua lại cao nhất')
plt.savefig('bieu_do_ty_le_mua_lai.png')
plt.close()
print("-> Đã vẽ và lưu xong 6 biểu đồ!")

# 6. K-MEANS PHÂN KHÚC KHÁCH HÀNG

print("--- Đang chạy phân khúc khách hàng (K-Means) ---")
df_user_freq = df_orders_clean.groupBy("user_id").count().withColumnRenamed("count", "frequency")
assembler = VectorAssembler(inputCols=["frequency"], outputCol="features")
df_rfm = assembler.transform(df_user_freq)

kmeans = KMeans(k=4, seed=42)
model = kmeans.fit(df_rfm)
df_segments = model.transform(df_rfm)
df_segments.select("user_id", "frequency", "prediction").write.mode("overwrite").parquet("gold_user_segments_parquet")
print("-> Đã phân khúc khách hàng thành 4 nhóm!")

# 7. XUẤT DỮ LIỆU CHO WEB APP (TV5)

print("--- Đang xuất dữ liệu cho Web App ---")
# Dùng .coalesce(1) để gom kết quả về 1 file duy nhất cho TV5 dễ dùng
df_hourly.to_csv("web_data_hourly.csv", index=False)
df_daily.to_csv("web_data_daily.csv", index=False)

# Top 20 sản phẩm cho Web
df_top_web = df_prior_clean.groupBy("product_id").count().orderBy(col("count").desc()).limit(20)
df_top_web_named = df_top_web.join(df_products, on="product_id").select("product_name", "count").toPandas()
df_top_web_named.to_csv("web_data_top_products.csv", index=False)

# Phân khúc cho Web
df_segments.select("user_id", "frequency", "prediction").toPandas().to_csv("web_data_user_segments.csv", index=False)

print("--- Đã xuất xong 4 file CSV cho Web App! ---")
print("=== ĐÃ CHẠY XONG TOÀN BỘ QUY TRÌNH TUẦN 2 ===")
spark.stop()