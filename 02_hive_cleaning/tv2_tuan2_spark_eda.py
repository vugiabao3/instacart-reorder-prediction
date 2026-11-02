import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, sum
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# 1. CẤU HÌNH ĐƯỜNG DẪN (Bắt buộc cho Windows)
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-17'
os.environ['PATH'] = os.environ['PATH'] + ';C:\\hadoop\\bin'

# 2. KHỞI TẠO SPARK
print("Đang khởi tạo Spark ")
spark = SparkSession.builder \
    .appName("Instacart_TV2_Week2") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 3. ĐỌC DỮ LIỆU MẪU 5% TỪ THƯ MỤC DATA_SAMPLE
# (Dùng ../ để lùi ra ngoài 1 cấp từ 02_hive_cleaning)
print("Đang đọc dữ liệu")
df_orders = spark.read.csv("../data_sample/orders.csv", header=True, inferSchema=True)
df_prior = spark.read.csv("../data_sample/order_products__prior.csv", header=True, inferSchema=True)
df_products = spark.read.csv("../data_sample/products.csv", header=True, inferSchema=True)
df_departments = spark.read.csv("../data_sample/departments.csv", header=True, inferSchema=True)

# 4. LÀM SẠCH DỮ LIỆU (TẠO TẦNG SILVER)
print("Đang làm sạch dữ liệu")
# Xử lý bảng Orders
df_orders_clean = df_orders.withColumn(
    "is_first_order", 
    when(col("days_since_prior_order").isNull(), 1).otherwise(0)
).dropna(subset=["user_id", "order_id"])

# Xử lý bảng Prior
df_prior_clean = df_prior.dropna(subset=["order_id", "product_id"]).dropDuplicates()

# Lưu Parquet
df_orders_clean.write.mode("overwrite").parquet("silver_orders_parquet")
df_prior_clean.write.mode("overwrite").parquet("silver_order_products_parquet")
print("-> Đã lưu 2 bảng Parquet (Silver) thành công!")

# 5. VẼ CÁC BIỂU ĐỒ EDA NÂNG CAO
print("Đang vẽ biểu đồ")
sns.set_theme(style="whitegrid")

# BIỂU ĐỒ 4: Khoảng cách giữa các đơn (Days since prior order)
print("Đang vẽ biểu đồ khoảng cách...")
df_distance = df_orders_clean.filter(col("is_first_order") == 0).select("days_since_prior_order").toPandas()
plt.figure(figsize=(10,5))
sns.histplot(df_distance['days_since_prior_order'], bins=30, color='orange')
plt.title('Phân bố khoảng cách giữa các đơn hàng (Đỉnh 7 & 30 ngày)')
plt.savefig('bieu_do_khoang_cach.png')
plt.close()

# BIỂU ĐỒ 5: Cỡ giỏ trung bình (Số lượng sản phẩm/đơn)
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
# Join prior với products và departments
df_reorder = df_prior_clean.join(df_products, on="product_id") \
                           .join(df_departments, on="department_id")
# Tính tỷ lệ mua lại (reordered = 1)
df_dept_reorder = df_reorder.groupBy("department") \
    .agg((sum("reordered") / count("reordered")).alias("reorder_rate")) \
    .orderBy(col("reorder_rate").desc()).limit(10).toPandas()

plt.figure(figsize=(12,6))
sns.barplot(data=df_dept_reorder, x='reorder_rate', y='department', palette='viridis')
plt.title('Top 10 Ngành hàng có tỷ lệ mua lại cao nhất')
plt.savefig('bieu_do_ty_le_mua_lai.png')
plt.close()

# 6. (TÙY CHỌN) K-MEANS PHÂN KHÚC KHÁCH HÀNG THEO TẦN SUẤT MUA
print("Đang chạy phân khúc khách hàng (K-Means)")
# Tính Frequency (Số đơn hàng của mỗi user)
df_user_freq = df_orders_clean.groupBy("user_id").count().withColumnRenamed("count", "frequency")

# Đưa dữ liệu vào Vector
assembler = VectorAssembler(inputCols=["frequency"], outputCol="features")
df_rfm = assembler.transform(df_user_freq)

# Chạy K-Means chia làm 4 nhóm
kmeans = KMeans(k=4, seed=42)
model = kmeans.fit(df_rfm)
df_segments = model.transform(df_rfm)

# Lưu bảng Gold
df_segments.select("user_id", "frequency", "prediction").write.mode("overwrite").parquet("gold_user_segments_parquet")
print("Đã phân khúc khách hàng thành 4 nhóm")

print("ĐÃ CHẠY XONG TOÀN BỘ QUY TRÌNH TUẦN 2")
spark.stop()