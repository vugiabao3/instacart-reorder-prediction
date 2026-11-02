import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import count

# Tạm thời trỏ vào file mẫu 5% có sẵn trong nhánh để test code
# Khi nào chạy thật trên 32 triệu dòng, bạn sẽ đổi lại đường dẫn này sau
file_path = r"C:\instacart_data\order_products__prior.csv"
print("=== 1. TEST PANDAS (XỬ LÝ 1 MÁY) ===")
start_pandas = time.time()
pdf = pd.read_csv(file_path)
pandas_result = pdf.groupby('product_id').size()
time_pandas = time.time() - start_pandas
print(f"Thời gian Pandas hoàn thành: {time_pandas:.4f} giây\n")

print("=== 2. TEST PYSPARK (XỬ LÝ PHÂN TÁN) ===")
spark = SparkSession.builder.appName("Benchmark_Instacart").getOrCreate()
start_spark = time.time()
sdf = spark.read.csv(file_path, header=True, inferSchema=True)
spark_result = sdf.groupBy("product_id").agg(count("*"))
spark_result.count() # Lệnh action bắt buộc để ép Spark thực thi
time_spark = time.time() - start_spark
print(f"Thời gian PySpark hoàn thành: {time_spark:.4f} giây")

spark.stop()