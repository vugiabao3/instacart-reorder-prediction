#!/bin/bash
echo "====================================================="
echo "   BẮT ĐẦU CHẠY TỰ ĐỘNG PIPELINE BIG DATA INSTACART  "
echo "====================================================="

# BƯỚC 1: Dọn dẹp HDFS cũ (tránh lỗi FileAlreadyExists)
echo -e "\n[1/3] Đang dọn dẹp thư mục đích trên HDFS..."
hdfs dfs -rm -r -f /user/student/instacart/order_products__prior

# BƯỚC 2: Chạy Sqoop Import
echo -e "\n[2/3] Đang chạy Sqoop nạp 32 triệu dòng từ MariaDB lên HDFS..."
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root \
  --password 'student' \
  --table order_products__prior \
  --m 1 \
  --target-dir /user/student/instacart/order_products__prior

# BƯỚC 3: Chạy Benchmark
echo -e "\n[3/3] Đang chạy đánh giá hiệu năng Pandas vs PySpark..."
# Lưu ý: Đảm bảo đường dẫn file dữ liệu trong script python đã trỏ đúng vào C:\instacart_data\... hoặc /home/student/...
python benchmark_pandas_vs_spark.py

echo -e "\n====================================================="
echo "         PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG!            "
echo "====================================================="