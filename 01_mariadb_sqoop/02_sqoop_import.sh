#!/bin/bash

echo "Bat dau import du lieu tu MariaDB len HDFS bang Sqoop..."

# 1. Import bang aisles (Bang nho -> dung 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table aisles \
  --target-dir /user/hadoop/bronze/aisles \
  --as-parquetfile \
  -m 1

# 2. Import bang departments (Bang nho -> dung 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table departments \
  --target-dir /user/hadoop/bronze/departments \
  --as-parquetfile \
  -m 1

# 3. Import bang products (Bang vua -> dung 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table products \
  --target-dir /user/hadoop/bronze/products \
  --as-parquetfile \
  -m 1

# 4. Import bang orders (Bang lon -> dung 4 mapper chia nho theo order_id)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table orders \
  --target-dir /user/hadoop/bronze/orders \
  --as-parquetfile \
  -m 4 \
  --split-by order_id

# 5. Import bang order_products__prior (Rat lon -> dung 4 mapper chia nho theo order_id)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table order_products__prior \
  --target-dir /user/hadoop/bronze/order_products__prior \
  --as-parquetfile \
  -m 4 \
  --split-by order_id

echo "Hoan thanh import 5 bang du lieu!"