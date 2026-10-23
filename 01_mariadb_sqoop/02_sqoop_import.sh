#!/bin/bash

echo "Bat dau import du lieu tu MariaDB len HDFS bang Sqoop..."

# 1. Import bảng aisles (Bảng nhỏ -> dùng 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table aisles \
  --target-dir /user/student/instacart/bronze/aisles \
  --as-parquetfile \
  -m 1

# 2. Import bảng departments (Bảng nhỏ -> dùng 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table departments \
  --target-dir /user/student/instacart/bronze/departments \
  --as-parquetfile \
  -m 1

# 3. Import bảng products (Bảng vừa -> dùng 1 mapper)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table products \
  --target-dir /user/student/instacart/bronze/products \
  --as-parquetfile \
  -m 1

# 4. Import bảng orders (Bảng lớn -> dùng 4 mapper chia nhỏ theo order_id)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table orders \
  --target-dir /user/student/instacart/bronze/orders \
  --as-parquetfile \
  -m 4 \
  --split-by order_id

# 5. Import bảng order_products__prior (Rất lớn -> dùng 4 mapper chia nhỏ theo order_id)
sqoop import \
  --connect jdbc:mysql://localhost:3306/instacart \
  --username root --password 'student' \
  --table order_products__prior \
  --target-dir /user/student/instacart/bronze/order_products__prior \
  --as-parquetfile \
  -m 4 \
  --split-by order_id

echo "Hoan thanh import 5 bang du lieu!"