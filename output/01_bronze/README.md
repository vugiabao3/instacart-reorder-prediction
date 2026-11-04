# TẦNG BRONZE - THÀNH VIÊN 1

* Dữ liệu thô (raw data) đã được trích xuất thành công từ cơ sở dữ liệu MariaDB đẩy lên hệ thống phân tán Hadoop HDFS thông qua Sqoop.
* Tương tự như tầng Silver, do tổng dung lượng các file gốc quá lớn vượt giới hạn 100MB của GitHub, tôi nộp 5% dữ liệu sample (trích xuất bằng lệnh `head`) theo đúng yêu cầu của nhóm trưởng.
* Các file đính kèm dùng để nhóm trưởng đối chiếu (Data Lineage) bao gồm:
  1. `orders_bronze_sample.csv`
  2. `products_bronze_sample.csv`
  3. `departments_bronze_sample.csv`
  4. `aisles_bronze_sample.csv`