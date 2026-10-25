# Báo cáo Tiến độ Tuần 1 - Thành viên 4 (04_spark_ml)

## 1. Đồng bộ dữ liệu & Đối chiếu số dòng (Data Lineage)
- **Nguồn dữ liệu đầu vào:** Đọc trực tiếp từ tầng Silver của Thành viên 2 (`02_hive_cleaning/silver_orders.csv` và `silver_order_products.csv`) kết hợp gán nhãn từ tập `train`.
- **Tổng số cặp ứng viên `(user_id, product_id)` trong `gold_features`:** `669,140` dòng (không trùng lặp, không rò rỉ dữ liệu).
  - Nhãn `label = 0` (Không mua lại): `604,223` dòng (`90.30%`).
  - Nhãn `label = 1` (Mua lại): `64,917` dòng (`9.70%`).

## 2. Kết quả mô hình Baseline Logistic Regression (Mẫu 5%)
- **Chia tập dữ liệu:** 80% User Train / 20% User Test (`seed = 42`).
- **Bộ chỉ số đánh giá trên tập Test:**
  - **AUC-ROC:** `0.8145`
  - **PR-AUC:** `0.3748`
  - **Ngưỡng quyết định tối ưu (Best Threshold):** `0.65`
  - **F1-Score (tại Best Threshold):** `0.8564`
  - **Precision@10:** `0.3053`
  - **Recall@10:** `0.5478`

## 3. Hợp đồng dữ liệu bàn giao cho Thành viên 5 (`gold_reorder_predictions.csv`)
Đã xuất bản Parquet đầy đủ và bản CSV (`2,000` dòng tiêu biểu) tại `04_spark_ml/gold_reorder_predictions.csv` và `data_sample/gold_reorder_predictions.csv` gồm 8 cột:
- `user_id`, `product_id`, `probability`, `predicted`: 4 cột chuẩn theo Hợp đồng dữ liệu.
- `product_name`, `rank_in_user`, `past_orders_count`, `trang_thai_goi_y`: 4 cột trực quan hỗ trợ hiển thị trên GitHub và Dashboard.
