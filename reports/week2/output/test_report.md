# Báo cáo kiểm thử — Member 5 Tuần 2

## Tổng quan

Báo cáo này tổng hợp kết quả kiểm thử tự động cho engines gợi ý
và bộ tải dữ liệu trong dashboard Instacart Reorder Prediction.

## Kết quả pytest

Tất cả 13 bài kiểm thử đều pass:

| # | Bài kiểm thử | Mô tả | Kết quả |
|---|-------------|-------|---------|
| 1 | test_load_rules_from_real_file | Tải file luật FP-Growth thật | PASS |
| 2 | test_load_predictions_from_real_file | Tải file dự đoán ML thật | PASS |
| 3 | test_engine_empty_data_returns_empty | Engine với dữ liệu trống trả về rỗng | PASS |
| 4 | test_ml_recommendations_existing_user | Gợi ý ML cho user có sẵn, sắp xếp theo xác suất | PASS |
| 5 | test_ml_recommendations_nonexistent_user | Gợi ý ML cho user không tồn tại trả về rỗng | PASS |
| 6 | test_fpgrowth_recommendations_matching_aisles | Gợi ý FP-Growth khi user có gian hàng khớp luật | PASS |
| 7 | test_fpgrowth_recommendations_no_match | Gợi ý FP-Growth khi không luật nào khớp | PASS |
| 8 | test_fpgrowth_excludes_purchased_products | FP-Growth không trả về sản phẩm đã mua | PASS |
| 9 | test_combined_ml_priority_dedup | Gợi ý kết hợp, ưu tiên ML khi trùng sản phẩm | PASS |
| 10 | test_combined_top_n_limit | Gợi ý kết hợp giới hạn theo top_n | PASS |
| 11 | test_user_summary | Tóm tắt người dùng có cấu trúc và giá trị đúng | PASS |
| 12 | test_user_summary_nonexistent_user | Tóm tắt user không tồn tại có số liệu bằng 0 | PASS |
| 13 | test_missing_file_returns_empty | Tải file không tồn tại trả về DataFrame rỗng | PASS |

## Nguyên tắc dữ liệu

- Không tự tạo probabilitas, lift hoặc kết quả.
- Tất cả số liệu từ file thực của các thành viên nhóm:
  - `03_spark_baskets/gold_rules_aisle.csv` (Member 3, FP-Growth)
  - `04_spark_ml/gold_reorder_predictions.csv` (Member 4, MLlib)
  - `silver_orders.csv`, `silver_order_products.csv` (Member 2, Silver)
  - `data_sample/products.csv`, `data_sample/aisles.csv` (tham chiếu)
- File dữ liệu kiểm thử trong `reports/week2/test_data/` được gắn nhãn rõ là TEST.

## Cách chạy kiểm thử

```bash
pip install pytest
cd C:\Users\HI TCC\Downloads\instacart-reorder-prediction
$env:PYTHONPATH="05_dashboard_gold"
python -m pytest tests/test_recommendation_engine.py -v
```

## Cách chạy dashboard

```bash
python -m streamlit run 05_dashboard_gold/app.py
```

Tab mới: "Tra cứu người dùng" và "Báo cáo kiểm thử".
