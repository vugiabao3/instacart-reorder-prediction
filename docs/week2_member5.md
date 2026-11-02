# Tài liệu tích hợp — Member 5 Tuần 2

## Mục đích

Tài liệu này mô tả cách tích hợp engines gợi ý (ML + FP-Growth) vào
dashboard Instacart Reorder Prediction (thành viên 5, tuần 2), bao gồm
cấu hình, tải dữ liệu, logic gợi ý, tab mới trên dashboard và kiểm thử
tự động.

## Cấu trúc file

```
05_dashboard_gold/
├── app.py                 # Dashboard Streamlit (2 tab mới)
├── config.py              # Đường dẫn file và cấu hình engine
├── data_loader.py         # Tải dữ liệu với validation và caching
├── recommendation_engine.py  # Engine gợi ý ML + FP-Growth
├── reconciliation.py      # Giữ nguyên từ tuần 1 (không thay đổi)
└── ...

reports/
├── week1/                 # Dữ liệu tuần 1 (giữ nguyên)
├── week2/
│   ├── input/             # (được réserve)
│   ├── output/
│   │   └── test_report.md # Báo cáo kiểm thử
│   └── test_data/         # Dữ liệu kiểm thử (gắn nhãn TEST)
│       ├── test_aisles.csv
│       ├── test_products.csv
│       ├── test_orders.csv
│       ├── test_order_products.csv
│       ├── test_rules.csv
│       └── test_predictions.csv
tests/
└── test_recommendation_engine.py  # 13 bài kiểm thử pytest
docs/
└── week2_member5.md       # Tài liệu này (file hiện tại)
```

## 1. config.py — Cấu hình và đường dẫn

Định nghĩa tất cả đường dẫn đến file dữ liệu của các thành viên
nhóm và các tham số engine:

- **RULES_FILE** → `03_spark_baskets/gold_rules_aisle.csv` (Member 3)
- **PREDICTIONS_FILE** → `04_spark_ml/gold_reorder_predictions.csv` (Member 4)
- **ORDERS_FILE** → `silver_orders.csv` (Member 2)
- **ORDER_PRODUCTS_FILE** → `silver_order_products.csv` (Member 2)
- **PRODUCTS_FILE** → `data_sample/products.csv`
- **AISLES_FILE** → `data_sample/aisles.csv`

Tham số engine:
- `DEFAULT_TOP_N_ML = 10`
- `DEFAULT_TOP_N_FPGROWTH = 10`
- `DEFAULT_TOP_N_COMBINED = 20`
- `MIN_CONFIDENCE = 0.0`
- `MIN_LIFT = 1.0`

## 2. data_loader.py — Tải dữ liệu

Mỗi hàm tải một file CSV với:
- **Validation:** kiểm tra các cột bắt buộc, ném ValueError nếu thiếu.
- **Caching:** dùng `@st.cache_data` để không tải lại nhiều lần.
- **Missing file:** trả về DataFrame rỗng nếu file không tồn tại, không làm crash dashboard.
- **Ép kiểu:** numeric columns sang int64/float64, drop NA.

Hàm tiện ích:
- `get_user_purchased_aisles(user_id, ...)` → set các aisle_id user đã mua.
- `get_user_purchased_products(user_id, ...)` → set các product_id user đã mua.
- `load_all_data()` → dict tất cả 6 dataframe.

## 3. recommendation_engine.py — Gợi ý ML + FP-Growth

Lớp `RecommendationEngine` nhận 6 dataframe (hoặc tự load từ file qua
`data_dir`).

### ML recommendations (`ml_recommendations`)
Lọc `gold_reorder_predictions.csv` theo `user_id`, sắp xếp theo
`probability` giảm dần, trả về `top_n` sản phẩm đầu tiên.

### FP-Growth recommendations (`fpgrowth_recommendations`)
1. Tìm các gian hàng user đã mua (từ đơn hàng + order_products +
   products).
2. Lọc luật FP-Growth có `confidence >= MIN_CONFIDENCE` và
   `lift >= MIN_LIFT`.
3. Giữ luật mà `antecedent_set` là tập con của gian hàng user đã mua.
4. Lấy `consequent_ids` (gian hàng được gợi ý).
5. Tìm sản phẩm trong các gian hàng đó, loại trừ sản phẩm user đã mua.
6. Điểm mỗi sản phẩm bằng lift cao nhất của luật liên quan.
7. Trả về `top_n` sản phẩm có điểm cao nhất.

### Combined recommendations (`combined_recommendations`)
- Nối ML (trước) + FP-Growth (sau).
- Đánh trùng theo `product_id`, giữ bản ML (ưu tiên ML).
- Sắp xếp: ML trước (theo probability), FP-Growth sau (theo lift).
- Giới hạn `top_n`.

### User summary (`user_summary`)
Trả về dict: tổng đơn hàng, số sản phẩm đã mua, số gian hàng duy nhất,
tên gian hàng, có dự đoán ML hay không, số dự đoán ML.

## 4. app.py — Dashboard (2 tab mới)

### Tab 6: Tra cứu người dùng
- Dropdown chọn `user_id` từ danh sách có sẵn trong ML predictions.
- Hiển thị tóm tắt: tổng đơn hàng, sản phẩm đã mua, gian hàng, dự đoán ML.
- Hiển thị gợi ý kết hợp (ML + FP-Growth) với bảng và nút tải CSV.
- Hiển thị gian hàng user đã mua.

### Tab 7: Báo cáo kiểm thử
- Hiển thị `reports/week2/output/test_report.md` nếu có.
- Bảng cấu hình dữ liệu: tên file, có dữ liệu hay không, số dòng.
- Danh sách user_id có sẵn để tra cứu.

Các tab tuần 1 (Tổng quan, Đối chiếu, Chất lượng, FP-Growth, Dự đoán
mua lại) được giữ nguyên không thay đổi.

## 5. Kiểm thử pytest

13 bài kiểm thử trong `tests/test_recommendation_engine.py`:

| Nhóm | Bài kiểm thử |
|------|-------------|
| Tải dữ liệu | load rules thật, load predictions thật, missing file |
| Engine rỗng | tất cả DataFrame rỗng → kết quả rỗng |
| ML | user có sẵn (sắp xếp đúng), user không tồn tại |
| FP-Growth | có luật khớp, không luật khớp, loại sản phẩm đã mua |
| Kết hợp | ưu tiên ML khi trùng, giới hạn top_n |
| User summary | cấu trúc đúng, user không tồn tại |

Chạy:
```bash
pip install pytest
$env:PYTHONPATH="05_dashboard_gold"
python -m pytest tests/test_recommendation_engine.py -v
```

## 6. Nguyên tắc dữ liệu (quan trọng)

- **Không tự tạo** probabilitas, lift hoặc kết quả. Tất cả điểm số từ
  file thật của các thành viên.
- File trong `reports/week2/test_data/` là dữ liệu giả để kiểm thử,
  gắn nhãn rõ là TEST, không dùng trong production dashboard.
- Không sửa file core của các thành viên khác (`01_`, `02_`, `03_`,
  `04_`).
- `reconciliation.py` từ tuần 1 được giữ nguyên.

## 7. Cách chạy dashboard

```bash
python -m streamlit run 05_dashboard_gold/app.py
```

Mặc định chạy tại `http://localhost:8501`.
