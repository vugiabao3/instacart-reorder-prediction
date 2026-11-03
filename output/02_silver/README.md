# TẦNG SILVER - THÀNH VIÊN 2
- Dữ liệu đã được làm sạch bằng PySpark trên toàn bộ 32 triệu dòng.
- Do file gốc `silver_orders.csv` và `silver_order_products.csv` chứa 32 triệu dòng, dung lượng > 1GB vượt quá giới hạn 100MB của GitHub, nên nhóm không thể up trực tiếp lên đây.
- Thay vào đó xin phép nhóm trưởng nộp các file `.csv` chứa **kết quả tổng hợp (EDA)** được trích xuất trực tiếp từ tầng Silver và Gold để minh chứng cho quá trình xử lý:
  1. `web_data_hourly.csv`: Thống kê đơn hàng theo giờ (từ Silver).
  2. `web_data_daily.csv`: Thống kê đơn hàng theo ngày (từ Silver).
  3. `web_data_top_products.csv`: Top sản phẩm bán chạy (từ Silver).
  4. `web_data_user_segments.csv`: Kết quả phân khúc khách hàng (từ Gold).