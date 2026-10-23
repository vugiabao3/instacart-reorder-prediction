# BÁO CÁO TUẦN 1 - THÀNH VIÊN 2 (SILVER & EDA)

## 1. Công việc đã hoàn thành
- Đã nhận dữ liệu mẫu 5% (orders, order_products__prior, products).
- Đã viết code làm sạch dữ liệu (Python .py): Xử lý NULL ở cột `days_since_prior_order` (thêm cờ `is_first_order`), loại bỏ trùng lặp.
- Xuất ra 2 bảng dữ liệu sạch: `silver_orders.csv` và `silver_order_products.csv`.
- Đã vẽ 3 biểu đồ EDA đầu tiên trên tập mẫu.

## 2. Nhận xét biểu đồ
1. **Biểu đồ giờ:** Khách hàng mua sắm cao điểm từ 9h sáng đến 4h chiều. Đỉnh cao nhất là 10h sáng.
2. **Biểu đồ ngày:** Ngày Chủ nhật (0) và Thứ hai (1) có lượng đơn hàng cao nhất trong tuần.
3. **Top sản phẩm:** Các sản phẩm trái cây tươi (chuối, dâu tây) và sữa là những mặt hàng được mua nhiều nhất.

## 3. Hợp đồng dữ liệu (Data Contract) bàn giao cho TV3 & TV4
- Bảng `silver_orders` gồm các cột: order_id, user_id, order_dow, order_hour_of_day, days_since_prior_order, is_first_order, eval_set.
- Bảng `silver_order_products` gồm các cột: order_id, product_id, add_to_cart_order, reordered.

## 4. Kế hoạch Tuần 2
- Chuyển đổi code Pandas sang Spark SQL để xử lý toàn bộ 32 triệu dòng trên cụm Hadoop của TV1.
- Vẽ thêm 3 biểu đồ còn lại (Phân bố khoảng cách, Cỡ giỏ trung bình, Tỷ lệ mua lại).
