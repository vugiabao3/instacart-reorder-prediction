### Cấu trúc bảng đầu ra `gold_reorder_predictions.csv` (Bàn giao cho Thành viên 5)
Bảng gồm 4 cột chuẩn theo Hợp đồng dữ liệu và 4 cột bổ sung giúp trực quan hóa trên Dashboard:
- `user_id`: ID khách hàng.
- `product_id`: ID sản phẩm ứng viên (khách đã từng mua ở các đơn prior).
- `product_name`: Tên sản phẩm (đã join từ `products.csv` để tiện hiển thị lên Dashboard).
- `probability`: Xác suất mua lại ở đơn tiếp theo (dùng làm `score` cho nguồn `reorder`).
- `predicted`: Nhãn dự đoán sau khi tối ưu ngưỡng (`1` = Mua lại, `0` = Không mua lại).
- `rank_in_user`: Thứ tự ưu tiên gợi ý của sản phẩm đối với từng `user_id` (sắp xếp theo `probability` giảm dần).
- `past_orders_count`: Số lần khách đã mua sản phẩm này trong quá khứ.
- `trang_thai_goi_y`: Nhãn mô tả tiếng Việt (`Nên gợi ý mua lại` / `Không gợi ý`).