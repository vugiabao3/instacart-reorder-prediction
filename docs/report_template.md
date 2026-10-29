# BÁO CÁO ĐỒ ÁN BIG DATA: PHÂN TÍCH GIỎ HÀNG VÀ DỰ ĐOÁN MUA LẠI TRÊN 3 TRIỆU ĐƠN HÀNG TẠP HÓA TRỰC TUYẾN (INSTACART)

* **Môn học:** Xử lý Dữ liệu lớn (Big Data Processing)
* **Đề tài số:** 09
* **Tên dự án:** Smart Reorder & Next-Basket Recommender System
* **Nhóm thực hiện:** Nhóm 5

---

## 👨‍💻 BẢNG PHÂN CÔNG CÔNG VIỆC VÀ TIẾN ĐỘ THÀNH VIÊN

| Thành viên | Vai trò | Trách nhiệm chính trong dự án | Trạng thái đóng góp Báo cáo |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | Nền tảng & Ingestion | Dựng cụm Big Data, MariaDB, Sqoop data ingestion, script tự động hóa pipeline | Đã hoàn thành Chương 2.1 |
| **Thành viên 2** | Silver, EDA & Phân khúc | Tầng Silver, làm sạch dữ liệu, phân tích hành vi (EDA), K-Means RFM | Đã hoàn thành Chương 2.2 & 2.3 |
| **Thành viên 3** | Khai phá giỏ hàng | Khai phá tập phổ biến & luật kết hợp (FP-Growth), Word2Vec (item2vec) | Đã hoàn thành Chương 3 |
| **Thành viên 4** | Dự đoán mua lại | Xây dựng đặc trưng (Features), mô hình ML (Logistic Regression, GBT) | Đã hoàn thành Chương 4 |
| **Thành viên 5** | Leader, Integration & BI | Data Lineage & đối chiếu số dòng, tích hợp gợi ý, Dashboard (Streamlit), tối ưu Spark | Đã hoàn thành Chương 1, 5, 6 |

---

## 📖 MỤC LỤC BÁO CÁO

1. **CHƯƠNG 1: TỔNG QUAN DỰ ÁN VÀ KIẾN TRÚC HỆ THỐNG**
   - 1.1. Bối cảnh bài toán và ý nghĩa kinh doanh (32 triệu dòng giao dịch)
   - 1.2. Giải pháp kiến trúc Medallion (Bronze - Silver - Gold)
   - 1.3. Lựa chọn và giải thích Tech Stack
2. **CHƯƠNG 2: LƯU TRỮ, LÀM SẠCH VÀ PHÂN TÍCH KHÁM PHÁ (BRONZE & SILVER)**
   - 2.1. Nạp dữ liệu từ MariaDB lên HDFS bằng Sqoop (Bronze Layer)
   - 2.2. Tiền xử lý, tối ưu lưu trữ và làm sạch dữ liệu (Silver Layer)
   - 2.3. Phân tích khám phá hành vi mua sắm (EDA)
3. **CHƯƠNG 3: KHAI PHÁ GIỎ HÀNG VỚI SPARK MLLIB FP-GROWTH & WORD2VEC**
   - 3.1. Gom giỏ hàng (`gold_baskets`) ở mức Aisle và Product
   - 3.2. Thuật toán FP-Growth và lọc luật kết hợp theo chỉ số Lift
   - 3.3. So sánh luật kết hợp với mô hình nhúng sản phẩm Word2Vec (item2vec)
4. **CHƯƠNG 4: MÔ HÌNH DỰ ĐOÁN SẢN PHẨM MUA LẠI (REORDER PREDICTION)**
   - 4.1. Chiến lược chia dữ liệu theo thời gian (Tránh Data Leakage)
   - 4.2. Kỹ thuật trích xuất đặc trưng 3 nhóm (User, Product, User-Product)
   - 4.3. Huấn luyện và đánh giá mô hình (Logistic Regression vs. GBTClassifier)
5. **CHƯƠNG 5: TÍCH HỢP HỆ GỢI Ý VÀ XÂY DỰNG WEB DASHBOARD DEMO**
   - 5.1. Logic tích hợp đa tầng gợi ý (`gold_recommendations`)
   - 5.2. Màn hình Web Dashboard phục vụ tra cứu và trực quan hóa (Streamlit)
   - 5.3. Phân khúc khách hàng RFM với K-Means
6. **CHƯƠNG 6: TỐI ƯU HỆ THỐNG, KIỂM THỬ VÀ KẾT LUẬN**
   - 6.1. Bảng đối chiếu số dòng và tính toàn vẹn dữ liệu (Data Lineage)
   - 6.2. Đánh giá hiệu năng và chiến lược tối ưu hóa Spark
   - 6.3. Kết luận và hướng phát triển

---

## 📝 NỘI DUNG CHI TIẾT CÁC CHƯƠNG

---

### CHƯƠNG 1: TỔNG QUAN DỰ ÁN VÀ KIẾN TRÚC HỆ THỐNG
*(Do Thành viên 5 chủ trì soạn thảo)*

#### 1.1. Bối cảnh bài toán và ý nghĩa kinh doanh
- **Thách thức kinh doanh:** Bộ dữ liệu Instacart gồm hơn 3,4 triệu đơn hàng của hơn 200.000 khách hàng. Trong đó, bảng chi tiết sản phẩm trong đơn (`order_products__prior`) có quy mô vượt quá **32 triệu dòng**. Việc truy vấn trực tiếp trên CSDL quan hệ truyền thống sẽ gây lỗi tràn bộ nhớ (Out-Of-Memory) hoặc quá tải hệ thống.
- **Mục tiêu kỹ thuật:** Xây dựng một End-to-End Big Data Pipeline xử lý phân tán 32 triệu dòng nhằm trả lời 2 câu hỏi lớn:
  1. Sản phẩm nào khách hàng sẽ mua lại trong đơn tiếp theo?
  2. Những sản phẩm nào thường được mua kèm cùng nhau?

#### 1.2. Giải pháp kiến trúc Medallion
Hệ thống được thiết kế theo kiến trúc **Medallion Architecture**:
- **Source Layer:** CSDL quan hệ MariaDB chứa 6 bảng gốc.
- **Bronze Layer (Raw):** Lưu trữ dữ liệu thô dạng Parquet trên HDFS thông qua Apache Sqoop.
- **Silver Layer (Cleansed):** Dữ liệu được làm sạch, xử lý NULL hợp lệ, đánh chỉ mục và phân vùng (Partitioning/Bucketing) bằng Pig/Hive/Spark SQL.
- **Gold Layer (Curated & ML):** Các bảng tổng hợp giỏ hàng, bảng đặc trưng ML, luật kết hợp FP-Growth và kết quả dự đoán mua lại.
- **Serving Layer:** Web Dashboard tương tác làm bằng Streamlit.

---

### CHƯƠNG 2: LƯU TRỮ, LÀM SẠCH VÀ PHÂN TÍCH KHÁM PHÁ (BRONZE & SILVER)
*(Do Thành viên 1 và Thành viên 2 biên soạn)*

#### 2.1. Quyết định Ingestion và nạp dữ liệu Sqoop (Thành viên 1)
- **Schema MariaDB:** Bảng thiết kế khóa chính, khóa ngoại (`user_id`, `order_id`, `product_id`).
- **Lệnh Sqoop Import:** Đưa 6 bảng từ MariaDB lên HDFS định dạng Parquet. Đối với các bảng lớn (`order_products__prior`), sử dụng tham số `--split-by order_id` và tăng số lượng mapper để nạp song song.
- **Bảng đối chiếu ban đầu:**
  *(Thành viên 1 chèn bảng đếm số dòng giữa MariaDB và HDFS)*

#### 2.2. Làm sạch và tối ưu lưu trữ tầng Silver (Thành viên 2)
- **Quy tắc xử lý dữ liệu:**
  - Chuẩn hóa tên sản phẩm, loại bỏ các bản ghi trùng lặp.
  - Giữ lại giá trị `NULL` tại cột `days_since_prior_order` cho đơn hàng đầu tiên của mỗi người dùng và thêm cờ `is_first_order = 1` (tránh xóa nhầm dữ liệu hợp lệ).
- **Định dạng lưu trữ:** Lưu trữ tầng Silver dưới dạng ORC/Parquet có phân vùng theo `order_dow` (ngày trong tuần) hoặc `user_id`.

#### 2.3. Kết quả phân tích hành vi mua sắm - EDA (Thành viên 2)
- Biểu đồ phân bố đơn hàng theo các khung giờ trong ngày (`order_hour_of_day`).
- Biểu đồ phân bố khoảng cách giữa các đơn hàng (các đỉnh phổ biến tại 7 ngày và 30 ngày).
- Top 10 sản phẩm bán chạy nhất và tỷ lệ mua lại (Reorder Rate) theo từng ngành hàng (Aisle/Department).

---

### CHƯƠNG 3: KHAI PHÁ GIỎ HÀNG VỚI SPARK MLLIB FP-GROWTH & WORD2VEC
*(Do Thành viên 3 biên soạn)*

#### 3.1. Gom giỏ hàng (`gold_baskets`)
- Xây dựng mảng danh sách sản phẩm (`items`) cho mỗi `order_id` ở 2 mức độ: mức sản phẩm (`product_id`) và mức nhóm hàng (`aisle_id`).

#### 3.2. Thuật toán FP-Growth và Lọc luật kết hợp
- **Cấu hình tham số:** Thử nghiệm các cặp `minSupport` và `minConfidence`.
- **Lọc luật theo chỉ số Lift:** Chỉ giữ lại các luật có $\text{Lift} > 1$ để loại bỏ các sản phẩm phổ biến ảo (như Chuối, Túi đựng) gây nhiễu.
- **Bảng kết quả top luật kết hợp nổi bật:**
  *(Thành viên 3 chèn bảng hiển thị Antecedent -> Consequent, Support, Confidence, Lift)*

#### 3.3. So sánh với Mô hình Nhúng sản phẩm (item2vec / Word2Vec)
- Huấn luyện Word2Vec trên danh sách giỏ hàng để tìm độ tương đồng vector giữa các sản phẩm và so sánh với danh sách gợi ý từ FP-Growth.

---

### CHƯƠNG 4: MÔ HÌNH DỰ ĐOÁN SẢN PHẨM MUA LẠI (REORDER PREDICTION)
*(Do Thành viên 4 biên soạn)*

#### 4.1. Chiến lược chia dữ liệu (Time-based Split)
- Sử dụng các đơn hàng lịch sử (`eval_set = 'prior'`) để trích xuất đặc trưng.
- Sử dụng đơn hàng cuối cùng (`eval_set = 'train'`) để làm nhãn dự đoán (1 nếu mua lại, 0 nếu không). Phương pháp này giúp tránh triệt để tình trạng rò rỉ dữ liệu (Data Leakage).

#### 4.2. Danh sách Đặc trưng (Feature Engineering)
1. **Nhóm User:** Tổng số đơn, cỡ giỏ trung bình, tỷ lệ mua lại tổng thể, khoảng cách trung bình giữa các đơn.
2. **Nhóm Product:** Tỷ lệ mua lại toàn hệ thống, tổng số khách đã mua, vị trí trung bình khi thêm vào giỏ.
3. **Nhóm User-Product:** Số lần mua sản phẩm đó, số đơn kể từ lần mua cuối, chuỗi mua liên tiếp, vị trí thêm vào giỏ trung bình.

#### 4.3. Đánh giá Mô hình Học máy
- **So sánh Mô hình:** Logistic Regression (Baseline) vs. GBTClassifier (Mô hình chính).
- **Chỉ số đánh giá:**
  *(Thành viên 4 điền bảng so sánh các chỉ số)*

| Mô hình | Precision | Recall | F1-Score | AUC-ROC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | ... | ... | ... | ... | ... |
| **GBTClassifier** | ... | ... | ... | ... | ... |

---

### CHƯƠNG 5: TÍCH HỢP HỆ GỢI Ý VÀ XÂY DỰNG WEB DASHBOARD DEMO
*(Do Thành viên 5 chủ trì soạn thảo)*

#### 5.1. Tích hợp Đa tầng Gợi ý (`gold_recommendations`)
- Lớp 1: Top sản phẩm có xác suất mua lại cao nhất từ mô hình ML (Spark MLlib).
- Lớp 2: Bổ sung các sản phẩm gợi ý mua kèm dựa trên luật kết hợp FP-Growth (Lift > 1).
- Lớp 3: Phân khúc khách hàng K-Means RFM để tùy biến gợi ý theo từng nhóm người dùng.

#### 5.2. Giao diện Web Dashboard (Streamlit)
- **Hình ảnh Màn hình 1:** Tổng quan biểu đồ EDA hành vi mua sắm.
- **Hình ảnh Màn hình 2:** Tra cứu danh sách luật kết hợp sản phẩm.
- **Hình ảnh Màn hình 3:** Giao diện tra cứu gợi ý giỏ hàng tiếp theo theo `user_id`.

---

### CHƯƠNG 6: TỐI ƯU HỆ THỐNG, KIỂM THỬ VÀ KẾT LUẬN
*(Chung cả nhóm - Thành viên 5 tổng hợp)*

#### 6.1. Bảng đối chiếu số dòng (Data Lineage / Data Reconciliation)
Bảng chứng minh tính toàn vẹn dữ liệu 100% qua các tầng do Thành viên 5 thực hiện:

| Bảng dữ liệu | MariaDB (Gốc) | Bronze Layer (HDFS) | Silver Layer (Cleaned) | Trạng thái |
| :--- | :--- | :--- | :--- | :--- |
| **orders** | 3,421,083 | 3,421,083 | 3,421,083 | PASSED |
| **order_products__prior** | 32,434,489 | 32,434,489 | 32,434,489 | PASSED |
| **products** | 49,688 | 49,688 | 49,688 | PASSED |
| **aisles** | 134 | 134 | 134 | PASSED |
| **departments** | 21 | 21 | 21 | PASSED |

#### 6.2. Đánh giá hiệu năng và tối ưu hóa Spark
- **Tối ưu Cấu hình Spark:**
  - Tăng `spark.sql.shuffle.partitions` từ 200 lên 400 để xử lý 32 triệu dòng không bị giật lag.
  - Kích hoạt Adaptive Query Execution (`spark.sql.adaptive.enabled = true`).
  - Áp dụng `broadcast()` join cho các bảng danh mục nhỏ (`products`, `aisles`, `departments`).
- **Bảng so sánh thời gian chạy trước và sau tối ưu:**

| Tiến trình xử lý | Thời gian trước tối ưu | Thời gian sau tối ưu | Tỷ lệ cải thiện |
| :--- | :--- | :--- | :--- |
| **Tạo bảng Silver & Làm sạch** | ...