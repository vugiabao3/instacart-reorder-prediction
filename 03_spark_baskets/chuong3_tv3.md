# CHƯƠNG 3: KHAI PHÁ GIỎ HÀNG VỚI SPARK MLLIB FP-GROWTH & WORD2VEC

*(Do Thành viên 3 biên soạn)*

## 3.1. Gom giỏ hàng (`gold_baskets`)

- **Đầu vào:** chi tiết sản phẩm trong các đơn thuộc tập `prior` (`order_products__prior`, khoảng 32 triệu dòng) và bảng `products` (49.688 sản phẩm, 134 nhóm hàng). Dữ liệu đầy đủ lấy từ bản sao bộ Instacart trên Kaggle; khi pipeline chạy trên cụm, nguồn này sẽ đổi sang bảng `silver_order_products`.
- **Cách gom:** với mỗi `order_id`, dùng `collect_set` để gom các sản phẩm thành một mảng và loại trùng trong cùng một đơn. Tạo hai mức: `items` (mảng `product_id`) và `aisle_items` (mảng `aisle_id`, có được bằng cách nối `products` theo `product_id`).
- **Kiểm tra:** trên toàn bộ dữ liệu có 3.214.874 giỏ hàng ở cả hai mức, bằng đúng số đơn có sản phẩm (code dừng nếu số giỏ hàng không khớp số đơn).
- **Lưu trữ:** trên mẫu 5% (162.845 giỏ hàng), hai bảng được ghi dạng Parquet bằng `step1_build_baskets.py`. Với toàn bộ dữ liệu, `run_baskets_fpgrowth.py` giữ giỏ hàng trong bộ nhớ để chạy FP-Growth và chỉ lưu bảng luật.

## 3.2. Thuật toán FP-Growth và lọc luật kết hợp

- **Cấu hình tham số:** thử nghiệm 9 cặp `minSupport` và `minConfidence` trên mẫu 5% (162.845 giỏ hàng, chạy trên Google Colab):

| minSupport | minConfidence | Số luật | Thời gian (giây) |
|---|---|---|---|
| 0.01 | 0.2 | 5.941 | 348,4 |
| 0.01 | 0.3 | 4.663 | 339,5 |
| 0.01 | 0.5 | 3.051 | 331,8 |
| 0.02 | 0.2 | 1.476 | 343,3 |
| 0.02 | 0.3 | 1.198 | 341,2 |
| 0.02 | 0.5 | 770 | 351,8 |
| 0.05 | 0.2 | 210 | 223,9 |
| 0.05 | 0.3 | 180 | 219,3 |
| 0.05 | 0.5 | 122 | 218,9 |

  Số luật giảm mạnh khi tăng `minSupport`; thời gian gần như không đổi theo `minConfidence` vì bước tốn thời gian nhất là tìm tập phổ biến.
- **Cặp tham số được chọn:** `minSupport = 0.02`, `minConfidence = 0.3`, mức nhóm hàng (aisle). Mức 0.01 sinh gần 4.700 luật dài, khó giải thích mà không nhanh hơn; mức 0.05 chỉ còn 180 luật, thiên về các nhóm hàng tươi quá phổ biến.
- **Lọc luật theo Lift:** chạy toàn bộ dữ liệu (3.214.874 giỏ hàng) cho 1.230 luật, chỉ giữ các luật có Lift > 1 để loại các sản phẩm phổ biến ảo, còn **1.219 luật**. Thời gian chạy không quá khoảng 18 phút (ước tính từ mốc giờ chạy, máy 16 GB RAM, `SPARK_MEM = 6g`, `SHUFFLE_PARTS = 64`).
- **Bảng kết quả top luật kết hợp nổi bật (mức aisle, toàn bộ dữ liệu):**

| # | Antecedent → Consequent | Support | Confidence | Lift | Diễn giải |
|---|---|---|---|---|---|
| 1 | lunch meat + yogurt → packaged cheese | 0,020 | 0,503 | 2,193 | Lift cao nhất trong các luật ngắn: khách mua đồ ăn lạnh, ăn nhanh hay mua thêm phô mai |
| 2 | fresh dips tapenades → chips pretzels | 0,035 | 0,357 | 2,135 | Mua nước chấm tươi thì mua snack để chấm |
| 3 | lunch meat + fresh fruits → bread | 0,025 | 0,349 | 2,126 | Chuẩn bị bữa ăn nhanh kiểu sandwich |
| 4 | pasta sauce → packaged cheese | 0,027 | 0,440 | 1,918 | Bữa mì Ý: sốt đi kèm phô mai |
| 5 | fresh herbs → fresh vegetables | 0,079 | 0,844 | 1,901 | Confidence rất cao nhưng Lift chỉ gần 1,9 |

- **Vì sao Lift quan trọng hơn Confidence:** ở luật 5, 84,4% đơn có rau thơm cũng có rau tươi, nhưng rau tươi vốn xuất hiện trong khoảng 44% tổng số đơn (0,844 / 1,901), nên mức đi kèm thực chất chỉ gấp khoảng 1,9 lần. Ngược lại luật 2 có Confidence thấp hơn (0,357) nhưng snack chỉ có trong khoảng 17% đơn, nên Lift tới 2,135 cho thấy liên hệ mạnh hơn.
- **Hạn chế:** Lift cao nhất chỉ khoảng 2,2 (liên hệ vừa phải); luật kết hợp chỉ cho thấy hai nhóm hàng hay đi cùng nhau, không chứng minh quan hệ nhân quả; FP-Growth mới chạy ở mức nhóm hàng, chưa chạy mức sản phẩm.

## 3.3. So sánh với mô hình nhúng sản phẩm (item2vec / Word2Vec)

- **Cách làm:** coi mỗi giỏ hàng là một "câu", mỗi sản phẩm là một "từ" và huấn luyện Word2Vec (Spark MLlib) với vectorSize = 50, minCount = 5, windowSize = 10, maxIter = 5, numPartitions = 4, seed = 42. Trên toàn bộ dữ liệu (3.214.874 giỏ hàng), huấn luyện mất 1.078 giây (máy 16 GB RAM) và cho vector của 47.618 sản phẩm. Vector của một nhóm hàng là trung bình vector đã chuẩn hóa của các sản phẩm trong nhóm; độ gần đo bằng cosine.
- **Ví dụ sản phẩm gần nhau:** Light Brown Sugar gần Cream Of Tartar, All Vegetable Shortening, Cake Blend Unbleached Flour và Baking Powder (nguyên liệu làm bánh); Guacamole gần Fresh Salsa, Jalapeno Jack Queso và Hommus (nhóm sốt chấm). Kết quả vẫn còn nhiễu ở một số sản phẩm.
- **Cách so sánh với FP-Growth:** với 254 luật dạng "một nhóm → một nhóm" (trong 1.219 luật mức aisle), xếp hạng nhóm vế phải trong 133 nhóm còn lại theo độ gần với nhóm vế trái.

| Nhóm vế phải nằm trong top-K nhóm gần nhất | Thực tế | Ngẫu nhiên |
|---|---|---|
| K = 1 | 2,4% | 0,8% |
| K = 3 | 7,1% | 2,3% |
| K = 5 | 8,7% | 3,8% |
| K = 10 | 13,8% | 7,5% |

- **Nhận xét:** hai phương pháp khớp ở mức vừa phải, tốt hơn ngẫu nhiên khoảng 2 đến 3 lần ở các hạng đầu; hạng trung vị của nhóm vế phải là 48/133, nhỉnh hơn mức ngẫu nhiên (khoảng 67). Kết quả trên mẫu 5% gần như trùng khớp, cho thấy kết luận ổn định. Với các luật Lift cao nhất, độ khớp tốt hơn: pasta sauce → packaged cheese và tortillas flat bread → packaged cheese xếp hạng 1, fresh dips tapenades → chips pretzels xếp hạng 2, cookies cakes → chips pretzels xếp hạng 3; lunch meat → bread xếp hạng 24.
- **Giải thích khả dĩ:** FP-Growth đo hai nhóm hàng có hay nằm cùng một giỏ hay không, còn Word2Vec đo hai sản phẩm có xuất hiện trong ngữ cảnh giỏ hàng giống nhau hay không, nên vector cũng đặt các sản phẩm thay thế nhau gần nhau. Đây là giả thuyết, chưa được kiểm chứng riêng trong bài.
- **Hạn chế:** kết quả tính với một bộ tham số và một hạt giống ngẫu nhiên; chỉ 254 luật được đem so sánh nên các tỷ lệ phần trăm dao động khá lớn; FP-Growth mới chạy ở mức nhóm hàng nên so sánh thực hiện ở mức nhóm hàng.
