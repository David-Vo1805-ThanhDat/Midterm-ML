# I13 — Đánh giá kết quả thực nghiệm EM/GMM

## Tóm tắt

Thực nghiệm kiểm chứng EM/GMM trên hai bộ dữ liệu đã chọn trong I04: dữ liệu tổng hợp 2D có 3 thành phần đã biết và Old Faithful 272 x 2 được chuẩn hóa z-score. Mỗi giá trị K được chạy với 30 seed khởi tạo; dùng covariance đầy đủ và giữ nghiệm có log-likelihood cao nhất trong 30 lượt.

| Kết quả | Tổng hợp 2D | Old Faithful chuẩn hóa |
|---|---:|---:|
| K được BIC chọn | 3 | 2 |
| BIC nhỏ nhất | 2387.722 | 832.585 |
| Ứng viên kế tiếp | K=4: 2396.292 (cao hơn 8.571) | K=3: 834.573 (cao hơn 1.988) |
| Hội tụ / log-likelihood không giảm | 180/180 lượt | 180/180 lượt |

Old Faithful nghiêng về K=2 theo BIC, nhưng khoảng cách với K=3 chỉ 1.988. Vì vậy báo cáo K=2 là lựa chọn theo tiêu chí này, không kết luận chắc chắn dữ liệu chỉ có đúng hai nhóm tự nhiên.

## Dữ liệu và phạm vi kiểm chứng

- Dữ liệu tổng hợp được tạo bằng hàm src.data.make_synthetic() với seed dữ liệu 42, n=300. Nhãn thành phần thật chỉ được dùng sau khi fit để tính ARI và ghép thành phần cho bảng tham số; không tham gia huấn luyện hay chọn K. Số điểm theo thành phần là 155, 88, 57.
- Old Faithful lấy từ data/faithful.csv, giữ hai cột eruptions và waiting, n=272; chuẩn hóa từng cột bằng mean và độ lệch chuẩn của chính bộ dữ liệu. Mean là 3.4878 phút và 70.8971 phút; độ lệch chuẩn là 1.1393 và 13.5700.
- Repo hiện không có mã EM hay kết quả từ dependency I11/I12. Vì vậy i13_experiment.py là chương trình kiểm chứng độc lập, tối giản, viết bằng NumPy; không được xem là kết quả đối chiếu với phần cài đặt EM chính của I11.

## Cấu hình và quy tắc so sánh

Chạy lại từ thư mục gốc repo bằng lệnh **python i13_experiment.py**.

| Mục | Giá trị |
|---|---|
| Seed khởi tạo | 0 đến 29, 30 lượt cho mỗi K |
| Seed dữ liệu tổng hợp | 42 |
| K khảo sát | 1, 2, 3, 4, 5, 6 |
| Covariance | full, mỗi thành phần có ma trận riêng |
| Khởi tạo | K mean lấy từ K hàng dữ liệu khác nhau; trọng số bằng nhau; covariance khởi tạo chung bằng covariance toàn dữ liệu |
| Regularization | reg_covar = 1e-6, cộng vào đường chéo covariance sau M-step |
| Dừng | Mức tăng log-likelihood trên mỗi điểm nhỏ hơn tol = 1e-6; tối đa 500 vòng |
| Chọn nghiệm trong một K | Lấy log-likelihood quan sát cao nhất từ 30 lượt |
| BIC | BIC = -2 log L + p ln(n), p = (K-1) + 2K + 3K = 6K-1 |

Mọi so sánh BIC bên dưới dùng cùng bộ dữ liệu, cùng cách chuẩn hóa, cùng covariance full, cùng regularization và cùng quy tắc đếm tham số. BIC Old Faithful được tính trên dữ liệu chuẩn hóa; không so trực tiếp với BIC trên phút gốc vì đổi đơn vị làm log-likelihood và BIC dịch theo hằng số.

## Log-likelihood và điều kiện dừng

Mỗi vòng tính lại log-likelihood quan sát sau khi cập nhật đủ tham số. Với mỗi bộ dữ liệu có 6 giá trị K x 30 seed = 180 lượt. Cả 180 lượt trên dữ liệu tổng hợp và 180 lượt trên Old Faithful đều đạt điều kiện dừng; log-likelihood trong từng lượt không giảm, với sai số kiểm tra số học 1e-8.

Một đường chạy đại diện:

| Dữ liệu và lượt | Vòng lặp | Log-likelihood |
|---|---:|---:|
| Tổng hợp, K=3, seed 9 | Khởi tạo | -1423.355 |
|  | 1 | -1334.294 |
|  | 5 | -1211.889 |
|  | 10 | -1167.322 |
|  | 15 | -1146.224 |
|  | 21 (dừng) | -1145.379 |
| Old Faithful, K=2, seed 13 | Khởi tạo | -606.047 |
|  | 1 | -467.933 |
|  | 5 | -388.480 |
|  | 8 | -385.461 |
|  | 10 (dừng) | -385.461 |

Ở hàng Old Faithful, giá trị -385.461 là vòng 8; vòng 9 cũng đạt -385.461 khi làm tròn 6 chữ số. Các số cho thấy likelihood tăng mạnh ở những vòng đầu rồi đổi rất ít gần điểm dừng. Điều kiện dừng chỉ nói mức cải thiện nhỏ theo ngưỡng đã chọn; nó không chứng minh nghiệm là cực đại toàn cục. EM chuẩn có tính chất log-likelihood không giảm (Dempster, Laird & Rubin, 1977); ở đây có regularization covariance nên kết luận thực nghiệm dựa trên log-likelihood thực đã tính sau mỗi vòng, không chỉ dựa vào định lý.

## Dữ liệu tổng hợp: khôi phục ba thành phần

BIC ở mỗi K dùng lượt có log-likelihood cao nhất trong 30 seed:

| K | Số tham số p | Log-likelihood tốt nhất | BIC | Seed tốt nhất |
|---:|---:|---:|---:|---:|
| 1 | 5 | -1384.126 | 2796.771 | 0 |
| 2 | 11 | -1219.236 | 2501.214 | 10 |
| 3 | 17 | -1145.379 | **2387.722** | 9 |
| 4 | 23 | -1132.553 | 2396.292 | 8 |
| 5 | 29 | -1127.161 | 2419.732 | 21 |
| 6 | 35 | -1118.449 | 2436.530 | 25 |

BIC chọn K=3, trùng số thành phần thật. K=4 có likelihood cao hơn K=3 nhưng bị phạt vì thêm tham số; BIC của K=4 cao hơn 8.571. Đây là ví dụ cụ thể về lý do không chọn K chỉ bằng likelihood.

Ở nghiệm K=3 tốt nhất, ARI giữa nhãn dự đoán và nhãn sinh dữ liệu là **0.9774**. ARI gần 1 cho thấy hai phân hoạch rất khớp. ARI chỉ được tính sau fit vì dữ liệu này có nhãn thật; Old Faithful không có nhãn để tính ARI. ARI điều chỉnh độ phù hợp của hai phân hoạch theo mức đồng thuận ngẫu nhiên (Hubert & Arabie, 1985).

Để đọc bảng tham số, các thành phần được ghép với thành phần thật gần nhất theo mean sau khi fit. Việc ghép này chỉ đổi thứ tự hàng, không đổi mô hình.

| Thành phần thật | Trọng số thật -> ước lượng | Mean thật -> ước lượng | Covariance thật -> ước lượng |
|---|---|---|---|
| 1 | 0.5000 -> 0.5148 | (0, 0) -> (-0.0318, -0.0012) | [[1, 0], [0, 1]] -> [[1.0605, 0.1742], [0.1742, 1.0094]] |
| 2 | 0.3000 -> 0.2954 | (4, 4) -> (4.0166, 3.9413) | [[1.5, 0.8], [0.8, 1.0]] -> [[1.8621, 0.8612], [0.8612, 0.9678]] |
| 3 | 0.2000 -> 0.1899 | (-3, 4) -> (-3.0741, 3.9245) | [[0.5, 0], [0, 2]] -> [[0.3830, 0.1917], [0.1917, 2.1844]] |

Các mean và trọng số gần giá trị thật; covariance cũng nắm được độ rộng/hướng chính nhưng khác đáng kể ở mẫu nhỏ. Ví dụ thành phần thứ nhất có hiệp phương sai ước lượng 0.1742 dù giá trị thật bằng 0. Đây là biến động mẫu; không nên gọi là lỗi thuật toán chỉ từ một con số.

## Độ nhạy seed và nghiệm cục bộ

Với dữ liệu tổng hợp, K=3, 30 khởi tạo cho log-likelihood trong khoảng -1145.378881 đến -1145.378698; các nghiệm gần như trùng nhau. ARI của nghiệm có likelihood tốt nhất là 0.9774.

Old Faithful cho thấy tình huống khác ở K=2:

| Seed | Log-likelihood | Số vòng | Trọng số | Mean hai thành phần, đã chuẩn hóa |
|---:|---:|---:|---|---|
| 13, tốt nhất | -385.460697 | 10 | (0.6441, 0.3559) | (0.7039, 0.6685), (-1.2740, -1.2099) |
| 20, thấp nhất | -540.512430 | 135 | (0.8215, 0.1785) | (0.0445, 0.1484), (-0.2047, -0.6830) |

Cả hai lượt đều đạt tiêu chuẩn dừng và likelihood từng lượt không giảm, nhưng seed 20 kết thúc thấp hơn seed 13 khoảng **155.052 log-likelihood units**. Seed 20 tạo hai Gaussian chồng lấn quanh vùng trung tâm thay vì tách rõ hai vùng dữ liệu. Đây là nghiệm hội tụ kém, phù hợp với hiện tượng EM nhạy điểm khởi tạo và có thể tới một nghiệm cục bộ không tốt; tài liệu này không tính Hessian nên không khẳng định đó là cực đại địa phương theo nghĩa giải tích.

Kết luận thực hành: converged=True hoặc likelihood tăng trong mỗi lượt không đủ để chọn nghiệm. Cần nhiều khởi tạo và chọn likelihood tốt nhất trong cùng một K trước khi so BIC.


## Kết luận phần thực nghiệm

Trên dữ liệu tổng hợp, BIC chọn đúng K=3 và ARI đạt 0.9774. Trên Old Faithful, BIC chọn K=2 nhưng chênh lệch với K=3 chỉ 1.988. Các seed khác nhau có thể dẫn tới chất lượng nghiệm khác nhau đáng kể.

## Tài liệu tham khảo

1. Christopher M. Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, Chương 9.
2. A. P. Dempster, N. M. Laird, D. B. Rubin (1977), “Maximum Likelihood from Incomplete Data Via the EM Algorithm,” *Journal of the Royal Statistical Society: Series B*, 39(1), 1–22. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
3. Lawrence Hubert and Phipps Arabie (1985), “Comparing partitions,” *Journal of Classification*, 2, 193–218. [DOI](https://doi.org/10.1007/BF01908075).
4. Nguồn dữ liệu và tiền xử lý: `docs/data-and-use-case.md`; Azzalini & Bowman (1990), *Applied Statistics*, 39, 357–365. [DOI](https://doi.org/10.2307/2347385).
