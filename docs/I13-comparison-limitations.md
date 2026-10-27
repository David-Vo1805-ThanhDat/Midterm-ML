# I13 — So sánh mô hình và phân tích giới hạn

## Mục tiêu

Đánh giá lựa chọn K, độ nhạy khởi tạo, label switching và nguy cơ covariance suy biến; nêu rõ điều kiện so sánh và giới hạn bằng chứng.

## Phạm vi và cấu hình

Các con số dưới đây đến từ kiểm chứng NumPy độc lập, chạy trên dữ liệu tổng hợp và Old Faithful chuẩn hóa. Mỗi K có 30 seed (0–29), covariance full, `reg_covar=1e-6`, `tol=1e-6`, tối đa 500 vòng; BIC dùng cùng thang dữ liệu và p=6K−1. Chi tiết dữ liệu và lệnh chạy được ghi trong [I13-evaluation-results.md](I13-evaluation-results.md).
eg_covar=1e-6, 	ol=1e-6, tối đa 500 vòng; BIC dùng cùng thang dữ liệu và p=6K−1. Chi tiết dữ liệu và lệnh chạy được ghi trong [I13-evaluation-results.md](I13-evaluation-results.md).

## Old Faithful: BIC, diễn giải và mức độ chắc chắn

| K | Số tham số p | Log-likelihood tốt nhất | BIC | Seed tốt nhất |
|---:|---:|---:|---:|---:|
| 1 | 5 | -544.993 | 1118.016 | 0 |
| 2 | 11 | -385.461 | **832.585** | 13 |
| 3 | 17 | -369.637 | 834.573 | 1 |
| 4 | 23 | -361.230 | 851.393 | 1 |
| 5 | 29 | -355.126 | 872.821 | 27 |
| 6 | 35 | -347.817 | 891.837 | 12 |

Theo BIC, chọn K=2. BIC của K=3 chỉ cao hơn 1.988, nên ưu thế của K=2 khá sát: thêm một thành phần cải thiện log-likelihood, nhưng phần phạt phức tạp gần như bù lại. Đây là lựa chọn theo BIC trong tập K khảo sát, không phải bằng chứng chắc chắn về số nhóm thật. BIC được dùng như tiêu chí xấp xỉ để so sánh mô hình; cách diễn giải mức chênh lệch cần thận trọng (Kass & Raftery, 1995).

Ở nghiệm K=2 tốt nhất, sau đổi ngược về phút:
- Thành phần 1: trọng số 0.644; mean xấp xỉ (4.290 phút phun, 79.968 phút chờ).
- Thành phần 2: trọng số 0.356; mean xấp xỉ (2.036 phút phun, 54.479 phút chờ).

Điều này phù hợp với hai vùng ngắn/chờ ngắn và dài/chờ dài thường thấy trong biểu đồ dữ liệu. Nhưng thành phần Gaussian chỉ mô tả mật độ; không đủ để kết luận có hai cơ chế vật lý của mạch nước.

## Label switching không phải thay đổi chất lượng

Tên thành phần 1, 2, 3 không có ý nghĩa cố định. Nếu hoán đổi chỉ số thành phần và hoán đổi cùng lúc trọng số, mean, covariance, phân phối hỗn hợp vẫn y hệt. Vì vậy:
- Không so trực tiếp mean thành phần 1 giữa hai lần chạy trước khi ghép nhãn.
- Trên dữ liệu tổng hợp, bảng tham số phía trên ghép theo mean với ground truth sau fit để dễ đọc.
- ARI so phân hoạch và bất biến với hoán vị nhãn; log-likelihood/BIC cũng không đổi khi đổi tên thành phần.
- Với Old Faithful không có nhãn thật, so log-likelihood, BIC, covariance và trách nhiệm; không dùng chỉ số cụm 0/1 để gán ý nghĩa.

Nguồn gốc so sánh phân hoạch và hiệu chỉnh Rand index: Hubert & Arabie (1985).

## Covariance suy biến và regularization

Trong các lượt Old Faithful, eigenvalue covariance nhỏ nhất theo K=1..6 lần lượt là 0.09919, 0.01923, 0.00285, 0.00186, 0.000885, 0.000939. Các giá trị này vẫn cao hơn reg_covar=1e-6, nên không thấy covariance chạm sàn regularization trong bộ chạy Old Faithful này.

Trên dữ liệu tổng hợp với K=5, seed khởi tạo 19 tạo covariance có eigenvalue nhỏ nhất xấp xỉ **1.0e-6**, đúng tại sàn regularization. Thành phần đó có trọng số 0.2973 và mean gần (4.004, 3.931). Đây là dấu hiệu fit đang muốn co covariance để tăng density; regularization ngăn ma trận thành suy biến thật. BIC tốt nhất của K=5 vẫn cao hơn K=3, nên nghiệm co này không làm thay đổi K được chọn.

Kết quả phụ thuộc reg_covar: tăng sàn có thể làm covariance ổn định hơn nhưng cũng đổi likelihood và nghiệm. Cần ghi giá trị regularization khi báo cáo; không so BIC từ hai cấu hình regularization khác nhau như thể chúng cùng một phép thử.

## So sánh với scikit-learn: tình trạng và điều kiện công bằng

Chưa chạy baseline sklearn.mixture.GaussianMixture: runtime hiện tại không cài scikit-learn, dù requirements.txt có khai báo thư viện. Đây là mục chưa hoàn tất, không phải kết quả âm tính.

Khi có baseline, so trên cùng ma trận X đã chuẩn hóa, cùng K, covariance_type="full", reg_covar=1e-6, tol, max_iter và số lượt khởi tạo. Để so tham số theo đúng seed, cần cấp cùng initial weights/means/precisions hoặc ít nhất mô tả rõ khác biệt khởi tạo. Sau fit phải ghép thứ tự thành phần trước khi so pi, mean, covariance; log-likelihood/BIC/ARI không cần ghép nhãn. Tài liệu API scikit-learn định nghĩa converged_, n_iter_, lower_bound_, lower_bounds_ và ý nghĩa của random_state, tol, reg_covar; không so lower_bound_ nếu chưa xác nhận cùng quy ước likelihood và cùng thang dữ liệu.

## Hạn chế và kết luận

1. **Chưa đối chiếu với EM cốt lõi của I11 hoặc baseline scikit-learn.** Repo hiện không có các kết quả đó; script này là kiểm chứng độc lập. Trước khi nhập số liệu vào báo cáo cuối, cần chạy lại cùng cài đặt EM chính và cùng cấu hình.
2. **Khởi tạo giới hạn.** 30 seed là bằng chứng về độ nhạy trong lượt chạy này, không bảo đảm tìm được nghiệm toàn cục.
3. **Old Faithful nhỏ và bị làm tròn.** Dữ liệu có 272 điểm; giá trị lặp và độ phân giải khác nhau giữa cột có thể ảnh hưởng covariance.
4. **Giả định i.i.d.** GMM coi mỗi quan sát độc lập, dù các lần phun theo thời gian có thể phụ thuộc.
5. **Gaussian có thể chỉ là xấp xỉ mật độ.** K không nhất thiết là số nhóm tự nhiên hay cơ chế vật lý.
6. **BIC là tiêu chí chọn mô hình, không phải xác nhận chân lý.** K=2 được chọn nhưng sát K=3; cần giữ kết luận thận trọng.
7. **Regularization làm thay đổi ước lượng.** reg_covar=1e-6 được dùng để giữ covariance xác định dương; kết quả không phải fit MLE hoàn toàn không ràng buộc khi covariance chạm sàn.

Tóm lại, kiểm chứng độc lập khôi phục tốt K=3 trên dữ liệu tổng hợp (BIC nhỏ nhất, ARI 0.9774) và chọn K=2 trên Old Faithful theo BIC. Thí nghiệm cũng cho thấy seed có thể dẫn đến nghiệm rất khác trên Old Faithful, và covariance có thể tiến sát sàn regularization khi K lớn. Các nhận định đều gắn số liệu cụ thể; chúng chưa thay thế việc đối chiếu với mã EM chính của nhóm.

## Tài liệu tham khảo

1. Christopher M. Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, Chương 9.
2. A. P. Dempster, N. M. Laird, D. B. Rubin (1977), “Maximum Likelihood from Incomplete Data Via the EM Algorithm,” *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22. [DOI](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
3. Lawrence Hubert and Phipps Arabie (1985), “Comparing partitions,” *Journal of Classification*, 2, 193-218. [DOI](https://doi.org/10.1007/BF01908075).
4. Robert E. Kass and Adrian E. Raftery (1995), “Bayes Factors,” *Journal of the American Statistical Association*, 90(430), 773-795. [DOI](https://doi.org/10.1080/01621459.1995.10476572).
5. scikit-learn developers, [GaussianMixture API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html), including convergence, initialization, covariance and BIC.
6. Data source and preprocessing rationale: docs/data-and-use-case.md (Issue I04); Azzalini & Bowman (1990), *Applied Statistics*, 39, 357-365. [DOI](https://doi.org/10.2307/2347385).
