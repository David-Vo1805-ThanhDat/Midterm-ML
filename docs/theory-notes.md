# GMM và thuật toán Expectation–Maximization (EM)

Tài liệu này giải thích mô hình hỗn hợp Gaussian (Gaussian Mixture Model, GMM), vì sao ta dùng EM để ước lượng tham số, và những giới hạn cần lưu ý. Ký hiệu và các cập nhật dưới đây theo Bishop, Ch. 9, và ghi chú CS229 của Andrew Ng; lập luận tổng quát về EM bắt nguồn từ Dempster, Laird và Rubin. [1–3]

## 1. Mô hình và ký hiệu

Cho dữ liệu độc lập $X=\{x_1,\ldots,x_N\}$, với $x_n\in\mathbb R^D$. GMM giả định mỗi điểm được sinh bởi một trong $K$ Gaussian:

$$
p(x_n\mid\theta)=\sum_{k=1}^{K}\pi_k\,\mathcal N(x_n\mid\mu_k,\Sigma_k),
\qquad \pi_k\geq 0,\quad \sum_{k=1}^{K}\pi_k=1.
$$

Ở đây $\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$; $\pi_k$ là mixing weight (xác suất chọn cụm $k$), $\mu_k$ là vector trung bình, và $\Sigma_k$ là ma trận hiệp phương sai đối xứng xác định dương. Mật độ Gaussian là

$$
\mathcal N(x\mid\mu,\Sigma)=
\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}
\exp\!\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right].
$$

Ta biểu diễn nhãn cụm chưa quan sát bằng biến ẩn one-hot $z_n=(z_{n1},\ldots,z_{nK})$, trong đó $z_{nk}\in\{0,1\}$, $\sum_k z_{nk}=1$. Nếu $z_{nk}=1$, điểm $x_n$ được sinh từ Gaussian thứ $k$. Khi biết nhãn, mật độ đồng thời có thể viết:

$$
p(x_n,z_n\mid\theta)=\prod_{k=1}^{K}
[\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)]^{z_{nk}}.
$$

Ký hiệu: $\pi_k$ là xác suất tiên nghiệm của thành phần; $r_{nk}=p(z_{nk}=1\mid x_n,\theta)$ là responsibility, tức xác suất hậu nghiệm thành phần $k$ “chịu trách nhiệm” cho điểm $n$. [1, §9.2; 2]

## 2. Mục tiêu likelihood và vì sao tối ưu trực tiếp khó

Với nhãn ẩn chưa biết, log-likelihood quan sát là

$$
\ell(\theta)=\log p(X\mid\theta)
=\sum_{n=1}^{N}\log\left[
\sum_{k=1}^{K}\pi_k\mathcal N(x_n\mid\mu_k,\Sigma_k)
\right].
$$

Nút thắt là **log của tổng** các thành phần. Các tham số $\pi_k,\mu_k,\Sigma_k$ bị ghép trong từng tổng, nên đạo hàm của biểu thức observed-data không cho các nghiệm MLE dạng đóng quen thuộc như khi đã biết nhãn. Ràng buộc $\sum_k\pi_k=1$ cũng cần được xử lý. Nếu nhãn $z_n$ đã biết, log-likelihood hoàn chỉnh lại là tổng có trọng số của log Gaussian và log mixing weight; khi đó các MLE là trung bình và hiệp phương sai có trọng số. EM luân phiên ước lượng phân bố của nhãn còn thiếu và tối ưu các tham số như thể có “nhãn mềm”. [1, §9.3; 2]

## 3. Ý tưởng EM: giải thích trước bằng lời

- **E-step (Expectation):** Giữ tham số hiện tại cố định. Với mỗi điểm dữ liệu, hỏi “nếu các tham số hiện tại là đúng, xác suất điểm này thuộc từng Gaussian là bao nhiêu?” Những xác suất đó là responsibility $r_{nk}$. Đây là nhãn mềm, không phải nhãn cứng.
- **M-step (Maximization):** Giữ responsibility vừa tính cố định. Cập nhật mixing weight, trung bình và hiệp phương sai bằng cách tối đa hóa log-likelihood hoàn chỉnh kỳ vọng, tức xem mỗi điểm đóng góp một phần $r_{nk}$ vào thành phần $k$.
- Lặp E-step và M-step cho đến khi log-likelihood hoặc các tham số thay đổi rất ít, hoặc đạt giới hạn số vòng lặp. Mỗi vòng dùng tham số cũ để tính responsibility, rồi dùng trách nhiệm đó để tính bộ tham số mới. [1, §§9.3–9.4; 3]

## 4. Công thức E-step

Theo Bayes, với tham số hiện tại $\theta^{old}$:

$$
r_{nk}=p(z_{nk}=1\mid x_n,\theta^{old})
=\frac{\pi_k^{old}\,\mathcal N(x_n\mid\mu_k^{old},\Sigma_k^{old})}
{\sum_{j=1}^{K}\pi_j^{old}\,\mathcal N(x_n\mid\mu_j^{old},\Sigma_j^{old})}.
$$

Tử số là xác suất tiên nghiệm nhân với mật độ điểm dưới Gaussian đó; mẫu số cộng các khả năng trên mọi cụm để chuẩn hóa. Vì vậy $0\le r_{nk}\le1$ và $\sum_k r_{nk}=1$ với từng $n$. [1, §9.3; 2]

## 5. Công thức M-step

Đặt số lượng hiệu dụng (effective membership) của thành phần $k$ là

$$
N_k=\sum_{n=1}^{N} r_{nk}.
$$

Các cập nhật MLE không ràng buộc thêm là

$$
\pi_k^{new}=\frac{N_k}{N},\qquad
\mu_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}r_{nk}x_n,
$$

$$
\Sigma_k^{new}=\frac{1}{N_k}\sum_{n=1}^{N}
r_{nk}(x_n-\mu_k^{new})(x_n-\mu_k^{new})^T.
$$

Do đó $\sum_k\pi_k^{new}=1$. Trong covariance, dùng trung bình mới $\mu_k^{new}$. Với dữ liệu 1D, $\mu_k$ là số vô hướng và $\Sigma_k$ trở thành phương sai $\sigma_k^2$:

$$
\sigma_k^{2,new}=\frac{1}{N_k}\sum_n r_{nk}(x_n-\mu_k^{new})^2.
$$

Các biểu thức là nghiệm cực đại của kỳ vọng log-likelihood dữ liệu hoàn chỉnh với responsibility cố định; chúng là MLE nên mẫu số là $N_k$, không phải $N_k-1$. Công thức cần $N_k>0$; nếu một thành phần có $N_k=0$, M-step không xác định cho thành phần đó và cần quy tắc khởi tạo lại/xử lý thành phần rỗng được nêu riêng. [1, §9.3; 2]

## 6. Vì sao likelihood không giảm qua mỗi vòng EM

Đặt $q_n(z_n)$ là một phân bố bất kỳ trên nhãn ẩn. Từ phân rã log xác suất và bất đẳng thức Jensen, ta có cận dưới

$$
\ell(\theta)=\sum_n\log\sum_{z_n}q_n(z_n)
\frac{p(x_n,z_n\mid\theta)}{q_n(z_n)}
\;\geq\;
\mathcal F(q,\theta)
=\sum_n\sum_{z_n}q_n(z_n)
\log\frac{p(x_n,z_n\mid\theta)}{q_n(z_n)}.
$$

Với $\theta^{old}$, chọn $q_n(z_n)=p(z_n\mid x_n,\theta^{old})$ làm cận dưới chạm log-likelihood tại tham số cũ. E-step tạo đúng phân bố này; M-step chọn $\theta^{new}$ làm $\mathcal F(q,\theta)$ không giảm. Vì cận chạm tại $\theta^{old}$, suy ra $\ell(\theta^{new})\geq\ell(\theta^{old})$. Đây là tính chất tăng đơn điệu trong số học chính xác của EM chuẩn. [3; 1, §9.4]

**Điều tính chất này không nói:** EM không đảm bảo tìm cực đại toàn cục, cũng không đảm bảo nghiệm có ý nghĩa thống kê. Nó đảm bảo likelihood không giảm ở các bước lý tưởng; thuật toán thường tiến tới điểm dừng cục bộ hoặc điểm dừng suy biến. [1, §9.4; 4]

## 7. Kiểm tra dừng trong thực hành

Sau mỗi vòng hoàn chỉnh, tính

$$
\ell^{(t)}=\sum_{n=1}^{N}\log\sum_{k=1}^{K}
\pi_k^{(t)}\mathcal N(x_n\mid\mu_k^{(t)},\Sigma_k^{(t)}).
$$

Một quy tắc minh bạch là dừng khi

$$
\frac{|\ell^{(t)}-\ell^{(t-1)}|}{1+|\ell^{(t-1)}|}<\varepsilon,
$$

hoặc khi đạt `max_iter`. Ghi rõ $\varepsilon$, giới hạn vòng lặp, và liệu log-likelihood đã được tính sau M-step hay chưa. Có thể theo dõi thêm biến thiên tham số và kích thước $N_k$; không nên chỉ dừng vì nhãn cụm có vẻ không đổi. Do tổng mật độ có thể rất nhỏ, triển khai ổn định số nên tính log-sum-exp thay vì lấy log trực tiếp của tổng các mật độ. Các giá trị thực tế của $\varepsilon$ tùy dữ liệu và độ chính xác số, không phải hằng số lý thuyết. [1, §9.4; 2]

## 8. Khởi tạo, local optimum và phụ thuộc điểm đầu

Likelihood GMM không phải hàm lõm theo toàn bộ tham số; nhiều cấu hình có thể tạo cực đại cục bộ khác nhau. Do đó EM phụ thuộc khởi tạo: các điểm bắt đầu khác nhau có thể dẫn đến phân cụm và likelihood cuối khác nhau. Khởi tạo thường dùng K-means, chọn ngẫu nhiên một số điểm làm mean, hoặc khởi tạo ngẫu nhiên nhiều lần rồi giữ nghiệm hợp lệ có likelihood tốt nhất. Các thành phần còn có thể đổi tên mà không đổi phân phối (label switching), nên hoán vị chỉ số cụm không phải nghiệm mô hình khác. [1, §9.3; 4]

Likelihood tăng đơn điệu không đồng nghĩa kết quả là cực đại toàn cục. Vì vậy nên thử nhiều khởi tạo, so sánh likelihood cuối trên cùng dữ liệu và cùng mô hình, đồng thời kiểm tra kích thước cụm, phương sai/covariance và tính diễn giải. [1, §9.4; 4]

## 9. Covariance suy biến và cách phòng tránh

Với covariance thành phần không bị ràng buộc, likelihood hỗn hợp Gaussian có thể không bị chặn trên: một Gaussian có thể đặt mean gần một điểm dữ liệu, làm covariance tiến về ma trận suy biến (trong 1D, variance tiến về 0). Mật độ của điểm đó tăng rất lớn trong khi thành phần khác vẫn giải thích các điểm còn lại. Vì thế một chuỗi EM có thể tiến tới nghiệm suy biến thay vì một mô hình hữu ích; công thức Gaussian không còn xác định khi covariance thực sự suy biến. [1, §9.3; 4]

Các biện pháp thực tế cần ghi nhận rõ vì chúng thay đổi bài toán hoặc giới hạn miền tham số:

1. **Regularize covariance:** dùng $\Sigma_k\leftarrow\Sigma_k+\lambda I$, $\lambda>0$, hoặc chặn eigenvalue tối thiểu. Đây là biện pháp ổn định số và ngăn collapse; vì đã thay đổi bước MLE thuần túy, không mặc định tính đơn điệu của EM chuẩn vẫn đúng nếu không chứng minh cho cách regularize cụ thể.
2. **Dùng covariance có cấu trúc:** ví dụ covariance chung, đường chéo, hoặc spherical nếu giả định ứng dụng phù hợp.
3. **Theo dõi thành phần nhỏ:** phát hiện $N_k$ quá nhỏ, covariance gần suy biến, hay mixing weight gần 0; khởi tạo lại hoặc loại thành phần theo một quy tắc được nêu rõ.
4. **Dùng prior/penalty hoặc phương pháp Bayes:** thay đổi mục tiêu so với MLE thuần túy và có thể hạn chế nghiệm biên.

Không được diễn giải “EM luôn tăng likelihood” thành “EM tránh suy biến”: likelihood có thể tăng trên đường đi tới biên suy biến. [4]

## 10. Danh mục tài liệu tham khảo

1. Christopher M. Bishop (2006). _Pattern Recognition and Machine Learning_. Springer, Ch. 9, đặc biệt §§9.2–9.4 (mixture distributions, EM cho GMM, và tính chất EM). [Bản sách/PDF](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/).
2. Andrew Ng. [CS229 Lecture Notes: Mixtures of Gaussians and the EM Algorithm](https://cs229.stanford.edu/summer2019/cs229-notes7b.pdf). Stanford University. Ghi chú trình bày biến ẩn, likelihood quan sát và các cập nhật GMM.
3. A. P. Dempster, N. M. Laird, D. B. Rubin (1977). “Maximum Likelihood from Incomplete Data Via the EM Algorithm.” _Journal of the Royal Statistical Society: Series B_, 39(1), 1–22. [doi:10.1111/j.2517-6161.1977.tb01600.x](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x).
4. Geoffrey J. McLachlan, Sharon X. Lee, and Suren I. Rathnayake (2019). “Finite Mixture Models.” _Annual Review of Statistics and Its Application_, 6, 355–378. [doi:10.1146/annurev-statistics-031017-100325](https://doi.org/10.1146/annurev-statistics-031017-100325). Tổng quan về local maxima, variance/covariance nhỏ và likelihood không bị chặn trong hỗn hợp Gaussian.
5. Kevin P. Murphy (2012). _Machine Learning: A Probabilistic Perspective_. MIT Press, Ch. 11, “Mixture models and the EM algorithm.” Tham khảo thêm cho mixture models, EM và các điểm tối ưu cục bộ.
