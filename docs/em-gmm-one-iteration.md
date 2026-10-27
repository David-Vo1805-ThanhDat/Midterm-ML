# Ví dụ tính tay một vòng E-step và M-step cho GMM 1D

Tài liệu này thực hiện trọn một vòng EM cho dữ liệu một chiều với hai Gaussian. Các phép tính làm tròn đến 6 chữ số thập phân; khi cộng các số đã làm tròn có thể có sai khác rất nhỏ.

## 1. Dữ liệu và khởi tạo

Chọn ba điểm dữ liệu:

$$
x_1=0,\qquad x_2=2,\qquad x_3=5;\qquad N=3,\quad K=2.
$$

Tham số ban đầu:

| Thành phần $k$ | $\pi_k^{(0)}$ | $\mu_k^{(0)}$ | $(\sigma_k^2)^{(0)}$ |
|---:|---:|---:|---:|
| 1 | 0.5 | 1 | 1 |
| 2 | 0.5 | 4 | 1 |

Mỗi Gaussian 1D có mật độ

$$
\mathcal N(x\mid\mu,\sigma^2)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

Vì cả hai phương sai bằng 1, hệ số chung là $1/\sqrt{2\pi}\approx0.398942$. Trong E-step, tính mật độ dưới từng Gaussian rồi nhân với mixing weight $\pi_k$.

## 2. E-step: tính density và responsibility

Responsibility của cụm $k$ cho điểm $x_n$ là

$$
r_{nk}=\frac{\pi_k^{(0)}\mathcal N(x_n\mid\mu_k^{(0)},(\sigma_k^2)^{(0)})}
{\sum_{j=1}^{2}\pi_j^{(0)}\mathcal N(x_n\mid\mu_j^{(0)},(\sigma_j^2)^{(0)})}.
$$

Ví dụ cho $x_1=0$:

$$
\mathcal N(0\mid1,1)=0.241971,\qquad
\mathcal N(0\mid4,1)=0.000134.
$$

Do hai trọng số đều là 0.5:

$$
r_{11}=\frac{0.5(0.241971)}{0.5(0.241971)+0.5(0.000134)}
\approx0.999447,\qquad r_{12}=1-r_{11}\approx0.000553.
$$

Bảng đầy đủ (density là mật độ Gaussian chưa nhân trọng số):

| Điểm $x_n$ | $\mathcal N(x_n\mid1,1)$ | $\mathcal N(x_n\mid4,1)$ | $r_{n1}$ | $r_{n2}$ | Tổng trách nhiệm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.241971 | 0.000134 | 0.999447 | 0.000553 | 1.000000 |
| 2 | 0.241971 | 0.053991 | 0.817574 | 0.182426 | 1.000000 |
| 5 | 0.000134 | 0.241971 | 0.000553 | 0.999447 | 1.000000 |

Vì $\pi_1=\pi_2$, các trọng số triệt tiêu trong tỷ số; mỗi responsibility tỷ lệ với density tương ứng. Mỗi hàng chuẩn hóa về 1, như xác suất hậu nghiệm phải có.

## 3. M-step: cập nhật tham số

### 3.1 Số lượng hiệu dụng và mixing weights

$$
N_1=\sum_{n=1}^{3}r_{n1}
=0.999447+0.817574+0.000553=1.817574,
$$

$$
N_2=\sum_{n=1}^{3}r_{n2}
=0.000553+0.182426+0.999447=1.182426.
$$

Kiểm tra: $N_1+N_2=3=N$ (sai khác ở chữ số cuối có thể do làm tròn). Trọng số mới là $\pi_k^{(1)}=N_k/N$:

$$
\pi_1^{(1)}=1.817574/3\approx0.605858,\qquad
\pi_2^{(1)}=1.182426/3\approx0.394142.
$$

Hai trọng số mới cộng lại thành 1.

### 3.2 Trung bình mới

$$
\mu_k^{(1)}=\frac{\sum_n r_{nk}x_n}{N_k}.
$$

Với thành phần 1:

$$
\mu_1^{(1)}=
\frac{0.999447(0)+0.817574(2)+0.000553(5)}{1.817574}
=\frac{1.637913}{1.817574}\approx0.901153.
$$

Với thành phần 2:

$$
\mu_2^{(1)}=
\frac{0.000553(0)+0.182426(2)+0.999447(5)}{1.182426}
=\frac{5.362087}{1.182426}\approx4.534820.
$$

### 3.3 Phương sai mới

Dùng trung bình **mới** trong công thức M-step:

$$
(\sigma_k^2)^{(1)}=
\frac{\sum_n r_{nk}(x_n-\mu_k^{(1)})^2}{N_k}.
$$

Thành phần 1:

$$
(\sigma_1^2)^{(1)}=
\frac{0.999447(0-0.901153)^2+0.817574(2-0.901153)^2+0.000553(5-0.901153)^2}
{1.817574}
\approx0.994791.
$$

Thành phần 2:

$$
(\sigma_2^2)^{(1)}=
\frac{0.000553(0-4.534820)^2+0.182426(2-4.534820)^2+0.999447(5-4.534820)^2}
{1.182426}
\approx1.183822.
$$

## 4. Log-likelihood trước và sau vòng lặp

Với tham số $\theta$, log-likelihood quan sát là

$$
\ell(\theta)=\sum_{n=1}^{3}\log\left[
\sum_{k=1}^{2}\pi_k\mathcal N(x_n\mid\mu_k,\sigma_k^2)
\right].
$$

### Trước E/M-step

Theo tham số ban đầu $(\pi_1,\pi_2)=(0.5,0.5)$, means $(1,4)$, variances $(1,1)$:

| Điểm | Mật độ hỗn hợp $p(x_n)$ | $\log p(x_n)$ |
|---:|---:|---:|
| 0 | 0.121052 | -2.111533 |
| 2 | 0.147981 | -1.910672 |
| 5 | 0.121052 | -2.111533 |

$$
\ell^{(0)}=-2.111533-1.910672-2.111533\approx-6.133738.
$$

### Sau một vòng E-step/M-step

Với $\pi^{(1)}=(0.605858,0.394142)$, $\mu^{(1)}=(0.901153,4.534820)$, và $(\sigma^2)^{(1)}=(0.994791,1.183822)$:

| Điểm | $\log p(x_n\mid\theta^{(1)})$ |
|---:|---:|
| 0 | -1.825450 |
| 2 | -1.954315 |
| 5 | -2.025358 |

$$
\ell^{(1)}\approx-1.825450-1.954315-2.025358=-5.805122.
$$

Log-likelihood tăng khoảng

$$
\ell^{(1)}-\ell^{(0)}\approx0.328616>0,
$$

phù hợp với tính chất không giảm của một vòng EM chuẩn. Giá trị log-likelihood là tổng log mật độ; nó có thể âm mà vẫn hợp lệ vì mật độ xác suất không bị giới hạn ở mức tối đa 1.

## 5. Tóm tắt tham số

| Thành phần | $\pi_k^{(0)}$ | $\mu_k^{(0)}$ | $(\sigma_k^2)^{(0)}$ | $N_k$ | $\pi_k^{(1)}$ | $\mu_k^{(1)}$ | $(\sigma_k^2)^{(1)}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.500000 | 1.000000 | 1.000000 | 1.817574 | 0.605858 | 0.901153 | 0.994791 |
| 2 | 0.500000 | 4.000000 | 1.000000 | 1.182426 | 0.394142 | 4.534820 | 1.183822 |

## 6. Đối chiếu số học và nguồn

- Mỗi điểm có $r_{n1}+r_{n2}=1$; tổng $N_1+N_2=N=3$; tổng mixing weights mới bằng 1.
- Phương sai dùng $N_k$ ở mẫu số (MLE) và trung bình mới, không dùng mẫu số hiệu chỉnh $N_k-1$.
- Các kết quả trung gian được tính từ công thức Gaussian 1D và cập nhật EM, sau đó làm tròn 6 chữ số. Sai số ở các tổng đã làm tròn thường dưới $10^{-5}$; các giá trị chưa làm tròn cho $\ell^{(1)}-\ell^{(0)}\approx0.3286161612$.
- Công thức GMM và cập nhật EM: Christopher M. Bishop, *Pattern Recognition and Machine Learning* (2006), §§9.2–9.3; Andrew Ng, [CS229 notes: Mixtures of Gaussians and the EM Algorithm](https://cs229.stanford.edu/summer2019/cs229-notes7b.pdf).
- Tính chất likelihood không giảm: A. P. Dempster, N. M. Laird, D. B. Rubin (1977), “[Maximum Likelihood from Incomplete Data Via the EM Algorithm](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x),” *JRSS B*, 39(1), 1–22.
