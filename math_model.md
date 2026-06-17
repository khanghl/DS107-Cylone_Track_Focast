# 🧮 Mô Hình Hóa Toán Học (Mathematical Modeling)
*Dự báo Quỹ đạo Bão sử dụng Khung học máy Đa phương thức (Multimodal ML Framework)*

Dựa trên nghiên cứu tham khảo từ bài báo **Hurricast (Boussioux et al., 2022)**, bài toán được mô hình hóa thành việc kết hợp dữ liệu bảng (Statistical) và dữ liệu không gian thời gian (Spatial-Temporal) thông qua kiến trúc Encoder-Decoder và Gradient Boosting.

---

## 1. Định nghĩa Không gian Dữ liệu (Input Space)
Tại mỗi mốc thời gian $t$ (của một cơn bão $i$), mô hình nhận hai luồng dữ liệu đầu vào trong một cửa sổ lịch sử $K$ bước (ví dụ $K=8$ tương ứng 24h quá khứ, mỗi bước cách nhau 3h).

### a. Dữ liệu Thống kê & Động học (Statistical Data)
Đại diện cho tọa độ, vận tốc, áp suất, độ ẩm, ENSO (các đặc trưng 1D hiện có):
$$ X_{stat}^{(t)} = [x_{t-K+1}, \dots, x_t] \in \mathbb{R}^{K \times F} $$
Với $F$ là số lượng đặc trưng bảng (hiện tại $F \approx 88$).

### b. Dữ liệu Bản đồ Khí tượng (Reanalysis Maps)
Thay vì trích xuất giá trị 1 điểm tại tâm bão, ta trích xuất một vùng không gian (Grid) xung quanh tâm bão:
$$ X_{viz}^{(t)} = [V_{t-K+1}, \dots, V_t] \in \mathbb{R}^{K \times C \times H \times W} $$
Trong đó:
- $H \times W$: Kích thước khung lưới bản đồ (ví dụ $25 \times 25$ pixels, tương ứng $25^\circ \times 25^\circ$ vĩ/kinh).
- $C$: Số lượng kênh ảnh (Channels). Lấy 3 thông số: Geopotential height ($z$), Zonal wind ($u$), Meridional wind ($v$) tại 3 mực áp suất (225, 500, 700 hPa) $\implies C = 9$.

---

## 2. Bài toán Dự đoán (Objective)
Mục tiêu là dự đoán hàm độ lệch (Displacement) của tọa độ trong chân trời dự báo $H$ (ví dụ +24h):
$$ Y = (\Delta Lat_{24h}, \Delta Lon_{24h}) \in \mathbb{R}^2 $$
$$ Position_{T+24} = Position_{T_0} + Y $$

---

## 3. Quá trình Trích xuất Đặc trưng (Feature Extraction)
### a. Spatial Encoder (CNN)
Tại mỗi bước thời gian $\tau \in [t-K+1, t]$, bản đồ $V_\tau$ được đưa qua Mạng nơ-ron Tích chập (CNN) để nén thông tin không gian:
$$ Z_{cnn}^{(\tau)} = \text{CNN}(V_\tau; \theta_{cnn}) \in \mathbb{R}^{D_{cnn}} $$
*(Đầu ra là một vector nén mang thông tin không gian vùng bão).*

### b. Temporal Decoder (GRU / Transformer)
Vector nén được ghép nối (concatenate) với dữ liệu bảng tại thời điểm đó, và đưa vào mạng chuỗi thời gian (RNN/GRU hoặc Transformer) để học sự phụ thuộc động lực học theo thời gian:
$$ I^{(\tau)} = Z_{cnn}^{(\tau)} \oplus X_{stat}^{(\tau)} $$
$$ H = \text{SequenceModel}\left( [I^{(t-K+1)}, \dots, I^{(t)}]; \theta_{seq} \right) \in \mathbb{R}^{D_{hidden}} $$

### c. Trích xuất Vision Embedding
Đầu ra của mô hình chuỗi $H$ đi qua các lớp Fully Connected (MLP) để dự đoán nháp. Ta rút trích vector ẩn từ lớp FC áp chót làm **Đặc trưng Nhúng (Vision Embedding)**:
$$ E = \text{MLP}_{\text{features}}(H) \in \mathbb{R}^{128} $$

**Hàm Loss để tối ưu khối Neural Network:**
$$ \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N || \hat{Y}_{nn}^{(i)} - Y^{(i)} ||_2^2 + \lambda ||\theta||_2^2 $$

---

## 4. Mô hình Lai - Multimodal Gradient Boosting
Sau khi huấn luyện thành công mạng Neural Network, trọng số $\theta$ được đóng băng (freeze). Quá trình dự báo chính thức được chuyển giao cho XGBoost / LightGBM.

### Augmented Feature Space
Vector đặc trưng mới đưa vào mô hình Boosting là sự kết hợp của $X_{stat}$ truyền thống và $E$ (thông tin 3D đã được nén):
$$ X_{final} = X_{stat}^{(t)} \oplus E \in \mathbb{R}^{F + 128} $$

### Thuật toán Boosting Cuối cùng
Mô hình Boosting học tập các tập luật rẽ nhánh (Decision Trees) để dự báo kết quả cuối:
$$ \hat{Y}_{final} = \text{XGBoost}(X_{final}) $$

---
**Tổng kết:** Toán học của hệ thống biến một bài toán Tabular thuần túy thành bài toán kết hợp Computer Vision & Time-series, giúp mô hình Boosting (vốn mù không gian) "nhìn" được bối cảnh 3D của các khối mây và hoàn lưu gió xung quanh tâm bão.
