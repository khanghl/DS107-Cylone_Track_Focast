# 🗺️ Kế Hoạch Triển Khai: Multimodal ML Framework (Vision + XGBoost)
*Mục tiêu: Đưa dữ liệu khí tượng ERA5 dưới dạng bản đồ lưới 3D (Grid Maps) kết hợp CNN/GNN với XGBoost/LightGBM để khắc phục nhược điểm của Regressor Chain.*

Đây là một sự thay đổi LỚN về mặt kiến trúc (Architecture Shift), chuyển từ Tabular Data thuần túy sang **Multimodal (Computer Vision + Tabular)**. Dưới đây là kế hoạch chi tiết từng bước.

---

## Giai đoạn 1: Chuẩn bị & Crawl Data Không gian (Spatial Data)
Hiện tại, code `_03_ERA5_clean.py` của bạn chỉ trích xuất giá trị 1 điểm (Point Extraction) tại tâm bão. Để dùng CNN, ta phải trích xuất một "Bức ảnh" (Grid Map).

- **Bước 1.1:** Viết script mới `_03_ERA5_grid_extractor.py`.
  - Thay vì lấy 1 điểm `(lat, lon)`, hệ thống sẽ cắt một khung bounding box `25x25` độ xung quanh tâm bão.
  - Các biến cần lấy: `z` (Geopotential height), `u` (Zonal wind), `v` (Meridional wind).
  - Các mực áp suất: `225hPa`, `500hPa`, `700hPa`.
  - Tổng cộng: Mỗi mốc thời gian sẽ có 1 Tensor kích thước `9 x 25 x 25`.
- **Bước 1.2:** Lưu trữ dữ liệu.
  - Do dữ liệu ảnh rất nặng, không thể lưu chung vào CSV. 
  - Lưu thành các file `numpy` (`.npy`) hoặc `.h5` cho từng cơn bão (chia theo thư mục `SID`), ánh xạ khớp với index trong `features.csv`.

---

## Giai đoạn 2: Xây dựng Pipeline Deep Learning (PyTorch)
- **Bước 2.1:** Viết Pytorch `Dataset` và `DataLoader` trong thư mục `src/data_loader/dl_loader.py`.
  - Load chuỗi 8 bước thời gian (24h lịch sử).
  - Input: Tensor kích thước `(8, 9, 25, 25)` và Vector thống kê kích thước `(8, F)`.
- **Bước 2.2:** Xây dựng khối CNN-Encoder (`src/models/vision_encoder.py`).
  - 3 lớp Conv2D + Batch Norm + ReLU + MaxPool.
  - Flatten về một vector 1D.
- **Bước 2.3:** Xây dựng khối Temporal-Decoder (GRU/Transformer).
  - Nối vector 1D của CNN với Vector Thống kê.
  - Đưa chuỗi 8 bước vào mạng GRU để tổng hợp chuỗi thời gian.
  - Lớp Linear cuối cùng đẩy ra 2 nốt (dự đoán $\Delta Lat, \Delta Lon$).

---

## Giai đoạn 3: Huấn luyện & Trích xuất (Training & Embedding)
- **Bước 3.1:** Train mạng End-to-End.
  - Viết file `train_dl.py`. Train mạng CNN+GRU với hàm Loss là MSE của $\Delta Lat, \Delta Lon$.
  - Lưu weights tốt nhất `.pth`.
- **Bước 3.2:** Trích xuất Đặc trưng Nhúng (Vision Embeddings).
  - Viết file `extract_embeddings.py`.
  - Load model `pth`, đóng băng trọng số (freeze).
  - Chạy toàn bộ tập Train/Val/Test qua model, lấy output của lớp Dense áp chót (Kích thước 128).
  - Gộp 128 cột mới này vào `features.csv` (Tạo thành `features_augmented.csv`).

---

## Giai đoạn 4: Tích hợp Boosting & So sánh kết quả
- **Bước 4.1:** Retrain các mô hình truyền thống (`src/pipeline/compare.py`).
  - Sử dụng `features_augmented.csv` để train lại XGBoost và LightGBM (Independent).
  - Chạy code tính DPE và lập bảng so sánh. 
  - *Mục tiêu mong đợi:* Việc XGBoost nhìn thấy "Vision Embedding" sẽ giảm DPE 24h xuống đáng kể (do đã hiểu được luồng gió xung quanh thay vì chỉ 1 điểm tâm bão).

---

## Giai đoạn 5: Cập nhật Web API và README
- **Bước 5.1:** Cập nhật `api/main.py` để hỗ trợ Multimodal.
  - Luồng `/predict`: Backend sẽ phải load lưới dữ liệu 3D $\to$ Đưa qua file `.pth` PyTorch để lấy Embedding 128D $\to$ Gắn vào vector tabular $\to$ Đưa vào `.pkl` của LightGBM.
- **Bước 5.2:** Viết lại toàn bộ `README.md` để "khoe" với giảng viên kiến trúc **Multimodal AI** (Sự kết hợp giữa Deep Learning và Gradient Boosting).

---

## ❓ Câu Hỏi Chờ Xác Nhận (Phản hồi để tiếp tục)
Việc crawl lại dữ liệu ERA5 dạng lưới 25x25 (Giai đoạn 1) có thể tốn khá nhiều dung lượng ổ cứng và thời gian chạy script trích xuất. Đồng thời hệ thống cần cài thêm thư viện `torch`. 
**Bạn có đồng ý triển khai toàn bộ lộ trình này trên máy của bạn ngay bây giờ không? Nếu OK, mình sẽ viết lại `README.md` trước để bạn duyệt, sau đó bắt tay vào viết code.**
