# Cyclone Track Forecast - Dự báo quỹ đạo & cường độ bão bằng Học máy Đa phương thức

> Bài toán để dự báo **đồng thời** quỹ đạo và cường độ của bão nhiệt đới tại khu vực Tây Bắc Thái Bình Dương (WNP). Khác với các phương pháp truyền thống, hệ thống này tiên phong áp dụng kiến trúc **Multimodal (Đa phương thức)** kết hợp với **Stacking Ensemble**, sau đó hiển thị trực quan thông qua một ứng dụng Web tương tác (Interactive Web App).

## GVHD:
TS. Nguyễn Văn Kiệt

CN. Trần Quốc Khánh

## Nhóm Sinh Viên Thực Hiện:
1. Phạm Đình Quang Huy - MSSV: 24520689 
2. Huỳnh Lâm Bảo Khang - MSSV: 24520738
3. Phạm Ngọc Minh - MSSV: 24521082
4. Nguyễn Thị Kim Ngân - MSSV: 24521130

---

## Mục Lục

- [Cyclone Track Forecast - Dự báo quỹ đạo \& cường độ bão bằng Học máy Đa phương thức](#cyclone-track-forecast---dự-báo-quỹ-đạo--cường-độ-bão-bằng-học-máy-đa-phương-thức)
  - [GVHD:](#gvhd)
  - [Nhóm Sinh Viên Thực Hiện:](#nhóm-sinh-viên-thực-hiện)
  - [Mục Lục](#mục-lục)
  - [Tổng Quan Bài Toán \& Giải Pháp](#tổng-quan-bài-toán--giải-pháp)
    - [Định nghĩa Bài Toán](#định-nghĩa-bài-toán)
  - [Kiến Trúc Mô Hình (Multimodal Stacking)](#kiến-trúc-mô-hình-multimodal-stacking)
  - [Hệ Thống Web App Trực Quan](#hệ-thống-web-app-trực-quan)
  - [Cấu Trúc Thư Mục](#cấu-trúc-thư-mục)
  - [Hướng Dẫn Chạy Toàn Bộ Hệ Thống](#hướng-dẫn-chạy-toàn-bộ-hệ-thống)
    - [1. Huấn Luyện \& So Sánh Mô Hình (ML Pipeline)](#1-huấn-luyện--so-sánh-mô-hình-ml-pipeline)
    - [2. Khởi Động API Server (Backend)](#2-khởi-động-api-server-backend)
    - [3. Mở Giao Diện Tương Tác Web (Frontend)](#3-mở-giao-diện-tương-tác-web-frontend)

---

## Tổng Quan Bài Toán & Giải Pháp

Dự án này chuyển đổi từ một framework học máy cơ bản ở đồ án trước thành hệ thống **Multimodal Learning, Multi-task & Stacking Ensemble** hoàn chỉnh:
1. **Dự báo Đa Mục Tiêu (Multi-task):** Không chỉ dự báo hướng đi (Track), mô hình nay đã dự báo song song cả Cường độ bão (Intensity - sức gió) với các khung thời gian tương lai: +6h, +12h, +24h.
2. **Deep Learning Core:** Trích xuất đặc trưng không gian (Vision Embedding) từ ảnh khí tượng vệ tinh 3D ERA5 thông qua mạng PyTorch **CNN+GRU**.
3. **Stacking Ensemble:** Thay vì chỉ sử dụng một mô hình ML, hệ thống ứng dụng kỹ thuật **Stacking** mạnh mẽ (kết hợp LightGBM và XGBoost, blend bằng Ridge Regression) để tối ưu hóa chống Overfitting và giảm thiểu mạnh mẽ sai số dự đoán.
4. **Hệ sinh thái:** Bao gồm Pipeline xử lý/huấn luyện mô hình chuẩn hóa với Rich Terminal, Backend API FastAPI và Frontend Web UI Interactive mượt mà.

### Định nghĩa Bài Toán 

Cho trạng thái không gian-thời gian hiện tại của bão $\mathcal{S}_t$ gồm dữ liệu bảng $\mathcal{X}_t \in \mathbb{R}^F$ (kinh/vĩ độ, áp suất, vận tốc) và dữ liệu lưới ảnh khí quyển $\mathcal{V}_t \in \mathbb{R}^{C \times H \times W}$ (với kích thước lưới $25^\circ \times 25^\circ$). 

Ta cần dự báo **Vector Chuyển vị** và **Thay đổi Cường độ**:
$$ \mathbf{Y}_{t+h} = \left[ \Delta \text{Lat}_{h}, \Delta \text{Lon}_{h}, \Delta \text{Wind}_{h} \right]^T $$
Với chân trời dự báo $h \in \{6, 12, 24\}$ giờ.

---

## Kiến Trúc Mô Hình (Multimodal Stacking)

Hệ thống được thiết kế theo luồng **Late-Fusion kết hợp Stacking Ensemble**:

1. **Spatial-Temporal Encoder (PyTorch):** 
   - Đầu vào là chuỗi ảnh không gian (Geopotential, U-Wind, V-Wind ở các tầng áp suất) cắt xung quanh tâm bão.
   - Mạng **CNN** 3 lớp trích xuất đặc trưng không gian, kết hợp với **GRU** để tổng hợp chuỗi thời gian quá khứ. Đầu ra là Vector tiềm ẩn $Z_t \in \mathbb{R}^{128}$ (Vision Embedding).

2. **Multitask Stacking Predictor:**
   - Hợp nhất $Z_t$ (Vision Embeddings) với dữ liệu tabular truyền thống.
   - Lớp Base Learners: Sử dụng sức mạnh kết hợp của `MultiOutputRegressor(LGBMRegressor)` và `MultiOutputRegressor(XGBRegressor)`.
   - Lớp Meta Learner: Trọng số hóa kết quả từ hai mô hình Boosting thông qua `Ridge(alpha=100.0)`. Giúp giảm mức sai số vị trí **DPE 24h xuống xấp xỉ ~127km**.

---

## Hệ Thống Web App Trực Quan 

Nhằm dễ dàng trực quan hóa các kết quả khô khan từ mô hình, một Web Frontend hiện đại (`HTML/CSS/VanillaJS`) tương tác trực tiếp với Backend FastAPI đã được phát triển với các tính năng ấn tượng:

- **Bản đồ Google Maps (Tiếng Việt)**: Hiển thị mượt mà bản đồ Leaflet có tính năng di chuyển camera (pan) và thu phóng (zoom) theo hành trình thực của cơn bão.
- **Animation Quỹ Đạo**: Mô phỏng sự di chuyển của bão trong quá khứ bằng hiệu ứng vẽ quỹ đạo theo thời gian thực tế, với icon bão (🌀) xoay tròn sống động ở trung tâm.
- **Cường Độ Bão Trực Quan (Chart.js)**: Biểu đồ đường so sánh trực tiếp sức gió thực tế (Observed) và sức gió do mô hình dự đoán (Predicted).
- **So Sánh Quỹ Đạo & Sai Số (DPE)**: Vẽ song song Quỹ đạo thực tế tương lai (đường đen) và Quỹ đạo dự đoán (+6h, +12h, +24h), đi kèm với các **Vòng tròn sai số DPE (Distance Position Error)** bao quanh điểm dự đoán để đánh giá khách quan độ chính xác.


---

## Cấu Trúc Thư Mục

```text
DS107-Cylone_Track_Focast/
├── api/                              # FastAPI Backend Server (API phục vụ web)
│   ├── main.py
│   └── models/                       # Chứa best_model.pkl và test_data.csv
├── experiments/                      # Thử nghiệm & Code visualize biểu đồ tĩnh (Matplotlib)
├── src/                              # Mã nguồn Lõi AI
│   ├── data_loader/                  # Tải DL gốc & Tạo Dataset PyTorch (ERA5)
│   ├── module/                       # Các khối kiến trúc ML/DL (CNN+GRU, LightGBM, XGBoost, Stacking)
│   └── pipeline/                     # Các bước huấn luyện (Train DL -> Extract Emb -> Tune LGBM -> Compare -> Export)
└── web/                              # Giao diện Frontend
    ├── app.js                        # Logic xử lý API, Animation Leaflet & Chart
    ├── index.html                    # Layout HTML chính
    └── style.css                     # Stylesheet & Glassmorphism design
```

---

## Hướng Dẫn Chạy Toàn Bộ Hệ Thống

### 1. Huấn Luyện & So Sánh Mô Hình (ML Pipeline)
Nếu bạn muốn tự chạy lại hệ thống từ đầu (Giả sử đã chạy các bước crawl ERA5 Data):
```bash
# 1. Huấn luyện Deep Learning Encoder
python src/pipeline/_01_train_dl.py

# 2. Rút trích Vision Embeddings
python src/pipeline/_02_extract_embeddings.py

# 3. Tuning Hyperparameters cho XGBoost và LightGBM
python src/pipeline/_05_tune_lgbm.py

# 4. Huấn luyện, đánh giá Stacking Ensemble (In ra bảng so sánh chi tiết bằng Rich)
python src/pipeline/_04_compare.py

# 5. Xuất mô hình tốt nhất sang thư mục API
python src/pipeline/export_model.py
```

### 2. Khởi Động API Server (Backend)
Bật máy chủ dự đoán API dùng FastAPI. Mở một terminal mới và chạy:
```bash
uvicorn api.main:app --reload
```
*(Server sẽ chạy tại: `http://127.0.0.1:8000`)*

### 3. Mở Giao Diện Tương Tác Web (Frontend)
Mở một terminal mới khác, di chuyển vào thư mục `web` và khởi tạo local server bằng Python:
```bash
cd web
python -m http.server 8080
```
Sau đó mở trình duyệt và truy cập: 👉 **[http://localhost:8080](http://localhost:8080)**

Trải nghiệm hệ thống web tương tác 100% Client-Side liên kết với mô hình AI Backend ngay lập tức!
