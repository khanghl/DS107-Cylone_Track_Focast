# 🌀 Dự báo quỹ đạo & cường độ bão nhiệt đới bằng AI Đa phương thức (Multimodal ML)

> Một hệ thống Trí tuệ Nhân tạo chuyên biệt để dự báo **đồng thời** quỹ đạo và cường độ của xoáy thuận nhiệt đới tại khu vực Tây Bắc Thái Bình Dương (WNP). Khác với các phương pháp truyền thống, hệ thống này tiên phong áp dụng kiến trúc **Multimodal (Đa phương thức)**: kết hợp dữ liệu bảng (Tabular chuỗi thời gian) và dữ liệu không gian lưới 3D (ERA5 Spatial Grids) thông qua mạng Deep Learning **(CNN + GRU)**, sau đó hợp nhất (Fusion) với thuật toán **XGBoost** để đưa ra dự đoán cuối cùng. Đi kèm là hệ thống Web **(FastAPI + Vite/Leaflet)** mô phỏng kết quả trực quan trên bản đồ.

## GVHD:
PSG.TS.Nguyễn Lưu Thùy Ngân

TS. Dương Ngọc Hảo

## Thành viên:
1. Phạm Đình Quang Huy - MSSV: 24520689 

2. Huỳnh Lâm Bảo Khang - MSSV: 24520738

---

## 📋 Mục Lục

- [Tổng Quan Bài Toán](#tổng-quan-bài-toán)
- [Kiến trúc Đa Phương Thức (Multimodal Architecture)](#kiến-trúc-đa-phương-thức-multimodal-architecture)
- [Hệ Thống Web App Trực Quan](#hệ-thống-web-app-trực-quan)
- [Hướng Dẫn Cài Đặt & Chạy Mô Hình Khai Phá](#hướng-dẫn-cài-đặt--chạy-mô-hình-khai-phá)
- [Hiệu Suất & Đánh Giá](#hiệu-suất--đánh-giá)

---

## Tổng Quan Bài Toán

Dự án này là phiên bản nâng cấp toàn diện dựa trên nền tảng của bài báo Hurricast, chuyển đổi từ một framework học máy cơ bản thành hệ thống **Multimodal Learning & Multi-task** hoàn chỉnh:
1. **Dự báo Đa Mục Tiêu (Multi-task):** Không chỉ dự báo hướng đi (Track), mô hình nay đã dự báo song song cả Cường độ bão (Intensity - sức gió).
2. **Deep Learning Core:** Trích xuất đặc trưng không gian (Vision Embedding) từ ảnh khí tượng vệ tinh 3D ERA5 thông qua mạng PyTorch CNN+GRU.
3. **Machine Learning Fusion:** Đưa Vision Embeddings vào huấn luyện mô hình XGBoost để tận dụng sức mạnh chống Overfitting và tốc độ suy luận của Gradient Boosting.
4. **Hệ sinh thái:** Bao gồm Pipeline xử lý dữ liệu lớn (CDS API), Backend FastAPI và Frontend UI hiện đại.

### Phát Biểu Bài Toán Học Máy (Toán Học)

Cho trạng thái không gian-thời gian hiện tại của bão $\mathcal{S}_t$ gồm dữ liệu bảng $\mathcal{X}_t \in \mathbb{R}^F$ và dữ liệu lưới ảnh khí quyển $\mathcal{V}_t \in \mathbb{R}^{C \times H \times W}$ (với kích thước lưới $25^\circ \times 25^\circ$). 

Ta cần dự báo **Vector Chuyển vị** và **Thay đổi Cường độ**:
$$ \mathbf{Y}_{t+h} = \left[ \Delta \text{Lat}_{h}, \Delta \text{Lon}_{h}, \Delta \text{Wind}_{h} \right]^T $$
Với chân trời dự báo $h \in \{6, 12, 24\}$ giờ.

---

## Kiến trúc Đa Phương Thức (Multimodal Architecture)

Hệ thống được thiết kế theo luồng **Late-Fusion**:

1. **Spatial-Temporal Encoder (PyTorch):** 
   - Đầu vào là chuỗi ảnh không gian (Geopotential, U-Wind, V-Wind ở các tầng áp suất) cắt xung quanh tâm bão kích thước `(B, 8, 5, 25, 25)`.
   - Mạng **CNN** 3 lớp trích xuất đặc trưng không gian, sau đó kết nối với **GRU** để tổng hợp chuỗi thời gian 8 mốc quá khứ.
   - Đầu ra là Vector tiềm ẩn $Z_t \in \mathbb{R}^{128}$ (Vision Embedding).

2. **Multitask Boosting Predictor:**
   - Hợp nhất $Z_t$ với dữ liệu IBTrACS truyền thống (Vận tốc, hướng di chuyển, ENSO).
   - Sử dụng **MultiOutputRegressor(XGBoost)** để dự đoán đồng thời 3 nhãn đích (Lat, Lon, Wind).

---

## Cấu Trúc Thư Mục Cốt Lõi

```text
Cyclone_Track_Forecast/
├── api/                              # Backend Server (FastAPI)
├── web/                              # Frontend (Vite + Leaflet Map)
├── data/                             # Quản lý Dữ liệu
│   └── raw/                          # Script tải ERA5 từ Copernicus
└── src/                              # Mã nguồn Lõi AI
    ├── data_loader/                  
    │   ├── crawl_era5_pressure.py    # Script tải dữ liệu ERA5 3D
    │   ├── _03_ERA5_grid_extractor.py# Tách lưới 25x25 độ thành file .npy
    │   └── dl_loader.py              # PyTorch Dataset Loader
    ├── models/                       
    │   └── vision_encoder.py         # Kiến trúc CNN + GRU PyTorch
    └── pipeline/                     
        ├── train_dl.py               # Huấn luyện mạng Deep Learning
        ├── extract_embeddings.py     # Trích xuất 128D Embeddings
        ├── multimodal.py             # Chiến lược XGBoost Multi-task
        └── export_model.py           # Xuất mô hình phục vụ Web API
```

---

## Hệ Thống Web App Trực Quan 

1. **Hiển Thị Cường Độ Bão:** Bổ sung tính năng mô phỏng sức gió (Knots) trong tương lai. Các popup tại các mốc +6h, +12h, +24h sẽ dự báo chính xác sức tàn phá của bão.
2. **Cone of Uncertainty (Vùng Sai Số):** Các vòng tròn đứt nét màu đỏ bao quanh điểm dự báo, nở ra tương ứng với độ bất định ở mỗi mốc thời gian (35km, 61km, 145km).
3. **Hiệu Ứng Tâm Bão (Storm Spinner):** Cập nhật CSS Animation mô phỏng tâm bão quay vòng kết hợp hiệu ứng đập nhịp (Pulsate) ngay trên bản đồ Dark Mode.

---

## Hướng Dẫn Cài Đặt & Chạy Mô Hình Khai Phá

### Bước 1: Thu Thập Dữ Liệu Lưới Không Gian (Tùy Chọn / GPU Khuyên Dùng)
Do dữ liệu ảnh ERA5 rất nặng, bạn cần có tài khoản CDS API và tải về máy chủ (Colab hoặc máy có GPU mạnh):
```bash
python src/data_loader/crawl_era5_pressure.py
python src/data_loader/_03_ERA5_grid_extractor.py
```

### Bước 2: Huấn luyện Mạng Deep Learning
```bash
# Cài đặt PyTorch trước
python src/pipeline/train_dl.py
python src/pipeline/extract_embeddings.py
```
> *Lưu ý: Nếu không chạy bước này, hệ thống sẽ tự động dùng dữ liệu truyền thống (features.csv) để đảm bảo không bị gián đoạn tính năng.*

### Bước 3: Xuất Mô hình Triển Khai
```bash
python src/pipeline/export_model.py
```
> Mô hình XGBoost đa mục tiêu sẽ được đóng gói tại `api/models/best_model.pkl`.

### Bước 4: Khởi Động Hệ Thống Realtime (Web)
Chạy FastAPI Backend:
```bash
uvicorn api.main:app --reload --port 8000
```
Chạy Vite Frontend (Tab mới):
```bash
cd web && npm run dev
```

---

## Đóng Góp & Liên Hệ

Đồ án được nâng cấp từ Baseline lên kiến trúc Deep Learning tiên tiến.
**Cập nhật lần cuối:** 15/06/2026
**Trạng thái:** ✅ Sẵn sàng Báo Cáo.
