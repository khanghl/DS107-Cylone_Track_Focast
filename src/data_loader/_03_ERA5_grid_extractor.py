import pandas as pd
import numpy as np
import os
import rasterio
from pathlib import Path
from scipy.ndimage import zoom

# --- 1. CẤU HÌNH ĐƯỜNG DẪN ---
KAGGLE_INPUT_DIR = Path('/kaggle/input')

# LƯU Ý QUAN TRỌNG: Hãy đổi 'YOUR_CSV_DATASET_NAME' thành tên thư mục chứa file dynamic_feature.csv của bạn
INPUT_CSV = KAGGLE_INPUT_DIR / 'YOUR_CSV_DATASET_NAME' / 'dynamic_feature.csv' 

# Thư mục chứa toàn bộ dữ liệu lưới ERA5 (2015-2025)
GRIB_DIR = KAGGLE_INPUT_DIR / 'datasets/phmhuy0511/era5-pressure'

# Thư mục lưu kết quả đầu ra
OUTPUT_GRID_DIR = Path('/kaggle/working/vision_grids')
os.makedirs(OUTPUT_GRID_DIR, exist_ok=True)

# --- 2. HÀM TRÍCH XUẤT (Có 3 tham số truyền vào) ---
def extract_grids(csv_path, grib_path, output_dir):
    """
    Trích xuất ma trận không gian từ file GRIB.
    Tự động nối mảng (concatenate) nếu dữ liệu của cơn bão đã tồn tại.
    """
    df = pd.read_csv(csv_path)
    df['DATETIME'] = pd.to_datetime(df['ISO_TIME'])
    
    print(f"\n--- Processing GRIB: {grib_path.name} ---")
    
    with rasterio.open(grib_path) as src:
        bounds = src.bounds
        height, width = src.shape
        
        sids = df['SID'].unique()
        total_storms = len(sids)
        
        for storm_idx, sid in enumerate(sids):
            storm_data = df[df['SID'] == sid].sort_values('DATETIME')
            
            storm_grids = []
            valid_indices = []
            
            for idx, row in storm_data.iterrows():
                lat = row['LAT']
                lon = row['LON']
                
                # Bounding box 25 độ
                lat_min, lat_max = lat - 12.5, lat + 12.5
                lon_min, lon_max = lon - 12.5, lon + 12.5
                
                row_min = int((bounds.top - lat_max) / (bounds.top - bounds.bottom) * height)
                row_max = int((bounds.top - lat_min) / (bounds.top - bounds.bottom) * height)
                col_min = int((lon_min - bounds.left) / (bounds.right - bounds.left) * width)
                col_max = int((lon_max - bounds.left) / (bounds.right - bounds.left) * width)
                
                row_min = max(0, min(row_min, height - 1))
                row_max = max(0, min(row_max, height - 1))
                col_min = max(0, min(col_min, width - 1))
                col_max = max(0, min(col_max, width - 1))
                
                if row_min > row_max: row_min, row_max = row_max, row_min
                if col_min > col_max: col_min, col_max = col_max, col_min
                
                try:
                    band_indices = [1, min(500, src.count), min(1000, src.count)]
                    grids = []
                    
                    for band in band_indices:
                        window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
                        try:
                            grid = src.read(band, window=window)
                            grids.append(grid)
                        except:
                            continue
                    
                    if len(grids) > 0:
                        grid_arr = np.array(grids)
                        current_shape = grid_arr[0].shape
                        
                        if current_shape[0] > 0 and current_shape[1] > 0:
                            scale_factors = (25.0 / current_shape[0], 25.0 / current_shape[1])
                            grid_resized = zoom(grid_arr, (1, scale_factors[0], scale_factors[1]), order=1)
                            grid_resized = grid_resized[:, :25, :25]
                            
                            storm_grids.append(grid_resized)
                            valid_indices.append(idx)
                            
                except Exception:
                    continue
            
            if len(storm_grids) > 0:
                storm_grids = np.stack(storm_grids)
                valid_indices = np.array(valid_indices)
                
                grids_file = output_dir / f"{sid}_grids.npy"
                indices_file = output_dir / f"{sid}_indices.npy"
                
                if grids_file.exists() and indices_file.exists():
                    existing_grids = np.load(grids_file)
                    existing_indices = np.load(indices_file)
                    
                    combined_grids = np.concatenate([existing_grids, storm_grids])
                    combined_indices = np.concatenate([existing_indices, valid_indices])
                    
                    sort_idx = np.argsort(combined_indices)
                    combined_grids = combined_grids[sort_idx]
                    combined_indices = combined_indices[sort_idx]
                    
                    np.save(grids_file, combined_grids)
                    np.save(indices_file, combined_indices)
                    total_len = len(combined_indices)
                else:
                    np.save(grids_file, storm_grids)
                    np.save(indices_file, valid_indices)
                    total_len = len(valid_indices)
                    
                print(f"  ✓ {sid}: Đã lưu {len(storm_grids)} mốc thời gian (Tổng hiện tại: {total_len})")

# --- 3. HÀM QUÉT VÀ CHẠY HÀNG LOẠT ---
def process_all_years():
    """Hàm quét và chạy tự động tất cả các file trong thư mục"""
    print(f"Đang quét thư mục: {GRIB_DIR}")
    
    grib_files = list(GRIB_DIR.glob('*.grib')) + list(GRIB_DIR.glob('*.grid'))
    grib_files.sort()
    
    if not grib_files:
        print("Lỗi: Không tìm thấy tệp GRIB/GRID nào. Vui lòng kiểm tra lại đường dẫn dataset.")
        return
        
    print(f"Đã tìm thấy {len(grib_files)} tệp. Bắt đầu xử lý hàng loạt...\n")
    
    for grib_path in grib_files:
        # Gọi hàm với ĐÚNG 3 tham số
        extract_grids(INPUT_CSV, grib_path, OUTPUT_GRID_DIR)
        
    print("\n✓ Pipeline hoàn tất cho toàn bộ các năm!")

# --- 4. THỰC THI ---
if __name__ == "__main__":
    process_all_years()