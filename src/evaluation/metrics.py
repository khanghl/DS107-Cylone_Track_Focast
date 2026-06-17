import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

def calculate_mae_r2(y_true_lat, y_true_lon, y_pred_lat, y_pred_lon):
    """Tinh MAE va R2 trung binh cho kinh do va vi do"""
    mae_lat = mean_absolute_error(y_true_lat, y_pred_lat)
    mae_lon = mean_absolute_error(y_true_lon, y_pred_lon)
    mae = (mae_lat + mae_lon) / 2
    
    r2_lat = r2_score(y_true_lat, y_pred_lat)
    r2_lon = r2_score(y_true_lon, y_pred_lon)
    r2 = (r2_lat + r2_lon) / 2
    return mae, r2

def haversine_distance(lat_true, lon_true, lat_pred, lon_pred):
    """Tính khoảng cách thực tế (km) trên mặt cầu Trái Đất"""
    lat_true, lon_true, lat_pred, lon_pred = map(np.radians, [lat_true, lon_true, lat_pred, lon_pred])
    dlat = lat_pred - lat_true
    dlon = lon_pred - lon_true
    a = np.sin(dlat/2.0)**2 + np.cos(lat_true) * np.cos(lat_pred) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Bán kính Trái Đất ~ 6371 km


def evaluate_advanced_metrics(y_true, y_pred, v_past):
    """
    y_true: Mảng chứa [delta_lat_obs, delta_lon_obs]
    y_pred: Mảng chứa [delta_lat_pred, delta_lon_pred]
    v_past: Vận tốc bão trong 24h qua (V_Actual_24h từ dữ liệu gốc)
    """
    
    # Chuyển đổi sang numpy array để tính toán vector nhanh
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    v_past = np.array(v_past)
    
    # Tách biệt Delta Lat (Y) và Delta Lon (X)
    dY_obs, dX_obs = y_true[:, 0], y_true[:, 1]
    dY_pred, dX_pred = y_pred[:, 0], y_pred[:, 1]
    
    # ---------------------------------------------------------
    # 2. DIRECTIONAL STABILITY (Độ ổn định hướng)
    # ---------------------------------------------------------
    # Điều kiện đúng: Cùng dấu (Tích > 0) cho cả vĩ độ và kinh độ
    correct_direction = (dX_obs * dX_pred > 0) & (dY_obs * dY_pred > 0)
    directional_stability = np.mean(correct_direction) * 100
    
    # ---------------------------------------------------------
    # 3. MOVING VELOCITY SENSITIVITY (Độ nhạy vận tốc)
    # ---------------------------------------------------------
    # Tính vận tốc dự báo và thực tế trong 24h tới (V_next)
    # Công thức: sqrt(delta_lat^2 + delta_lon^2) / thời gian
    # Ở đây dùng đơn vị tương đối (quãng đường dịch chuyển) để so sánh tỷ lệ
    v_next_obs = np.sqrt(dX_obs**2 + dY_obs**2)
    v_next_pred = np.sqrt(dX_pred**2 + dY_pred**2)
    
    # Tránh lỗi chia cho 0 nếu bão đứng yên
    v_past = np.where(v_past == 0, 0.001, v_past)
    
    # Xác định các ca biến động mạnh (Ground Truth)
    accel_cases = (v_next_obs / v_past > 2.0)
    decel_cases = (v_next_obs / v_past < 0.5)
    
    # Kiểm tra mô hình có bắt trúng không
    accel_hit = accel_cases & (v_next_pred / v_past > 2.0)
    decel_hit = decel_cases & (v_next_pred / v_past < 0.5)
    
    # Tính tỷ lệ nhạy (Sensitivity)
    total_anomalies = np.sum(accel_cases) + np.sum(decel_cases)
    total_hits = np.sum(accel_hit) + np.sum(decel_hit)
    
    # Tránh chia cho 0 nếu tập test không có ca đột biến nào
    velocity_sensitivity = (total_hits / total_anomalies * 100) if total_anomalies > 0 else 0
    
    # In kết quả
    # print("--- KẾT QUẢ ĐÁNH GIÁ CHUYÊN SÂU ---")
    print(f"1. Directional Stability: {directional_stability:.2f}%")
    # print(f"   (Kỳ vọng: > 87%)")
    print(f"2. Moving Velocity Sensitivity: {velocity_sensitivity:.2f}% \n")
    # print(f"   (Kỳ vọng: ~ 40-60% - Đây là điểm yếu của GBDT)")
    # print(f"   - Số ca đột biến thực tế: {total_anomalies}")
    # print(f"   - Số ca mô hình bắt trúng: {total_hits}")

    return directional_stability, velocity_sensitivity


