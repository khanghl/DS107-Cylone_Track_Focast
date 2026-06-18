import config as config
from sklearn.linear_model import LinearRegression


class CLIPERForecastingStrategy:
    """Mô hình Baseline CLIPER truyền thống dựa trên Hồi quy tuyến tính đa biến"""
    def __init__(self):
        # Khai báo các cột đặc trưng chuẩn CLIPER từ file features.csv của ông
        self.cliper_features = [
            'LAT', 'LON', 'WMO_WIND', 'WMO_PRES',                # 1. Trạng thái hiện tại
            'LAT_T-6', 'LON_T-6', 'LAT_T-12', 'LON_T-12',        # 2. Vị trí quá khứ (Persistence)
            'V_Zonal_6h', 'V_Meridional_6h', 'V_Actual_6h',      # 3. Động lực học quán tính
            'Azimuth_6h',                                        # 4. Góc di chuyển và Yếu tố mùa
            'X79_Total_Storms', 'X81_Mean_V_Zonal',              # 5. Thống kê khí hậu lịch sử (Climatology)
            'X82_Mean_V_Meridional', 'X84_Mean_Azimuth'
        ]
        
        # Khởi tạo mô hình hồi quy độc lập cho từng mốc thời gian
        self.model_lat = {h: LinearRegression() for h in config.HORIZONS}
        self.model_lon = {h: LinearRegression() for h in config.HORIZONS}
        self.model_wind = {h: LinearRegression() for h in config.HORIZONS}

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, y_train_wind_dict, *args, **kwargs):
        """Huấn luyện mô hình CLIPER phẳng (Không cần tập Valid vì Linear không bị Overfit sâu)"""
        # Trích xuất đúng các đặc trưng CLIPER
        X_train_cliper = X_train[self.cliper_features]
        
        for horizon in config.HORIZONS:
            print(f"   [CLIPER Baseline] Training horizon +{horizon}h...")
            
            # Khớp mô hình hồi quy tuyến tính
            self.model_lat[horizon].fit(X_train_cliper, y_train_lat_dict[horizon])
            self.model_lon[horizon].fit(X_train_cliper, y_train_lon_dict[horizon])
            self.model_wind[horizon].fit(X_train_cliper, y_train_wind_dict[horizon])

    def predict(self, X_test):
        """Dự báo mốc tương lai trực tiếp từ các đặc trưng tĩnh ban đầu"""
        X_test_cliper = X_test[self.cliper_features]
        predictions_lat = {}
        predictions_lon = {}
        predictions_wind = {}
        
        for horizon in config.HORIZONS:
            predictions_lat[horizon] = self.model_lat[horizon].predict(X_test_cliper)
            predictions_lon[horizon] = self.model_lon[horizon].predict(X_test_cliper)
            predictions_wind[horizon] = self.model_wind[horizon].predict(X_test_cliper)
            
        return predictions_lat, predictions_lon, predictions_wind