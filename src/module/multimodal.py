import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import config as config

class MultimodalForecastingStrategy:
    """
    Chiến lược Dự báo Đa phương thức & Đa mục tiêu (Multimodal & Multi-task).
    Dự báo đồng thời: Quỹ đạo (Delta_Lat, Delta_Lon) và Cường độ (Target_Wind).
    Nếu features_augmented.csv có sẵn, mô hình sẽ tận dụng 128 biến Vision Embedding.
    """
    def __init__(self, model_name='xgboost'):
        self.model_name = model_name
        self.models = {} 
        
    def _get_base_model(self):
        if self.model_name == 'xgboost':
            return XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1)
        else:
            return LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, n_jobs=-1)
            
    def train(self, X_train, y_train_lat, y_train_lon, y_train_wind, X_val=None, y_val_lat=None, y_val_lon=None, y_val_wind=None):
        for h in config.HORIZONS:
            print(f"Training Multitask {self.model_name.upper()} for +{h}h (Track & Intensity)...")
            
            # Gộp 3 biến mục tiêu
            y_train = np.column_stack((y_train_lat[h], y_train_lon[h], y_train_wind[h]))
            
            # MultiOutputRegressor giúp XGBoost dự báo nhiều nhánh song song
            model = MultiOutputRegressor(self._get_base_model())
            model.fit(X_train, y_train)
            
            self.models[h] = model
            
    def predict(self, X_test):
        pred_lat, pred_lon, pred_wind = {}, {}, {}
        for h in config.HORIZONS:
            preds = self.models[h].predict(X_test)
            pred_lat[h] = preds[:, 0]
            pred_lon[h] = preds[:, 1]
            pred_wind[h] = preds[:, 2]
            
        return pred_lat, pred_lon, pred_wind
