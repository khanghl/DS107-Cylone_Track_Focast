import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import config as config

def get_model(model: str):
    model_name = model.lower()
    if model_name == 'lightgbm':
        return lgb.LGBMRegressor(**config.LGBM_PARAMS)
    
    elif model_name == 'gdbt':
        return GradientBoostingRegressor(**config.GBDT_PARAMS)
    
    elif model_name == 'xgboost':
        return xgb.XGBRegressor(early_stopping_rounds=50, **config.XGB_PARAMS)
    
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")