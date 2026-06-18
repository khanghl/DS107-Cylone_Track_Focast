import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import config as config
from evaluation.metrics import haversine_distance
import lightgbm as lgb
import xgboost as xgb
import optuna

# Import load_and_split_data from _04_compare
from pipeline._04_compare import load_and_split_data

print("Loading data...")
(X_train, X_val, X_test, 
 y_train_lat, y_train_lon, y_train_wind,
 y_val_lat, y_val_lon, y_val_wind,
 y_test_lat, y_test_lon, y_test_wind, test_df) = load_and_split_data()

# Reconstruct val_df to get base LAT and LON for validation set haversine distance
features_path = os.path.join(config.FINAL_DATA_DIR, "features_augmented_enso.csv")
features = pd.read_csv(features_path)
target = pd.read_csv(config.TARGET_PATH)
df = features.merge(target, on=config.MERGE_LABELS, how='left')

# Filter for validation years (2022-2023)
val_df = df[(df['YEAR'] >= 2022) & (df['YEAR'] <= 2023)].copy()

# Drop NaNs to align with load_and_split_data
all_targets = []
for h in config.HORIZONS:
    all_targets.extend([f'DELTA_LAT_{h}h', f'DELTA_LON_{h}h', f'TARGET_WIND_{h}h'])
val_df = val_df.dropna(subset=all_targets)

# Dynamic column identification
all_cols = list(X_train.columns)
vis_cols = [col for col in all_cols if col.startswith('vision_emb_')]
raw_cols = [
    'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND', 'ENSO',
    'LAT_T-6', 'LON_T-6', 'WIND_T-6', 'PRES_T-6',
    'LAT_T-12', 'LON_T-12', 'WIND_T-12', 'PRES_T-12',
    'LAT_T-18', 'LON_T-18', 'WIND_T-18', 'PRES_T-18',
    'LAT_T-24', 'LON_T-24', 'WIND_T-24', 'PRES_T-24',
    'DIST2LAND_T-6'
]
raw_cols = [col for col in raw_cols if col in all_cols]
eng_cols = [col for col in all_cols if col not in raw_cols and col not in vis_cols]

# We will tune on Scenario 5 features (all features, no PCA)
feature_cols = raw_cols + eng_cols + vis_cols
X_tr = X_train[feature_cols]
X_va = X_val[feature_cols]

def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': config.RANDOM_STATE,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'n_jobs': -1
    }
    
    val_errors = []
    for h in config.HORIZONS:
        model_lat = lgb.LGBMRegressor(**params)
        model_lat.fit(
            X_tr, y_train_lat[h],
            eval_set=[(X_va, y_val_lat[h])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        model_lon = lgb.LGBMRegressor(**params)
        model_lon.fit(
            X_tr, y_train_lon[h],
            eval_set=[(X_va, y_val_lon[h])],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        pred_lat = model_lat.predict(X_va)
        pred_lon = model_lon.predict(X_va)
        
        true_lat_geo = val_df['LAT'].values + y_val_lat[h].values
        true_lon_geo = val_df['LON'].values + y_val_lon[h].values
        
        pred_lat_geo = val_df['LAT'].values + pred_lat
        pred_lon_geo = val_df['LON'].values + pred_lon
        
        err = haversine_distance(true_lat_geo, true_lon_geo, pred_lat_geo, pred_lon_geo)
        val_errors.append(np.mean(err))
        
    return np.mean(val_errors)

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': config.RANDOM_STATE,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'n_jobs': -1,
        'early_stopping_rounds': 50
    }
    
    val_errors = []
    for h in config.HORIZONS:
        model_lat = xgb.XGBRegressor(**params)
        model_lat.fit(
            X_tr, y_train_lat[h],
            eval_set=[(X_va, y_val_lat[h])],
            verbose=False
        )
        
        model_lon = xgb.XGBRegressor(**params)
        model_lon.fit(
            X_tr, y_train_lon[h],
            eval_set=[(X_va, y_val_lon[h])],
            verbose=False
        )
        
        pred_lat = model_lat.predict(X_va)
        pred_lon = model_lon.predict(X_va)
        
        true_lat_geo = val_df['LAT'].values + y_val_lat[h].values
        true_lon_geo = val_df['LON'].values + y_val_lon[h].values
        
        pred_lat_geo = val_df['LAT'].values + pred_lat
        pred_lon_geo = val_df['LON'].values + pred_lon
        
        err = haversine_distance(true_lat_geo, true_lon_geo, pred_lat_geo, pred_lon_geo)
        val_errors.append(np.mean(err))
        
    return np.mean(val_errors)

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("========================================")
    print("Starting Optuna tuning for LightGBM (30 trials)...")
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective, n_trials=30)
    print("\n[LightGBM] Best validation error (Mean Haversine):", study_lgb.best_value)
    print("[LightGBM] Best params:", study_lgb.best_params)
    
    print("\n========================================")
    print("Starting Optuna tuning for XGBoost (30 trials)...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=30)
    print("\n[XGBoost] Best validation error (Mean Haversine):", study_xgb.best_value)
    print("[XGBoost] Best params:", study_xgb.best_params)
