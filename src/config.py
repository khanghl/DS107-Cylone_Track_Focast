import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FINAL_DATA_DIR = os.path.join(BASE_DIR, "data", "final")

FINAL_FEATURES_PATH = os.path.join(FINAL_DATA_DIR, "features.csv")
TARGET_PATH = os.path.join(FINAL_DATA_DIR, "target.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "experiments")

RANDOM_STATE = 42


MERGE_LABELS = ['SID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'LAT', 'LON']

DROPS_COLUMNS = ['SID', 'YEAR', 'MONTH', 'DAY', 'HOUR'] + [
                col for col in [f'DELTA_LAT_{i}h' for i in [6, 12, 24]]] + [
                col for col in [f'DELTA_LON_{i}h' for i in [6, 12, 24]]]


HORIZONS = [6, 12, 24]


# Thư viện LightGBM
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': RANDOM_STATE,
    'learning_rate': 0.07327294720616905,
    'n_estimators': 1347,
    'max_depth': 4,
    'num_leaves': 34,
    'feature_fraction': 0.63960540084321,
    'bagging_fraction': 0.7007190711558733,
    'bagging_freq': 1,
    'lambda_l1': 0.0001357423331071847,
    'lambda_l2': 0.007903293650218675,
    'min_child_samples': 57,
}
