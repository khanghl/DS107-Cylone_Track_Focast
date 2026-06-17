import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import config as config
import evaluation.metrics as metrics

def prepare_data(sub: list):
    features = pd.read_csv(config.FINAL_FEATURES_PATH)
    target = pd.read_csv(config.TARGET_PATH)

    df = features.merge(target, on=config.MERGE_LABELS, how='left')

    df = df.dropna(subset=sub)
    train_df = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2020)]
    val_df = df[(df['YEAR'] >= 2021) & (df['YEAR'] <= 2022)]  
    test_df = df[(df['YEAR'] >= 2023) & (df['YEAR'] <= 2024)]

    return train_df, val_df, test_df

def gdbt_6h():
    subsets = [[f'DELTA_LAT_{i}h', f'DELTA_LON_{i}h'] 
           for i in [6, 12, 24]]

    for subset in subsets:
        print(f'Running subset {subset}')
        train_df, val_df, test_df = prepare_data(subset)
        X_train = train_df.drop(columns=config.DROPS_COLUMNS)
        y_train_lat = train_df[subset[0]]
        y_train_lon = train_df[subset[1]]

        X_val = val_df.drop(columns=config.DROPS_COLUMNS)
        y_val_lat = val_df[subset[0]]
        y_val_lon = val_df[subset[1]]

        X_test = test_df.drop(columns=config.DROPS_COLUMNS)
        y_test_lat = test_df[subset[0]]
        y_test_lon = test_df[subset[1]]

        current_lat_test = test_df['LAT'].values
        current_lon_test = test_df['LON'].values

        print("Đang huấn luyện mô hình GBDT cho Vĩ độ (Latitude)...")
        model_lat = GradientBoostingRegressor(**config.GBDT_PARAMS)
        model_lat.fit(X_train, y_train_lat)

        print("Đang huấn luyện mô hình GBDT cho Kinh độ (Longitude)...")
        model_lon = GradientBoostingRegressor(**config.GBDT_PARAMS)
        model_lon.fit(X_train, y_train_lon)

        pred_delta_lat = model_lat.predict(X_test)
        pred_delta_lon = model_lon.predict(X_test)

        pred_lat_6h = current_lat_test + pred_delta_lat
        pred_lon_6h = current_lon_test + pred_delta_lon

        true_lat_6h = current_lat_test + y_test_lat.values
        true_lon_6h = current_lon_test + y_test_lon.values

        test_errors_km = metrics.haversine_distance(true_lat_6h, true_lon_6h, pred_lat_6h, pred_lon_6h)
        mean_error = np.mean(test_errors_km)

        print(f"MFE of {subset}: {mean_error:.2f} km")


        # More clearly
        v_past = test_df['V_Actual_12h'].values

        actual = np.column_stack((y_test_lat, y_test_lon))
        y_test_actual = pd.DataFrame(actual)

        predict = np.column_stack((pred_delta_lat, pred_delta_lon))
        y_test_predicted = pd.DataFrame(predict)

        metrics.evaluate_advanced_metrics(y_test_actual, y_test_predicted, v_past)

if __name__ == "__main__":
    gdbt_6h()