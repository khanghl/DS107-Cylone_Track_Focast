import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import os
import config as config
from pipeline.normal import IndependentForecastingStrategy
from pipeline.cliper import CLIPERForecastingStrategy
from evaluation.metrics import haversine_distance, evaluate_advanced_metrics, calculate_mae_r2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pipeline.mlp_strategy import MLPForecastingStrategy

def load_and_split_data():
    """Đọc dữ liệu và chia thành 3 phần Train/Valid/Test cho tất cả các mốc"""
    features_path = os.path.join(config.FINAL_DATA_DIR, "features_augmented.csv")
    features = pd.read_csv(features_path)
    target = pd.read_csv(config.TARGET_PATH)

    df = features.merge(target, on=config.MERGE_LABELS, how='left')

    # Loại bỏ NaN trên toàn bộ các cột Target để tập dữ liệu đồng nhất
    all_targets = []
    for h in config.HORIZONS:
        all_targets.extend([f'DELTA_LAT_{h}h', f'DELTA_LON_{h}h', f'TARGET_WIND_{h}h'])
    df = df.dropna(subset=all_targets)

    # Chia tập dữ liệu theo năm như logic của bạn (Data từ 2015-2025)
    train_df = df[(df['YEAR'] >= 2015) & (df['YEAR'] <= 2021)].copy()
    val_df = df[(df['YEAR'] >= 2022) & (df['YEAR'] <= 2023)].copy()  
    test_df = df[(df['YEAR'] >= 2024) & (df['YEAR'] <= 2025)].copy()

    # Gom tách target thành các dictionary tương ứng với từng mốc thời gian
    y_train_lat, y_train_lon, y_train_wind = {}, {}, {}
    y_val_lat, y_val_lon, y_val_wind = {}, {}, {}
    y_test_lat, y_test_lon, y_test_wind = {}, {}, {}

    for h in config.HORIZONS:
        y_train_lat[h] = train_df[f'DELTA_LAT_{h}h']
        y_train_lon[h] = train_df[f'DELTA_LON_{h}h']
        y_train_wind[h] = train_df[f'TARGET_WIND_{h}h']
        y_val_lat[h] = val_df[f'DELTA_LAT_{h}h']
        y_val_lon[h] = val_df[f'DELTA_LON_{h}h']
        y_val_wind[h] = val_df[f'TARGET_WIND_{h}h']
        y_test_lat[h] = test_df[f'DELTA_LAT_{h}h']
        y_test_lon[h] = test_df[f'DELTA_LON_{h}h']
        y_test_wind[h] = test_df[f'TARGET_WIND_{h}h']

    all_targets_to_drop = config.DROPS_COLUMNS.copy()
    for h in config.HORIZONS:
        all_targets_to_drop.extend([f'DELTA_LAT_{h}h', f'DELTA_LON_{h}h', f'TARGET_WIND_{h}h'])
        
    # Ensure no duplicates and only drop existing columns
    all_targets_to_drop = list(set(all_targets_to_drop).intersection(set(train_df.columns)))

    X_train = train_df.drop(columns=all_targets_to_drop)
    X_val = val_df.drop(columns=all_targets_to_drop)
    X_test = test_df.drop(columns=all_targets_to_drop)

    return (X_train, X_val, X_test, 
            y_train_lat, y_train_lon, y_train_wind,
            y_val_lat, y_val_lon, y_val_wind,
            y_test_lat, y_test_lon, y_test_wind, test_df)


def run_experiment_matrix():
    (X_train, X_val, X_test, 
     y_train_lat, y_train_lon, y_train_wind,
     y_val_lat, y_val_lon, y_val_wind,
     y_test_lat, y_test_lon, y_test_wind, test_df) = load_and_split_data()
    
    report_rows = []
    trajectory_dfs = []
    storm_identities = {
        'SID': test_df['SID'].values,
        'YEAR': test_df['YEAR'].values,
        'MONTH': test_df['MONTH'].values,
        'DAY': test_df['DAY'].values,
        'HOUR': test_df['HOUR'].values,
        'BASE_LAT': test_df['LAT'].values,
        'BASE_LON': test_df['LON'].values
    }
    
    # Dynamic column identification
    all_cols = list(X_train.columns)
    vis_cols = [col for col in all_cols if col.startswith('vision_emb_')]
    raw_cols = [
        'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'DIST2LAND',
        'LAT_T-6', 'LON_T-6', 'WIND_T-6', 'PRES_T-6',
        'LAT_T-12', 'LON_T-12', 'WIND_T-12', 'PRES_T-12',
        'LAT_T-18', 'LON_T-18', 'WIND_T-18', 'PRES_T-18',
        'LAT_T-24', 'LON_T-24', 'WIND_T-24', 'PRES_T-24',
        'DIST2LAND_T-6'
    ]
    raw_cols = [col for col in raw_cols if col in all_cols]
    eng_cols = [col for col in all_cols if col not in raw_cols and col not in vis_cols]
    
    print(f"Features loaded: Raw={len(raw_cols)}, Eng={len(eng_cols)}, Vision={len(vis_cols)}")

    scenarios = [
        {
            "name": "1. CLIPER (ibtracs)",
            "strategy": "cliper",
            "features": None
        },
        {
            "name": "2. LightGBM global (ibtracs)",
            "strategy": "lightgbm",
            "features": raw_cols
        },
        {
            "name": "3. era5->3D->vector->LightGBM",
            "strategy": "lightgbm",
            "features": vis_cols
        },
        {
            "name": "4. ibtracs + 3D --> LightGBM",
            "strategy": "lightgbm",
            "features": raw_cols + vis_cols
        },
        {
            "name": "5. ibtracs_feature_engineering + 3D --> LightGBM",
            "strategy": "lightgbm",
            "features": raw_cols + eng_cols + vis_cols
        },
        {
            "name": "6. ibtracs_feature_engineering_PCA + 3D --> LightGBM",
            "strategy": "lightgbm_pca",
            "features": None
        },
        {
            "name": "7. MLP (ibtracs_feature_engineering_PCA + 3D)",
            "strategy": "mlp_pca",
            "features": None
        }
    ]
    
    for scenario in scenarios:
        name = scenario["name"]
        strategy = scenario["strategy"]
        print(f"\n==================== RUNNING SCENARIO: {name} ====================")
        
        # Train and Predict depending on the strategy
        if strategy == "cliper":
            runner = CLIPERForecastingStrategy()
            runner.train(X_train, y_train_lat, y_train_lon)
            pred_lat, pred_lon = runner.predict(X_test)
            
        elif strategy in ["lightgbm_pca", "mlp_pca"]:
            # Scale engineered features
            scaler = StandardScaler()
            X_train_eng_scaled = scaler.fit_transform(X_train[eng_cols])
            X_val_eng_scaled = scaler.transform(X_val[eng_cols])
            X_test_eng_scaled = scaler.transform(X_test[eng_cols])
            
            # Apply PCA
            pca = PCA(n_components=0.95, random_state=config.RANDOM_STATE)
            X_train_pca = pca.fit_transform(X_train_eng_scaled)
            X_val_pca = pca.transform(X_val_eng_scaled)
            X_test_pca = pca.transform(X_test_eng_scaled)
            
            n_comps = X_train_pca.shape[1]
            if strategy == "lightgbm_pca":
                print(f"   PCA reduced {len(eng_cols)} engineered features to {n_comps} components (95% variance explained)")
            
            pca_cols = [f'pca_eng_{i}' for i in range(n_comps)]
            X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
            X_val_pca_df = pd.DataFrame(X_val_pca, columns=pca_cols, index=X_val.index)
            X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
            
            # Concat everything
            X_tr = pd.concat([X_train[raw_cols], X_train_pca_df, X_train[vis_cols]], axis=1)
            X_va = pd.concat([X_val[raw_cols], X_val_pca_df, X_val[vis_cols]], axis=1)
            X_te = pd.concat([X_test[raw_cols], X_test_pca_df, X_test[vis_cols]], axis=1)
            
            if strategy == "lightgbm_pca":
                runner = IndependentForecastingStrategy('lightgbm')
            else:
                runner = MLPForecastingStrategy()
                
            runner.train(X_tr, y_train_lat, y_train_lon, X_va, y_val_lat, y_val_lon)
            pred_lat, pred_lon = runner.predict(X_te)
            
        else: # Standard LightGBM
            feature_list = scenario["features"]
            X_tr = X_train[feature_list]
            X_va = X_val[feature_list]
            X_te = X_test[feature_list]
            
            runner = IndependentForecastingStrategy('lightgbm')
            runner.train(X_tr, y_train_lat, y_train_lon, X_va, y_val_lat, y_val_lon)
            pred_lat, pred_lon = runner.predict(X_te)
            
        # Evaluation
        for h in config.HORIZONS:
            true_lat = test_df['LAT'].values + y_test_lat[h].values
            true_lon = test_df['LON'].values + y_test_lon[h].values
            
            pred_lat_geo = test_df['LAT'].values + pred_lat[h]
            pred_lon_geo = test_df['LON'].values + pred_lon[h]
            
            err = haversine_distance(true_lat, true_lon, pred_lat_geo, pred_lon_geo)
            err_mean = np.mean(err)
            print(f"   Horizon +{h}h -> haversine_distance: {err_mean:.2f} km")
            
            v_past = test_df[f'V_Actual_{h}h'].values
            y_test_actual = pd.DataFrame(np.column_stack((y_test_lat[h], y_test_lon[h])))
            y_test_predicted = pd.DataFrame(np.column_stack((pred_lat[h], pred_lon[h])))
            
            directional_stability, velocity_sensitivity = evaluate_advanced_metrics(y_test_actual, y_test_predicted, v_past)
            
            # Calculate MAE and R2 using the new metrics function
            mae, r2 = calculate_mae_r2(y_test_lat[h], y_test_lon[h], pred_lat[h], pred_lon[h])
            print(f"   Horizon +{h}h -> MAE: {mae:.4f} deg, R2: {r2:.4f}")
            
            report_rows.append({
                "Model/Strategy": name,
                "Horizon": f"+{h}h",
                "haversine_distance": round(err_mean, 2),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                'Directional_Stability': directional_stability,
                'Moving_Velocity_Sensitivity': velocity_sensitivity
            })
            
            traj_df = pd.DataFrame(storm_identities)
            traj_df['Horizon'] = f"+{h}h"
            traj_df['Strategy'] = name
            traj_df['TRUE_TARGET_LAT'] = true_lat
            traj_df['TRUE_TARGET_LON'] = true_lon
            traj_df['PRED_TARGET_LAT'] = pred_lat_geo
            traj_df['PRED_TARGET_LON'] = pred_lon_geo
            trajectory_dfs.append(traj_df)
            
    # Save reports
    df_report = pd.DataFrame(report_rows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    report_path = os.path.join(config.RESULTS_DIR, "final_comparison_report_ver03.csv")
    df_report.to_csv(report_path, index=False)
    print(f"\n[SUCCESS] Saved report to: {report_path}")
    
    df_trajectories = pd.concat(trajectory_dfs, ignore_index=True)
    traj_path = os.path.join(config.RESULTS_DIR, "detailed_trajectories_predictions.csv")
    df_trajectories.to_csv(traj_path, index=False)
    print(f"[SUCCESS] Saved detailed trajectories to: {traj_path}")

if __name__ == "__main__":
    run_experiment_matrix()
