import pandas as pd
import numpy as np
from pathlib import Path

REF_LON = 108.20
REF_LAT = 16.05
BASE_PATH = Path(__file__).parent.parent.parent

DYNAMIC_DIR = BASE_PATH / 'data' / 'processed' / 'dynamic_feature.csv'

OUTPUT = BASE_PATH / 'data' / 'processed' / 'final_data.csv'

def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DYNAMIC_DIR)
    print(df.head())
    df = df.drop(columns='ISO_TIME')
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for t in [6, 12, 18, 24]:
        step = t // 6
        df[f'LAT_T-{t}'] = df.groupby('SID')['LAT'].shift(step)
        df[f'LON_T-{t}'] = df.groupby('SID')['LON'].shift(step)
        df[f'WIND_T-{t}'] = df.groupby('SID')['WMO_WIND'].shift(step)
        df[f'PRES_T-{t}'] = df.groupby('SID')['WMO_PRES'].shift(step)

    df['DIST2LAND_T-6'] = df.groupby('SID')['DIST2LAND'].shift(1)
    return df

def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    for hours in [6, 12, 18, 24]:
        # Zonal and meridional displacement (degrees per interval)
        df[f'V_Zonal_{hours}h']      = df['LON'] - df[f'LON_T-{hours}']
        df[f'V_Meridional_{hours}h'] = df['LAT'] - df[f'LAT_T-{hours}']

        # Intensity tendency
        df[f'Delta_PRES_{hours}h'] = df['WMO_PRES'] - df[f'PRES_T-{hours}']
        df[f'Delta_WIND_{hours}h'] = df['WMO_WIND'] - df[f'WIND_T-{hours}']

        # Resultant translation speed magnitude
        df[f'V_Actual_{hours}h'] = np.sqrt(
            df[f'V_Zonal_{hours}h']**2 + df[f'V_Meridional_{hours}h']**2
        )

    return df


def calculate_azimuth_and_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    for hours in [6, 12, 18, 24]:
        # Movement azimuth in degrees (-180 to +180, measured counter-clockwise from East)
        df[f'Azimuth_{hours}h'] = np.degrees(
            np.arctan2(df[f'V_Meridional_{hours}h'], df[f'V_Zonal_{hours}h'])
        )

    # Translational acceleration: change in the most-recent 6h velocity vector
    # computed over 12h and 24h look-back windows.
    df[f'Acc_Zonal_12h']      = df[f'V_Zonal_6h']      - df.groupby('SID')[f'V_Zonal_6h'].shift(2)
    df[f'Acc_Meridional_12h'] = df[f'V_Meridional_6h'] - df.groupby('SID')[f'V_Meridional_6h'].shift(2)
    df[f'Acc_Zonal_24h']      = df[f'V_Zonal_6h']      - df.groupby('SID')[f'V_Zonal_6h'].shift(4)
    df[f'Acc_Meridional_24h'] = df[f'V_Meridional_6h'] - df.groupby('SID')[f'V_Meridional_6h'].shift(4)

    # Resultant acceleration magnitude
    df[f'Acc_Actual_12h'] = np.sqrt(df[f'Acc_Zonal_12h']**2 + df[f'Acc_Meridional_12h']**2)
    df[f'Acc_Actual_24h'] = np.sqrt(df[f'Acc_Zonal_24h']**2 + df[f'Acc_Meridional_24h']**2)
    return df

def calc_angle_diff(angle1, angle2):
    return (angle1 - angle2 + 180) % 360 - 180

def calculate_directional_change(df: pd.DataFrame) -> pd.DataFrame:
    df['Diff_Azimuth_6_12']  = calc_angle_diff(df['Azimuth_6h'],  df['Azimuth_12h'])
    df['Diff_Azimuth_12_24'] = calc_angle_diff(df['Azimuth_12h'], df['Azimuth_24h'])
    df['Diff_Azimuth_6_18']  = calc_angle_diff(df['Azimuth_6h'],  df['Azimuth_18h'])
    df['Diff_Azimuth_6_24']  = calc_angle_diff(df['Azimuth_6h'],  df['Azimuth_24h'])
    return df

def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df['X55_Prod_V_6h']    = df['V_Zonal_6h']    * df['V_Meridional_6h']
    df['X56_Prod_V_12h']   = df['V_Zonal_12h']   * df['V_Meridional_12h']
    df['X57_Prod_V_18h']   = df['V_Zonal_18h']   * df['V_Meridional_18h']
    df['X58_Prod_V_24h']   = df['V_Zonal_24h']   * df['V_Meridional_24h']
    df['X59_Prod_Acc_12h'] = df['Acc_Zonal_12h'] * df['Acc_Meridional_12h']
    df['X60_Prod_Acc_24h'] = df['Acc_Zonal_24h'] * df['Acc_Meridional_24h']
    return df

def calculate_angular_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    _azimuth_past_12h = df.groupby('SID')['Azimuth_6h'].shift(2)
    _azimuth_past_24h = df.groupby('SID')['Azimuth_6h'].shift(4)
    df['X61_Angular_Acc_12h'] = calc_angle_diff(df['Azimuth_6h'], _azimuth_past_12h)
    df['X62_Angular_Acc_24h'] = calc_angle_diff(df['Azimuth_6h'], _azimuth_past_24h)
    return df

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

def calculate_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df['X63_Dist_6h']  = haversine(df['LON'], df['LAT'], df['LON_T-6'],  df['LAT_T-6'])
    df['X64_Dist_12h'] = haversine(df['LON'], df['LAT'], df['LON_T-12'], df['LAT_T-12'])
    df['X65_Dist_18h'] = haversine(df['LON'], df['LAT'], df['LON_T-18'], df['LAT_T-18'])
    df['X66_Dist_24h'] = haversine(df['LON'], df['LAT'], df['LON_T-24'], df['LAT_T-24'])
    return df

def calculate_Actual_Acceleration(df: pd.DataFrame) -> pd.DataFrame:
    # Temporary helpers — not stored as columns.
    _v_past_12h = df.groupby('SID')['V_Actual_6h'].shift(2)
    _v_past_24h = df.groupby('SID')['V_Actual_6h'].shift(4)

    # Speed acceleration: did the storm speed up or slow down?
    df['X67_Actual_Acc_12h'] = df['V_Actual_6h'] - _v_past_12h
    df['X68_Actual_Acc_24h'] = df['V_Actual_6h'] - _v_past_24h
    return df

def calulate_distance_to_land_change(df: pd.DataFrame) -> pd.DataFrame:
    df['Dist_Ref_Zonal']      = df['LON'] - REF_LON
    df['Dist_Ref_Meridional'] = df['LAT'] - REF_LAT
    df['Dist_Ref_Actual']     = np.sqrt(df['Dist_Ref_Zonal']**2 + df['Dist_Ref_Meridional']**2)
    return df

def calculate_grib(df: pd.DataFrame) -> pd.DataFrame:
    df['_Grid_LON'] = (df['LON'] // 2) * 2
    df['_Grid_LAT'] = (df['LAT'] // 2) * 2

    years = sorted(df['YEAR'].unique())
    climatology_list = []

    for current_year in years:
        historical_data   = df[df['YEAR'] < current_year]
        current_year_data = df[df['YEAR'] == current_year].copy()

        clim_cols = ['X79_Total_Storms', 'X81_Mean_V_Zonal',
                    'X82_Mean_V_Meridional', 'X84_Mean_Azimuth']

        if not historical_data.empty:
            stats = historical_data.groupby(['_Grid_LAT', '_Grid_LON']).agg(
                X79_Total_Storms      = ('SID',            'nunique'),
                X81_Mean_V_Zonal      = ('V_Zonal_6h',     'mean'),
                X82_Mean_V_Meridional = ('V_Meridional_6h','mean'),
                X84_Mean_Azimuth      = ('Azimuth_6h',     'mean'),
            ).reset_index()

            current_year_data = current_year_data.merge(
                stats, on=['_Grid_LAT', '_Grid_LON'], how='left'
            )
        else:
            # Year 2000 has no historical data; columns initialised to NaN, filled below.
            for col in clim_cols:
                current_year_data[col] = np.nan

        climatology_list.append(current_year_data)

    df = pd.concat(climatology_list, ignore_index=True)

    clim_cols = ['X79_Total_Storms', 'X81_Mean_V_Zonal',
                'X82_Mean_V_Meridional', 'X84_Mean_Azimuth']
    global_means = {}
    for year in years:
        hist = df[df['YEAR'] < year]
        global_means[year] = hist[clim_cols].mean() if not hist.empty else pd.Series({c: 0.0 for c in clim_cols})

    for year in years:
        mask = df['YEAR'] == year
        df.loc[mask, clim_cols] = df.loc[mask, clim_cols].fillna(global_means[year])

    cols_to_fix = df.columns.difference(['SID'])
    df[cols_to_fix] = df.groupby('SID')[cols_to_fix].transform(lambda x: x.bfill())
    df = df.fillna(df.mean(numeric_only=True))


    df = df.drop(columns=['_Grid_LON', '_Grid_LAT'])

    print("Total features (excl. SID + time + targets):", df.shape[1])
    print("NaN count after fillna:", df.isnull().sum().sum())

    return df

def feature_engineering():
    df = load_clean_data()
    df = add_lag_features(df)
    df = calculate_derived_features(df)
    df = calculate_azimuth_and_acceleration(df)
    df = calculate_directional_change(df)
    df = create_product_features(df)
    df = calculate_angular_acceleration(df)
    df = calculate_distance_features(df)
    df = calculate_Actual_Acceleration(df)
    df = calulate_distance_to_land_change(df)
    df = calculate_grib(df)
    df.to_csv(OUTPUT, index=False)
    print("Saved -> data/clean/final_data.csv")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    feature_engineering()