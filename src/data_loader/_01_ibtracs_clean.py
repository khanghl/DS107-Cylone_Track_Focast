import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

FEATURES = [
    "SID",              # ID của bão
    "SEASON",           # Mùa
    "NAME",             # Tên bão
    "ISO_TIME",         # Thời gian (định dạng ISO)
    "LAT",              # Vĩ độ
    "LON",              # Kinh độ
    'BASIN',            # Vùng bão
    'SUBBASIN',         # Vùng phụ bão
    "WMO_WIND",         # Tốc độ gió (theo WMO)
    "WMO_PRES",         # Áp suất (theo WMO)
    "DIST2LAND",        # Khoảng cách đến đất liền
    "STORM_SPEED",      # Tốc độ di chuyển của bão
    "STORM_DIR",        # Hướng di chuyển của bão (theo độ, 0-360)
]

PATH = BASE_DIR / 'data' / 'raw' / 'ibtracs.csv'
OUTPUT = BASE_DIR / 'data' / 'processed' / 'dynamic_feature.csv'

def load_ibtracs_data() -> pd.DataFrame:
    df = pd.read_csv(PATH, engine="python", on_bad_lines='skip')
    df = df.iloc[1:]
    return df

def clean_ibtracs_data(df: pd.DataFrame) -> pd.DataFrame:
    features = FEATURES
    df = df[features]
    return df

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    to_num_cols = ['WMO_WIND','WMO_PRES','DIST2LAND',
                   'STORM_SPEED','STORM_DIR', 'SEASON', 'LAT', 'LON']
    for col in to_num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df['BASIN']    = df['BASIN'].astype(str)
    df['SUBBASIN'] = df['SUBBASIN'].astype(str)
    df['NAME']     = df['NAME'].astype(str)
    return df

def filter_ibtracs_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['SEASON'] <= 2025) & 
            (df['SEASON'] >= 2015) & 
            (df['BASIN'] == 'WP')]

    df = df[df['ISO_TIME'].dt.hour % 6 == 0]
    df = df[df['WMO_WIND'] >= 30]

    print(f"Rows after filter: {len(df):,}")
    print(f"Unique storms: {df['SID'].nunique()}")
    return df

def filter_long_duration_storms(df: pd.DataFrame) -> pd.DataFrame:
    storm_time = df.groupby('SID')['ISO_TIME'].agg(['min','max'])
    storm_time['duration'] = storm_time['max'] - storm_time['min']

    more3day = storm_time[storm_time['duration'] > pd.Timedelta(days=3)].index.tolist()
    df = df[df['SID'].isin(more3day)]

    print(f"Số bão cuối cùng (WP, >3 ngày): {df['SID'].nunique()}")
    return df

def extract_datetime_components(df: pd.DataFrame) -> pd.DataFrame:
    df['YEAR'] = df['ISO_TIME'].dt.year
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['DAY'] = df['ISO_TIME'].dt.day
    df['HOUR'] = df['ISO_TIME'].dt.hour
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ['BASIN', 'SUBBASIN', 'NAME', 'SEASON', 'STORM_SPEED', 'STORM_DIR']
    df = df.drop(columns=drop_cols)

    cols = ['SID', 'ISO_TIME', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'LAT', 'LON',
            'WMO_WIND', 'WMO_PRES', 'DIST2LAND']
    df = df[cols]
    return df

def preprocess_ibtracs_data() -> pd.DataFrame:
    df = load_ibtracs_data()
    df = clean_ibtracs_data(df)
    df = convert_to_numeric(df)
    df = filter_ibtracs_data(df)
    df = filter_long_duration_storms(df)
    df = extract_datetime_components(df)
    df = drop_unnecessary_columns(df)
    df.to_csv(OUTPUT, index=False)
    print(f"Preprocessed IBTrACS data with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Cleaned IBTrACS data saved to {OUTPUT}")

if __name__ == "__main__":
    df_ibtracs = preprocess_ibtracs_data()