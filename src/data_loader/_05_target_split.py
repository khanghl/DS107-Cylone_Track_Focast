from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
INPUT_DIR = BASE_DIR / 'data' / 'processed' / 'final_data.csv'

TARGET_DIR = BASE_DIR / 'data' / 'final' / 'target.csv'
FEATURES_DIR = BASE_DIR / 'data' / 'final' / 'features.csv'

FEATURES = ['SID','YEAR','MONTH','DAY', 'HOUR', 'LAT', 'LON']

def load_data():
    df = pd.read_csv(INPUT_DIR)
    return df

def split_6h(df):
    df['DELTA_LAT_6h'] = df.groupby('SID')['LAT'].shift(-1) - df['LAT']
    df['DELTA_LON_6h'] = df.groupby('SID')['LON'].shift(-1) - df['LON']
    df['TARGET_WIND_6h'] = df.groupby('SID')['WMO_WIND'].shift(-1) - df['WMO_WIND']
    return df

def split_12h(df):
    df['DELTA_LAT_12h'] = df.groupby('SID')['LAT'].shift(-2) - df['LAT']
    df['DELTA_LON_12h'] = df.groupby('SID')['LON'].shift(-2) - df['LON']
    df['TARGET_WIND_12h'] = df.groupby('SID')['WMO_WIND'].shift(-2) - df['WMO_WIND']
    return df

def split_24h(df):
    df['DELTA_LAT_24h'] = df.groupby('SID')['LAT'].shift(-4) - df['LAT']
    df['DELTA_LON_24h'] = df.groupby('SID')['LON'].shift(-4) - df['LON']
    df['TARGET_WIND_24h'] = df.groupby('SID')['WMO_WIND'].shift(-4) - df['WMO_WIND']
    return df

def target_split():
    df = pd.read_csv(INPUT_DIR)
    df = split_6h(df) 
    df = split_12h(df)
    df = split_24h(df)
    target_cols = FEATURES + ['DELTA_LAT_6h', 'DELTA_LON_6h', 'TARGET_WIND_6h',
                              'DELTA_LAT_12h', 'DELTA_LON_12h', 'TARGET_WIND_12h',
                              'DELTA_LAT_24h', 'DELTA_LON_24h', 'TARGET_WIND_24h']
    features_cols = [col for col in df.columns if col not in target_cols]
    target_df = df[target_cols]
    features_df = df[FEATURES + features_cols]

    target_df.to_csv(TARGET_DIR, index=False)
    features_df.to_csv(FEATURES_DIR, index=False)

if __name__ == "__main__":
    target_split()