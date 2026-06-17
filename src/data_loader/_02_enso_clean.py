
import pandas as pd
from pathlib import Path

YEAR_START = 2000
YEAR_END = 2025

MONTH_MAPPING = {
    'YEAR': 'YEAR',
    'DJ': 1,   # Dec-Jan (Winter)
    'JF': 2,   # Jan-Feb
    'FM': 3,   # Feb-Mar
    'MA': 4,   # Mar-Apr
    'AM': 5,   # Apr-May
    'MJ': 6,   # May-Jun
    'JJ': 7,   # Jun-Jul (Summer)
    'JA': 8,   # Jul-Aug
    'AS': 9,   # Aug-Sep
    'SO': 10,  # Sep-Oct
    'ON': 11,  # Oct-Nov
    'ND': 12   # Nov-Dec
}

# Input and output paths
BASE_DIR = Path(__file__).parent.parent.parent
INPUT_FILE = BASE_DIR / 'data' / 'raw' / 'meiv2.data'
OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'enso_clean.csv'

def _load_raw_enso_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=r'\s+')
    return df

def _filter_year_range(df: pd.DataFrame, year_start: int, year_end: int) -> pd.DataFrame:
    df = df[(df['YEAR'] >= year_start) & (df['YEAR'] <= year_end)]
    return df.reset_index(drop=True)


def _rename_month_columns(df: pd.DataFrame, month_mapping: dict) -> pd.DataFrame:
    df.columns = df.columns.map(month_mapping)
    return df

def _reshape_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.melt(
        id_vars='YEAR',
        var_name='MONTH',
        value_name='ENSO'
    )
    return df


def _sort_by_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['YEAR', 'MONTH']).reset_index(drop=True)
    return df

def process_enso_data(input_path: str = None, output_path: str = None) -> pd.DataFrame:
    if input_path is None:
        input_path = INPUT_FILE
    if output_path is None:
        output_path = OUTPUT_FILE
    
    print("--- 📊 PROCESSING ENSO (MEI v2) DATA ---")
    
    # Step 1: Load data
    print(f"Loading raw ENSO data from: {input_path}")
    df = _load_raw_enso_data(input_path)
    
    # Step 2: Filter year range
    print(f"Filtering data to years {YEAR_START}-{YEAR_END}")
    df = _filter_year_range(df, YEAR_START, YEAR_END)
    
    # Step 3: Rename columns
    print("Renaming month columns (abbreviations → numeric indices)")
    df = _rename_month_columns(df, MONTH_MAPPING)
    
    # Step 4: Reshape to long format
    print("Reshaping from wide to long format (tidy)")
    df = _reshape_to_long_format(df)
    
    # Step 5: Sort
    print("Sorting by YEAR and MONTH")
    df = _sort_by_year_month(df)
    
    # Step 6: Save
    print(f"Saving processed data to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"✅ ENSO processing complete!")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Columns: {list(df.columns)}")
    
    return df


if __name__ == '__main__':
    process_enso_data()