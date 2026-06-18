import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as config

def merge_enso():
    features_path = os.path.join(config.FINAL_DATA_DIR, "features_augmented.csv")
    enso_path = os.path.join(config.PROCESSED_DATA_DIR, "enso_clean.csv")
    output_path = os.path.join(config.FINAL_DATA_DIR, "features_augmented_enso.csv")
    
    print(f"Loading features from {features_path}...")
    df_features = pd.read_csv(features_path)
    
    print(f"Loading ENSO from {enso_path}...")
    df_enso = pd.read_csv(enso_path)
    
    # Check if ENSO is already in features
    if 'ENSO' in df_features.columns:
        print("ENSO already exists in features. Skipping merge.")
        df_features.to_csv(output_path, index=False)
        return
        
    print(f"Merging ENSO based on YEAR and MONTH...")
    # Left join on YEAR and MONTH
    df_merged = df_features.merge(df_enso, on=['YEAR', 'MONTH'], how='left')
    
    # Fill missing ENSO values using interpolation or forward fill just in case
    missing_enso = df_merged['ENSO'].isna().sum()
    if missing_enso > 0:
        print(f"Found {missing_enso} missing ENSO values. Interpolating...")
        df_merged['ENSO'] = df_merged['ENSO'].interpolate(method='linear', limit_direction='both')
        df_merged['ENSO'] = df_merged['ENSO'].fillna(df_merged['ENSO'].mean())
        
    print(f"Saving merged data to {output_path}...")
    df_merged.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    merge_enso()
