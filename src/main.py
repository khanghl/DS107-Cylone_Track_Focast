from pathlib import Path
from data_loader import (
    preprocess_ibtracs_data,
    process_enso_data,
    ERA5_extracting,
    feature_engineering,
    target_split
)
from pipeline.compare import run_experiment_matrix

BASE_DIR = Path(__file__).parent.parent

def run_full_pipeline() -> None:
    print("\n" + "=" * 80)
    print("🌧️  CYCLONE TRACK FORECAST SYSTEM")
    print("=" * 80)
    
    '''# Step 1: Process IBTRACS data
    print("\n[STEP 1/6] 🛰️  IBTRACS Data Preprocessing...")
    preprocess_ibtracs_data()
    print("✅ IBTRACS data preprocessing complete!")
    
    # Step 2: Process ENSO data
    print("\n[STEP 2/6] 🔽 Processing ENSO data...")
    process_enso_data()
    print("✅ ENSO data processing complete!")
    
    # Step 3: Process single levels
    print("\n[STEP 3/6] 🔺 Processing ERA5 single-level data...")
    ERA5_extracting()
    print("✅ Single-level processing complete!")'''

    # Step 4: Feature engineering
    print("\n[STEP 4/6] 🔺 Feature Engineering...")
    feature_engineering()
    print("✅ Feature engineering complete!")

    # Step 5: Target splitting
    print("\n[STEP 5/6] 🔀 Target Splitting...")
    target_split()
    print("✅ Target splitting complete!")
    
    # Step 6: Pipeline running
    print('\n [STEP 6/6] Pipeline running')
    run_experiment_matrix()
    print('Completed!')

def main() -> None:
    """
    Main entry point with menu for different pipeline modes.
    """
    run_full_pipeline()


if __name__ == '__main__':
    main()
