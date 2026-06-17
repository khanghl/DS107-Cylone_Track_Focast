import os
import sys
import pickle
import pandas as pd
from pathlib import Path

# Thêm src vào đường dẫn để import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / 'src'))

from pipeline.compare import load_and_split_data
from pipeline.multimodal import MultimodalForecastingStrategy

def export_model_and_data():
    print("Loading data...")
    (X_train, X_val, X_test, 
     y_train_lat, y_train_lon, y_train_wind,
     y_val_lat, y_val_lon, y_val_wind,
     y_test_lat, y_test_lon, y_test_wind, test_df) = load_and_split_data()
    
    print("Training XGBoost Multimodal (Track + Intensity)...")
    chain = MultimodalForecastingStrategy(model_name='xgboost')
    chain.train(X_train, y_train_lat, y_train_lon, y_train_wind, X_val, y_val_lat, y_val_lon, y_val_wind)
    
    # Define save paths
    models_dir = BASE_DIR / 'api' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / 'best_model.pkl'
    data_path = models_dir / 'test_data.csv'
    
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(chain, f)
        
    print("Saving test data for API demo...")
    test_df.to_csv(data_path, index=False)
    
    print("Export completed successfully!")

if __name__ == "__main__":
    export_model_and_data()
