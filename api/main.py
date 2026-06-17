import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / 'src'))

import config

app = FastAPI(title="Cyclone Track Forecast API")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = BASE_DIR / 'api' / 'models' / 'best_model.pkl'
DATA_PATH = BASE_DIR / 'api' / 'models' / 'test_data.csv'

# Global variables to hold model and data
model_chain = None
test_df = None
feature_cols = None
storm_dpe_24h = {}

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

@app.on_event("startup")
def load_assets():
    global model_chain, test_df, feature_cols
    print("Loading model...")
    if MODEL_PATH.exists():
        with open(MODEL_PATH, 'rb') as f:
            model_chain = pickle.load(f)
    else:
        print("Warning: Model not found!")
        
    print("Loading test data...")
    if DATA_PATH.exists():
        test_df = pd.read_csv(DATA_PATH)
        # Identify feature columns (everything not in DROPS_COLUMNS and not TRUE_TARGETS)
        target_cols = []
        for h in config.HORIZONS:
            target_cols.extend([f'DELTA_LAT_{h}h', f'DELTA_LON_{h}h', f'TARGET_WIND_{h}h'])
            
        drops = config.DROPS_COLUMNS + target_cols
        feature_cols = [c for c in test_df.columns if c not in drops]
        
        # Precalculate mean DPE_24h for each storm
        if model_chain is not None:
            print("Precalculating DPE 24h for storms...")
            try:
                pred_lat, pred_lon, pred_wind = model_chain.predict(test_df[feature_cols])
            except ValueError:
                # Fallback if old model is loaded
                pred_lat, pred_lon = model_chain.predict(test_df[feature_cols])
            
            true_lat_24 = test_df['LAT'] + test_df['DELTA_LAT_24h']
            true_lon_24 = test_df['LON'] + test_df['DELTA_LON_24h']
            pred_lat_24 = test_df['LAT'] + pred_lat[24]
            pred_lon_24 = test_df['LON'] + pred_lon[24]
            
            test_df['DPE_24h'] = haversine_dist(true_lat_24, true_lon_24, pred_lat_24, pred_lon_24)
            mean_dpe = test_df.groupby('SID')['DPE_24h'].mean().to_dict()
            global storm_dpe_24h
            storm_dpe_24h = mean_dpe
            
    else:
        print("Warning: Test data not found!")

@app.get("/")
def read_root():
    return {"message": "Cyclone Track Forecast API is running"}

@app.get("/api/storms")
def get_storms():
    """Return a list of unique storms available in the test dataset."""
    if test_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    # Get first occurrence of each storm
    storms = test_df.groupby('SID').first().reset_index()
    result = []
    for _, row in storms.iterrows():
        sid = row["SID"]
        result.append({
            "sid": sid,
            "year": int(row["YEAR"]),
            "month": int(row["MONTH"]),
            "start_lat": float(row["LAT"]),
            "start_lon": float(row["LON"]),
            "dpe_24h": round(storm_dpe_24h.get(sid, 0.0), 2)
        })
    return result

@app.get("/api/storms/{sid}/times")
def get_storm_times(sid: str):
    """Return available timestamps for a specific storm."""
    if test_df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    storm_data = test_df[test_df['SID'] == sid].sort_values(by=['YEAR', 'MONTH', 'DAY', 'HOUR'])
    if storm_data.empty:
        raise HTTPException(status_code=404, detail="Storm not found")
        
    times = []
    for idx, row in storm_data.iterrows():
        times.append({
            "index": int(idx),
            "time_str": f"{int(row['YEAR'])}-{int(row['MONTH']):02d}-{int(row['DAY']):02d} {int(row['HOUR']):02d}:00",
            "lat": float(row["LAT"]),
            "lon": float(row["LON"])
        })
    return times

class PredictRequest(BaseModel):
    index: int

@app.post("/api/predict")
def predict_track(request: PredictRequest):
    """Predict the track (6h, 12h, 24h) for a specific storm state given its DataFrame index."""
    if model_chain is None or test_df is None:
        raise HTTPException(status_code=500, detail="Model or Data not loaded")
        
    if request.index not in test_df.index:
        raise HTTPException(status_code=404, detail="Index not found in dataset")
        
    row = test_df.loc[[request.index]]
    X_input = row[feature_cols].copy()
    
    # Predict using the Multimodal Forecasting Strategy
    try:
        pred_lat, pred_lon, pred_wind = model_chain.predict(X_input)
    except ValueError:
        pred_lat, pred_lon = model_chain.predict(X_input)
        pred_wind = {h: [0.0] for h in config.HORIZONS}
    
    base_lat = float(row['LAT'].values[0])
    base_lon = float(row['LON'].values[0])
    base_wind = float(row['WMO_WIND'].values[0])
    
    predictions = {}
    for h in config.HORIZONS:
        d_lat = float(pred_lat[h][0])
        d_lon = float(pred_lon[h][0])
        d_wind = float(pred_wind[h][0])
        
        predictions[f"{h}h"] = {
            "delta_lat": d_lat,
            "delta_lon": d_lon,
            "delta_wind": d_wind,
            "pred_lat": base_lat + d_lat,
            "pred_lon": base_lon + d_lon,
            "pred_wind": base_wind + d_wind,
            "true_lat": base_lat + float(row[f'DELTA_LAT_{h}h'].values[0]),
            "true_lon": base_lon + float(row[f'DELTA_LON_{h}h'].values[0]),
            "true_wind": base_wind + float(row[f'TARGET_WIND_{h}h'].values[0]),
        }
        
    return {
        "sid": row['SID'].values[0],
        "current_lat": base_lat,
        "current_lon": base_lon,
        "predictions": predictions
    }
