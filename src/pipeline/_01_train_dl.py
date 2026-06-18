import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import sys

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.module.dl_loader import get_dataloader
from src.module.vision_encoder import HurricastCNNEncoder

def train_multimodal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    features_csv = BASE_DIR / 'data' / 'final' / 'features.csv'
    target_csv = BASE_DIR / 'data' / 'final' / 'target.csv'
    grids_dir = BASE_DIR / 'data' / 'processed' / 'vision_grids'
    
    print("Initialize Dataloader...")
    try:
        dataloader = get_dataloader(str(features_csv), str(target_csv), str(grids_dir), batch_size=16)
        print(f"Loaded {len(dataloader.dataset)} data samples.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Note: Run _03_ERA5_grid_extractor.py first!")
        return
        
    if len(dataloader.dataset) == 0:
        print("Vision data is empty!")
        return

    sample = next(iter(dataloader))
    tabular_dim = sample['tabular'].shape[-1]
    in_channels = sample['vision'].shape[2]
    
    # Khởi tạo mô hình Multi-task
    model = HurricastCNNEncoder(in_channels=in_channels, tabular_dim=tabular_dim, cnn_out_dim=128, hidden_dim=128)
    model = model.to(device)
    
    criterion_track = nn.MSELoss()
    criterion_intensity = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Hệ số Multi-task learning
    alpha = 1.0 # Trọng số cho MSE Quỹ đạo
    beta = 0.5  # Trọng số cho MSE Cường độ
    
    epochs = 50
    
    print("Start End-to-End Multitask Training...")
    best_loss = float('inf')
    model_dir = BASE_DIR / 'api' / 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            vision = batch['vision'].to(device)
            tabular = batch['tabular'].to(device)
            target = batch['target'].to(device)
            
            target_track = target[:, 0:2] # Delta Lat, Delta Lon
            target_wind = target[:, 2:3]  # Delta Wind
            
            optimizer.zero_grad()
            
            track_pred, intensity_pred, _ = model(vision, tabular)
            
            loss_track = criterion_track(track_pred, target_track)
            loss_intensity = criterion_intensity(intensity_pred, target_wind)
            
            # Hàm Loss tổng hợp (Đa mục tiêu)
            loss = alpha * loss_track + beta * loss_intensity
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients further
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Total Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_dir / 'hurricast_multitask.pth')
            print(f"  -> Saved best model with loss {best_loss:.4f}")

    print("Finished training. Best model weights saved to api/models/hurricast_multitask.pth")

if __name__ == "__main__":
    train_multimodal()
