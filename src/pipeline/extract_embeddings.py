import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader.dl_loader import get_dataloader
from src.models.vision_encoder import HurricastCNNEncoder

def extract():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    features_csv = BASE_DIR / 'data' / 'final' / 'features.csv'
    target_csv = BASE_DIR / 'data' / 'final' / 'target.csv'
    grids_dir = BASE_DIR / 'data' / 'processed' / 'vision_grids'
    
    try:
        dataloader = get_dataloader(str(features_csv), str(target_csv), str(grids_dir), batch_size=32, shuffle=False)
    except Exception as e:
        print("Data load error")
        return
        
    if len(dataloader.dataset) == 0:
        return
        
    sample = next(iter(dataloader))
    tabular_dim = sample['tabular'].shape[-1]
    in_channels = sample['vision'].shape[2]
    
    # Khởi tạo mô hình và đóng băng trọng số (Freeze)
    model = HurricastCNNEncoder(in_channels=in_channels, tabular_dim=tabular_dim)
    model.load_state_dict(torch.load(BASE_DIR / 'api' / 'models' / 'hurricast_multitask.pth', map_location=device))
    model = model.to(device)
    model.eval() # Bật chế độ suy luận
    
    features_df = pd.read_csv(features_csv)
    
    # Tạo sẵn 128 cột cho Vision Embedding
    emb_cols = [f'vision_emb_{i}' for i in range(128)]
    for col in emb_cols:
        features_df[col] = np.nan
        
    print("Start extracting Vision Embeddings 128D...")
    with torch.no_grad():
        for batch in dataloader:
            vision = batch['vision'].to(device)
            tabular = batch['tabular'].to(device)
            last_indices = batch['last_idx'].numpy()
            
            # Lấy vector ẩn từ lớp FC chung áp chót
            _, _, embeddings = model(vision, tabular)
            embeddings = embeddings.cpu().numpy()
            
            # Mapping vào DataFrame gốc
            for i, idx in enumerate(last_indices):
                features_df.loc[idx, emb_cols] = embeddings[i]
                
    # Những dòng không đủ 8 mốc thời gian sẽ bị NA, ta fillna(0)
    features_df = features_df.fillna(0)
    
    output_path = BASE_DIR / 'data' / 'final' / 'features_augmented.csv'
    features_df.to_csv(output_path, index=False)
    print(f"Created augmented dataset at: {output_path}")

if __name__ == "__main__":
    extract()
