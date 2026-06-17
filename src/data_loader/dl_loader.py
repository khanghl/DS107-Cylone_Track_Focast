import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path

class MultimodalStormDataset(Dataset):
    def __init__(self, features_csv, target_csv, grids_dir, sequence_length=8):
        self.features_df = pd.read_csv(features_csv)
        self.target_df = pd.read_csv(target_csv)
        self.grids_dir = grids_dir
        self.seq_len = sequence_length
        self.samples = self._build_sequences()
        
    def _build_sequences(self):
        samples = []
        # Group by SID to create time series sequences
        grouped = self.features_df.groupby('SID')
        
        for sid, group in grouped:
            indices = group.index.values
            try:
                # Load mapping between dataframe index and grid array index
                grid_indices = np.load(f"{self.grids_dir}/{sid}_indices.npy")
                idx_map = {global_idx: local_idx for local_idx, global_idx in enumerate(grid_indices)}
            except FileNotFoundError:
                continue
                
            for i in range(len(indices) - self.seq_len + 1):
                seq_indices = indices[i:i+self.seq_len]
                # Check if all 8 steps have valid grid images
                if all(idx in idx_map for idx in seq_indices):
                    # Target is defined at the end of the 24h history sequence (T0)
                    last_idx = seq_indices[-1]
                    target_row = self.target_df.loc[last_idx]
                    
                    if pd.notna(target_row['DELTA_LAT_24h']) and pd.notna(target_row['TARGET_WIND_24h']):
                        local_grid_indices = [idx_map[idx] for idx in seq_indices]
                        samples.append({
                            'sid': sid,
                            'seq_indices': seq_indices,
                            'grid_indices': local_grid_indices,
                            'target_lat': target_row['DELTA_LAT_24h'],
                            'target_lon': target_row['DELTA_LON_24h'],
                            'target_wind': target_row['TARGET_WIND_24h']
                        })
        return samples

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sid = sample['sid']
        
        # Memory-mapped read to avoid loading huge arrays into RAM
        grids = np.load(f"{self.grids_dir}/{sid}_grids.npy", mmap_mode='r')
        grid_seq = grids[sample['grid_indices']] # Shape: (8, Channels, 25, 25)
        
        # Load tabular features
        tab_seq = self.features_df.loc[sample['seq_indices']].select_dtypes(include=[np.number]).values
        
        # Target: [Delta_Lat, Delta_Lon, Delta_Wind]
        target = np.array([sample['target_lat'], sample['target_lon'], sample['target_wind']], dtype=np.float32)
        
        return {
            'vision': torch.FloatTensor(grid_seq),
            'tabular': torch.FloatTensor(tab_seq),
            'target': torch.FloatTensor(target),
            'last_idx': sample['seq_indices'][-1]
        }

def get_dataloader(features_csv, target_csv, grids_dir, batch_size=32, shuffle=True):
    dataset = MultimodalStormDataset(features_csv, target_csv, grids_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
