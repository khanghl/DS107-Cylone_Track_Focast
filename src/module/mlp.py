import torch
import torch.nn as nn

class StormMLP(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.3):
        super(StormMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        return self.network(x)
