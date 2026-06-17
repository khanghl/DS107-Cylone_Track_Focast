import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config as config
from models.mlp import StormMLP

class MLPForecastingStrategy:
    def __init__(self, epochs=200, batch_size=64, lr=0.001, patience=15):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        
        self.models = {}
        self.scalers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MLP Strategy initialized using device: {self.device}")

    def train(self, X_train, y_train_lat_dict, y_train_lon_dict, X_val, y_val_lat_dict, y_val_lon_dict):
        # We will train a separate model for each horizon
        for horizon in config.HORIZONS:
            print(f"   [MLP] Training for horizon +{horizon}h...")
            
            # Prepare data
            X_tr_np = X_train.values.astype(np.float32)
            X_va_np = X_val.values.astype(np.float32)
            
            # Scale features
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_np)
            X_va_scaled = scaler.transform(X_va_np)
            self.scalers[horizon] = scaler
            
            y_tr_lat = y_train_lat_dict[horizon].values.astype(np.float32)
            y_tr_lon = y_train_lon_dict[horizon].values.astype(np.float32)
            y_tr_stacked = np.column_stack((y_tr_lat, y_tr_lon))
            
            y_va_lat = y_val_lat_dict[horizon].values.astype(np.float32)
            y_va_lon = y_val_lon_dict[horizon].values.astype(np.float32)
            y_va_stacked = np.column_stack((y_va_lat, y_va_lon))
            
            # Convert to tensors
            train_dataset = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr_stacked))
            val_dataset = TensorDataset(torch.tensor(X_va_scaled), torch.tensor(y_va_stacked))
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            input_dim = X_tr_scaled.shape[1]
            model = StormMLP(input_dim=input_dim).to(self.device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            # Early stopping variables
            best_val_loss = float('inf')
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * batch_X.size(0)
                
                train_loss = train_loss / len(train_loader.dataset)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                
                val_loss = val_loss / len(val_loader.dataset)
                
                # Check early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= self.patience:
                    #print(f"      Early stopping at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                    break
            
            # Load best model weights
            model.load_state_dict(best_model_wts)
            self.models[horizon] = model

    def predict(self, X_test):
        predictions_lat = {}
        predictions_lon = {}
        
        for horizon in config.HORIZONS:
            scaler = self.scalers[horizon]
            model = self.models[horizon]
            
            X_te_np = X_test.values.astype(np.float32)
            X_te_scaled = scaler.transform(X_te_np)
            
            test_dataset = TensorDataset(torch.tensor(X_te_scaled))
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            model.eval()
            preds = []
            with torch.no_grad():
                for batch_X in test_loader:
                    batch_X = batch_X[0].to(self.device)
                    outputs = model(batch_X)
                    preds.append(outputs.cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            predictions_lat[horizon] = preds[:, 0]
            predictions_lon[horizon] = preds[:, 1]
            
        return predictions_lat, predictions_lon
