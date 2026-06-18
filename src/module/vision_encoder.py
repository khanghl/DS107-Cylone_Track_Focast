import torch
import torch.nn as nn

class HurricastCNNEncoder(nn.Module):
    def __init__(self, in_channels=5, tabular_dim=88, cnn_out_dim=128, hidden_dim=128):
        super().__init__()
        
        # Khối Encoder Trích xuất Đặc trưng Không gian (Spatial CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 25x25 -> 12x12
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 12x12 -> 6x6
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 6x6 -> 3x3
            
            nn.Flatten()
        )
        
        # Đưa ảnh đã flatten về không gian nén
        self.cnn_fc = nn.Linear(128 * 3 * 3, cnn_out_dim)
        
        # Khối Decoder Tổng hợp Thời gian (Temporal GRU)
        # Đầu vào GRU = Ảnh đã nén + Đặc trưng Bảng
        gru_input_dim = cnn_out_dim + tabular_dim
        self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        
        # --- Multi-task Learning Heads ---
        # Nhánh 1: Dự đoán Quỹ đạo (Delta Lat, Delta Lon)
        self.track_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Nhánh 2: Dự đoán Cường độ (Delta Wind)
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, vision_seq, tabular_seq):
        """
        vision_seq: Tensor (B, T, C, H, W)
        tabular_seq: Tensor (B, T, F)
        """
        B, T, C, H, W = vision_seq.shape
        
        # Duỗi chuỗi thời gian để đưa vào CNN cùng lúc
        vision_flat = vision_seq.view(B * T, C, H, W)
        cnn_features = self.cnn(vision_flat)
        cnn_features = self.cnn_fc(cnn_features)
        
        # Phục hồi chiều thời gian
        cnn_features = cnn_features.view(B, T, -1)
        
        # Nối đặc trưng ảnh 1D với đặc trưng Tabular 1D
        combined_features = torch.cat([cnn_features, tabular_seq], dim=-1)
        
        # Chạy qua GRU để học chuỗi lịch sử
        gru_out, _ = self.gru(combined_features)
        
        # Lấy trạng thái của bước thời gian cuối cùng (T0)
        last_hidden = gru_out[:, -1, :] # Đây chính là Vision Embedding 128-D
        
        # Dự đoán Multi-task
        track_pred = self.track_head(last_hidden)
        intensity_pred = self.intensity_head(last_hidden)
        
        # Trả về kèm theo last_hidden để dùng cho XGBoost sau này
        return track_pred, intensity_pred, last_hidden
