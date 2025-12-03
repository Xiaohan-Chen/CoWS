import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, feature_size, dim_ffd, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=1, \
            dim_feedforward=dim_ffd, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
    def forward(self, x):
        return self.encoder(x)