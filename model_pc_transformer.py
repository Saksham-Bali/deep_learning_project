import torch
import torch.nn as nn

class PositionalEncodingMLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords):
        # coords: (B, N, 3)
        return self.net(coords)


class PointTransformerSeg(nn.Module):
    """
    Simple transformer-based per-point segmentation model.
    - Input: xyz (B,N,3), feats (B,N,F)
    - Uses transformer encoder layers over points.
    """

    def __init__(self,
                 in_channels=6,      # features: RGB + intensity + scan_angle + gps_time
                 num_classes=9,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=256,
                 dropout=0.1):
        super().__init__()

        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        self.pos_mlp = PositionalEncodingMLP(in_dim=3, hidden_dim=64, out_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B,N,C)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, xyz, feats):
        """
        xyz:   (B, N, 3)
        feats: (B, N, F)
        """
        x = self.input_mlp(feats)           # (B,N,C)
        pos = self.pos_mlp(xyz)            # (B,N,C)
        x = x + pos                        # add positional encoding

        x = self.encoder(x)                # (B,N,C)
        logits = self.classifier(x)        # (B,N,num_classes)
        return logits
