import torch
import torch.nn as nn
import torch.nn.functional as F

class PointTransformerLayer(nn.Module):
    def __init__(self, d_model, k=16):
        super().__init__()
        self.k = k
        self.to_qkv = nn.Linear(d_model, 3 * d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, xyz):
        # x: (B, N, C)  |  xyz: (B, N, 3)
        B, N, _ = x.shape

        # find k-NN in xyz
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz)   # (B, N, N)
            knn_idx = dist.topk(self.k, largest=False).indices   # (B,N,k)

        qkv = self.to_qkv(x).view(B, N, 3, -1)  # (B,N,3,C)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # gather k neighbors
        k_neigh = torch.gather(k.unsqueeze(2).expand(-1, -1, self.k, -1), 1, knn_idx.unsqueeze(-1).expand(-1,-1,-1,k.shape[-1]))
        v_neigh = torch.gather(v.unsqueeze(2).expand(-1, -1, self.k, -1), 1, knn_idx.unsqueeze(-1).expand(-1,-1,-1,v.shape[-1]))

        attn = torch.einsum('bnd,bnsk->bns', q, k_neigh) / (k.shape[-1] ** 0.5)
        attn = torch.softmax(attn, dim=2).unsqueeze(-1)

        x_out = (attn * v_neigh).sum(dim=2)
        return self.fc(x_out) + x
