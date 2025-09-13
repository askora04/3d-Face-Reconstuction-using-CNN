# models/skull_regressor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        # x: B x N x 3
        x = x.permute(0,2,1)  # B x 3 x N
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  # B x 1024 x N
        x = torch.max(x, 2)[0]     # B x 1024
        x = F.relu(self.fc(x))
        return x

class SkullToFlame(nn.Module):
    def __init__(self, flame_shape_dim=100, flame_expr_dim=50, latent_dim=512):
        super().__init__()
        self.encoder = PointNetEncoder(in_ch=3)
        self.fc_shape = nn.Linear(latent_dim, flame_shape_dim)
        self.fc_expr  = nn.Linear(latent_dim, flame_expr_dim)
        self.fc_pose  = nn.Linear(latent_dim, 6)  # small pose params

    def forward(self, pts):
        z = self.encoder(pts)    # B x latent_dim
        shape = self.fc_shape(z)
        expr  = self.fc_expr(z)
        pose  = self.fc_pose(z)
        return {'shape': shape, 'expr': expr, 'pose': pose}
