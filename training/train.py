# training/train.py (sketch)
import torch
from torch.utils.data import DataLoader
from models.skull_regressor import SkullToFlame
from models.flame_wrapper import FlameWrapper
from training.losses import mesh_vertex_loss, landmark_loss

# dataset should yield (skull_points, gt_face_vertices, landmarks)
# Dataset code omitted for brevity

model = SkullToFlame().to(device)
flame = FlameWrapper('models/flame_model.pkl', device=device)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(epochs):
    for skull_pts, gt_face_verts, gt_landmarks in dataloader:
        skull_pts = skull_pts.to(device)   # [B, N, 3]
        out = model(skull_pts)
        pred_verts, faces = flame.params_to_mesh(out['shape'], out['expr'], out['pose'])
        pred_verts = torch.tensor(pred_verts, device=device)  # [B,V,3]
        loss_v = mesh_vertex_loss(pred_verts, gt_face_verts.to(device))
        loss_l = landmark_loss(pred_verts, gt_landmarks.to(device))
        loss = loss_v + 10.0*loss_l
        optim.zero_grad()
        loss.backward()
        optim.step()
