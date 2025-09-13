# models/flame_wrapper.py
import torch
import numpy as np

class FlameWrapper:
    def __init__(self, flame_model_path, device='cpu'):
        # placeholder: load the FLAME model object (e.g., from official code)
        from some_flame_library import FLAME  # replace with actual import
        self.device = device
        self.flame = FLAME(flame_model_path).to(device)

    def params_to_mesh(self, shape_params, expr_params, pose_params):
        # shape_params: [B, n_shape]
        # expr_params: [B, n_expr]
        # pose_params: [B, 3] (neck/jaw) etc
        verts, faces = self.flame(shape_params, expr_params, pose_params)
        # verts: [B, V, 3]
        return verts.detach().cpu().numpy(), faces
