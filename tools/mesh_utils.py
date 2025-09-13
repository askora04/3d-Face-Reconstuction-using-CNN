# tools/mesh_utils.py
import open3d as o3d
import numpy as np
import trimesh

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh

def normalize_mesh(mesh):
    verts = np.asarray(mesh.vertices)
    centroid = verts.mean(axis=0)
    verts -= centroid
    scale = np.linalg.norm(verts, axis=1).max()
    verts /= scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh, centroid, scale

def repair_mesh(path_in, path_out):
    t = trimesh.load(path_in)
    t.remove_unreferenced_vertices()
    t = t.fill_holes()  # basic
    t.export(path_out)
