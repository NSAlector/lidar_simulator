import math
import numpy as np

def build_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Builds a 3x3 rotation matrix from yaw (Y), pitch (X), and roll (Z) angles in degrees.
    """
    cy = math.cos(math.radians(yaw))
    sy = math.sin(math.radians(yaw))
    cp = math.cos(math.radians(pitch))
    sp = math.sin(math.radians(pitch))
    cr = math.cos(math.radians(roll))
    sr = math.sin(math.radians(roll))
    
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    
    return Ry @ Rx @ Rz

def compute_mesh_normalization(vertices: np.ndarray):
    """
    Computes center and scale for a given contiguous (N, 3) mesh vertices array
    to normalize it to a [-1, 1] bounding box.
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    size = (max_coords - min_coords).max()
    scale = 2.0 / size if size > 0 else 1.0
    return center, scale

def transform_vertex(vertex: np.ndarray, rot_matrix: np.ndarray, offset: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """
    Transforms a single vertex or an array of vertices.
    If applying to an Nx3 array, the operation is broadcasting-friendly.
    """
    centered_scaled = (vertex - center) * scale
    if len(centered_scaled.shape) == 1:
        rotated = rot_matrix @ centered_scaled
    else:
        rotated = (rot_matrix @ centered_scaled.T).T
    return rotated + offset
