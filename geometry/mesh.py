# geometry/mesh.py
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from geometry.bvh import build_bvh, intersect_bvh
from core.material import Material

@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal: np.ndarray
    color: np.ndarray
    material: Material
    uv0: np.ndarray = None
    uv1: np.ndarray = None
    uv2: np.ndarray = None

class Mesh:
    def __init__(self, triangles: List[Triangle]) -> None:
        self.triangles = triangles
        self.bvh_root = build_bvh(triangles) if triangles else None

    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64]
    ) -> Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray, Material, np.ndarray]]:
        
        if not self.bvh_root:
            return None
            
        result = intersect_bvh(self.bvh_root, start_vector, guiding_vector)
        
        if result:
            t_min, hit_tri = result
            hit_point = start_vector + guiding_vector * t_min
            
            # Вычисление барицентрических координат для UV
            edge1 = hit_tri.v1 - hit_tri.v0
            edge2 = hit_tri.v2 - hit_tri.v0
            pvec = np.cross(guiding_vector, edge2)
            det = np.dot(edge1, pvec)
            
            if abs(det) < 1e-6:
                u = 0
                v = 0
            else:
                inv_det = 1.0 / det
                tvec = start_vector - hit_tri.v0
                u = np.dot(tvec, pvec) * inv_det
                qvec = np.cross(tvec, edge1)
                v = np.dot(guiding_vector, qvec) * inv_det
            
            if hit_tri.uv0 is not None:
                uv = hit_tri.uv0 + u * (hit_tri.uv1 - hit_tri.uv0) + v * (hit_tri.uv2 - hit_tri.uv0)
                return (t_min, hit_point, hit_tri.normal, hit_tri.color, hit_tri.material, uv)
            
            return (t_min, hit_point, hit_tri.normal, hit_tri.color, hit_tri.material, None)
        return None