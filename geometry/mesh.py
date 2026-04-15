import numpy as np
from typing import Optional
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

class Mesh:
    def __init__(self, triangles: list[Triangle]) -> None:
        self.triangles = triangles
        self.bvh_root = build_bvh(triangles) if triangles else None

    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64]
    ) -> Optional[tuple[float, np.ndarray, np.ndarray, np.ndarray, Material]]:
        
        if not self.bvh_root:
            return None
            
        result = intersect_bvh(self.bvh_root, start_vector, guiding_vector)
        
        if result:
            t_min, hit_tri = result
            hit_point = start_vector + guiding_vector * t_min
            return (t_min, hit_point, hit_tri.normal, hit_tri.color, hit_tri.material)
        return None