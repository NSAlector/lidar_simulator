import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from geometry.mesh import Mesh

@dataclass
class MeshGroup:
    meshes: List[Mesh]
    
    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64]
    ) -> Optional[tuple]:
        
        t_min = float('inf')
        hit_result = None
        
        for mesh in self.meshes:
            result = mesh.intersect(guiding_vector, start_vector)
            if result:
                t = result[0]
                if t < t_min:
                    t_min = t
                    hit_result = result
        
        return hit_result