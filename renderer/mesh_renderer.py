# renderer/mesh_renderer.py
from renderer.base import BaseRenderer
import numpy as np

class MeshRenderer(BaseRenderer):
    def __init__(self, mesh, camera, plane, light):
        super().__init__(camera, plane, light)
        self.mesh = mesh
        self.last_material = None
        self.last_color = None
    
    def _find_nearest(self, origin, direction):
        result = self.mesh.intersect(direction, origin)
        if result is None:
            return float('inf'), None, None, None
        
        # result может быть: (t, hit_point, normal, color, material) или (t, hit_point, normal, color, material, uv)
        t = result[0]
        hit_point = result[1]
        normal = result[2]
        color = result[3]
        material = result[4]
        
        self.last_material = material
        self.last_color = color
        return t, self.mesh, hit_point, normal
    
    def _get_material_and_color(self, mesh, hit_point):
        return self.last_material, self.last_color