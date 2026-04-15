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
        
        t, hit_point, normal, color, material = result
        self.last_material = material
        self.last_color = color
        return t, self.mesh, hit_point, normal
    
    def _get_material_and_color(self, mesh, hit_point):
        return self.last_material, self.last_color