# renderer/mesh_group_renderer.py
from renderer.base import BaseRenderer
import numpy as np

class MeshGroupRenderer(BaseRenderer):
    def __init__(self, mesh_group, camera, plane, light):
        super().__init__(camera, plane, light)
        self.mesh_group = mesh_group
        self.last_material = None
        self.last_color = None
        self.last_uv = None
    
    def _find_nearest(self, origin, direction):
        result = self.mesh_group.intersect(direction, origin)
        if result is None:
            return float('inf'), None, None, None
        
        # result: (t, hit_point, normal, color, material, uv)
        t = result[0]
        hit_point = result[1]
        normal = result[2]
        color = result[3]
        material = result[4]
        uv = result[5] if len(result) > 5 else None
        
        self.last_material = material
        self.last_color = color
        self.last_uv = uv
        return t, self.mesh_group, hit_point, normal
    
    def _get_material_and_color(self, hit_obj, hit_point):
        if self.last_uv is not None and hasattr(self.last_material, 'texture') and self.last_material.texture is not None:
            texture = self.last_material.texture
            h, w = texture.shape[:2]
            u = self.last_uv[0] % 1.0
            v = self.last_uv[1] % 1.0
            x = int(u * (w - 1))
            y = int(v * (h - 1))
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            tex_color = texture[y, x]
            return self.last_material, tex_color
        
        return self.last_material, self.last_color