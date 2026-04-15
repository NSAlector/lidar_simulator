# renderer/sphere_renderer.py
from renderer.base import BaseRenderer
import numpy as np

class SphereRenderer(BaseRenderer):
    def __init__(self, spheres, camera, plane, light):
        super().__init__(camera, plane, light)
        self.spheres = spheres
    
    def _find_nearest(self, origin, direction):
        t_min = float('inf')
        hit_sphere = None
        hit_point = None
        normal = None
        
        for sphere in self.spheres:
            t = sphere.intersect(direction, origin)
            if t is not None and t > 1e-4 and t < t_min:
                t_min = t
                hit_sphere = sphere
                hit_point = origin + direction * t
                
                if hasattr(sphere, 'sphere'):  # TexturedSphere
                    center = sphere.sphere.center
                else:
                    center = sphere.center
                
                normal = self._normalize(hit_point - center)
        
        return t_min, hit_sphere, hit_point, normal
    
    def _get_material_and_color(self, sphere, hit_point):
        if hasattr(sphere, 'get_color'):
            material = sphere.sphere.material
            base_color = sphere.get_color(hit_point)
        else:
            material = sphere.material
            base_color = material.color
        return material, base_color