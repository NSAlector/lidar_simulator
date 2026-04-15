# renderer/unified_renderer.py
import numpy as np
from renderer.base import BaseRenderer
from geometry.sphere import Sphere
from geometry.mesh import Mesh
from geometry.mesh_group import MeshGroup
from typing import List, Union, Optional

class UnifiedRenderer(BaseRenderer):
    """Универсальный рендерер для сфер и мешей с отражениями"""
    
    def __init__(self, objects: List[Union[Sphere, Mesh, MeshGroup]], camera, plane, light):
        super().__init__(camera, plane, light)
        self.objects = objects
        self.last_hit = None
    
    def _find_nearest(self, origin, direction):
        """Находит ближайшее пересечение со всеми объектами"""
        t_min = float('inf')
        hit_obj = None
        hit_point = None
        hit_normal = None
        hit_color = None
        hit_material = None
        hit_uv = None
        
        for obj in self.objects:
            if isinstance(obj, Sphere):
                result = self._intersect_sphere(obj, origin, direction)
                if result:
                    t, point, normal, color, material = result
                    if t < t_min and t > 1e-4:
                        t_min = t
                        hit_obj = obj
                        hit_point = point
                        hit_normal = normal
                        hit_color = color
                        hit_material = material
                        hit_uv = None
                        
            elif isinstance(obj, Mesh):
                result = obj.intersect(direction, origin)
                if result:
                    t = result[0]
                    point = result[1]
                    normal = result[2]
                    color = result[3]
                    material = result[4]
                    uv = result[5] if len(result) > 5 else None
                    
                    if t < t_min and t > 1e-4:
                        t_min = t
                        hit_obj = obj
                        hit_point = point
                        hit_normal = normal
                        hit_color = color
                        hit_material = material
                        hit_uv = uv
                        
            elif isinstance(obj, MeshGroup):
                result = obj.intersect(direction, origin)
                if result:
                    t = result[0]
                    point = result[1]
                    normal = result[2]
                    color = result[3]
                    material = result[4]
                    uv = result[5] if len(result) > 5 else None
                    
                    if t < t_min and t > 1e-4:
                        t_min = t
                        hit_obj = obj
                        hit_point = point
                        hit_normal = normal
                        hit_color = color
                        hit_material = material
                        hit_uv = uv
        
        self.last_hit = (hit_color, hit_material, hit_uv)
        return t_min, hit_obj, hit_point, hit_normal
    
    def _intersect_sphere(self, sphere: Sphere, origin, direction):
        """Пересечение луча со сферой"""
        a = np.sum(direction**2)
        b = 2 * np.sum(direction * (origin - sphere.center))
        c = np.sum((origin - sphere.center)**2) - sphere.radius**2
        
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None
        
        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        t = min(t1, t2) if t1 > 0 else t2
        if t < 0:
            return None
        
        point = origin + direction * t
        normal = self._normalize(point - sphere.center)
        
        return (t, point, normal, sphere.color, sphere.material)
    
    def _get_material_and_color(self, hit_obj, hit_point):
        """Возвращает материал и цвет в точке попадания"""
        if self.last_hit:
            return self.last_hit[1], self.last_hit[0]
        return None, np.array([100, 100, 100])