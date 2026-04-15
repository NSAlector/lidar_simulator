from core.camera import Camera
from core.plane import Plane
from core.light import Light
import numpy as np
import sys

class BaseRenderer:
    def __init__(self, camera: Camera, plane: Plane, light: Light):
        self.camera = camera
        self.plane = plane
        self.light = light
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)
    
    def _reflect(self, I: np.ndarray, N: np.ndarray) -> np.ndarray:
        return I - 2 * np.dot(I, N) * N
    
    def _refract(self, I: np.ndarray, N: np.ndarray, n1: float, n2: float) -> np.ndarray:
        eta = n1 / n2
        cosi = -max(-1.0, min(1.0, np.dot(I, N)))
        if cosi < 0:
            return self._refract(I, -N, n2, n1)
        k = 1 - eta * eta * (1 - cosi * cosi)
        if k < 0:
            return None
        return eta * I + (eta * cosi - np.sqrt(k)) * N
    
    def _get_ray_direction(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        u = (x + 0.5) / width - 0.5
        v = 0.5 - (y + 0.5) / height
        return self._normalize(
            self.camera.forward +
            u * self.camera.viewport_width * self.camera.right +
            v * self.camera.viewport_height * self.camera.real_up
        )
    
    def render(self, width: int, height: int, max_depth: int = 3) -> np.ndarray:
        image = np.full((height, width, 3), [100, 100, 100], dtype=np.uint8)
        total_pixels = width * height
        pixels_done = 0
        
        for y in range(height):
            for x in range(width):
                direction = self._get_ray_direction(x, y, width, height)
                color = self._cast_ray(self.camera.position, direction, 0, max_depth)
                image[y, x] = np.clip(color, 0, 255).astype(np.uint8)
                
                pixels_done += 1
                if pixels_done % 10000 == 0:
                    percent = (pixels_done / total_pixels) * 100
                    sys.stdout.write(f"\r{pixels_done}/{total_pixels} ({percent:.1f}%)")
                    sys.stdout.flush()
        
        print()
        return image
    
    def _find_nearest(self, origin, direction):
        raise NotImplementedError
    
    def _get_material_and_color(self, hit_obj, hit_point):
        raise NotImplementedError
    
    def _cast_ray(self, origin, direction, depth, max_depth):
        if depth > max_depth:
            return np.array([100, 100, 100])
        
        t_min, hit_obj, hit_point, normal = self._find_nearest(origin, direction)
        
        if hit_obj is None:
            return np.array([100, 100, 100])
        
        if np.dot(direction, normal) > 0:
            normal = -normal
        
        material, base_color = self._get_material_and_color(hit_obj, hit_point)
        light_dir = self._normalize(self.light.position - hit_point)
        
        shadow_origin = hit_point + normal * 1e-4
        t_shadow, shadow_obj, _, _ = self._find_nearest(shadow_origin, light_dir)
        in_shadow = shadow_obj is not None and t_shadow > 1e-4
        
        diffuse = max(0.1, np.dot(normal, light_dir)) if not in_shadow else 0.1
        diffuse *= material.diffuse
        
        color = np.zeros(3)
        
        if material.reflection > 0:
            reflect_dir = self._reflect(direction, normal)
            reflect_color = self._cast_ray(hit_point + normal * 1e-4, reflect_dir, depth + 1, max_depth)
            color += reflect_color * material.reflection
        
        if material.refraction > 0:
            refract_dir = self._refract(direction, normal, 1.0, material.refractive_index)
            if refract_dir is not None:
                refract_color = self._cast_ray(hit_point - normal * 1e-4, refract_dir, depth + 1, max_depth)
                color += refract_color * material.refraction
        
        if material.diffuse > 0:
            color += base_color * diffuse
        
        if material.specular > 0 and not in_shadow:
            view_dir = self._normalize(self.camera.position - hit_point)
            reflect_dir = self._reflect(light_dir, normal)
            specular = max(0, np.dot(view_dir, reflect_dir)) ** material.shininess
            color += np.array([255, 255, 255]) * specular * material.specular
        
        return color