# renderer/base.py
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
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    
    def _reflect(self, I: np.ndarray, N: np.ndarray) -> np.ndarray:
        """
        Зеркальное отражение (формула 2.12 из документа)
        R = I - 2(I·N)N
        """
        return I - 2 * np.dot(I, N) * N
    
    def _refract(self, I: np.ndarray, N: np.ndarray, n1: float, n2: float) -> np.ndarray:
        """
        Преломление по векторной формуле Снеллиуса (формула 1.6 из документа)
        I - падающий луч (unit)
        N - нормаль к поверхности (unit)
        n1 - IOR среды откуда идет луч
        n2 - IOR среды куда входит луч
        
        Возвращает None при полном внутреннем отражении
        """
        r = n1 / n2
        c1 = np.dot(N, I)
        
        # Управление знаком нормали как в документе
        if c1 < 0:
            c1 = -c1
        else:
            N = -N
            r = 1.0 / r
        
        # Проверка на полное внутреннее отражение
        discriminant = 1 - r * r * (1 - c1 * c1)
        if discriminant < 0:
            return None
        
        c2 = np.sqrt(discriminant)
        T = r * I + (r * c1 - c2) * N
        
        return self._normalize(T)
    
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
    
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, depth: int, max_depth: int) -> np.ndarray:
        if depth > max_depth:
            return np.array([100, 100, 100])
        
        t_min, hit_obj, hit_point, normal = self._find_nearest(origin, direction)
        
        if hit_obj is None:
            # Градиент неба
            t = 0.5 * (direction[1] + 1.0)
            return np.array([100 + t * 155, 100 + t * 155, 255]) * 0.8
        
        # Нормаль должна быть направлена против падающего луча
        if np.dot(direction, normal) > 0:
            normal = -normal
        
        material, base_color = self._get_material_and_color(hit_obj, hit_point)
        light_dir = self._normalize(self.light.position - hit_point)
        
        # Проверка тени
        shadow_origin = hit_point + normal * 1e-4
        t_shadow, shadow_obj, _, _ = self._find_nearest(shadow_origin, light_dir)
        in_shadow = shadow_obj is not None and t_shadow < 1e-3
        
        color = np.zeros(3)
        remaining = 1.0
        
        # Отражение
        if material.reflection > 0 and remaining > 0:
            reflect_dir = self._reflect(direction, normal)
            reflect_color = self._cast_ray(hit_point + normal * 1e-4, reflect_dir, depth + 1, max_depth)
            reflect_weight = material.reflection * remaining
            color += reflect_color * reflect_weight
            remaining -= reflect_weight
        
        # Преломление
        if material.refraction > 0 and remaining > 0:
            n1 = 1.0  # Воздух
            n2 = material.refractive_index
            refract_dir = self._refract(direction, normal, n1, n2)
            
            if refract_dir is not None:
                # Луч внутри объекта - смещаем внутрь
                inside_origin = hit_point - normal * 1e-4
                refract_color = self._cast_ray(inside_origin, refract_dir, depth + 1, max_depth)
                refract_weight = material.refraction * remaining
                color += refract_color * refract_weight
                remaining -= refract_weight
        
        # Диффузное освещение
        if material.diffuse > 0 and remaining > 0:
            if in_shadow:
                diffuse = 0.15
            else:
                diffuse = max(0.15, np.dot(normal, light_dir))
            diffuse *= material.diffuse
            color += base_color * diffuse * remaining
        
        # Специлярное освещение
        if material.specular > 0 and not in_shadow and remaining > 0:
            view_dir = self._normalize(self.camera.position - hit_point)
            reflect_dir = self._reflect(light_dir, normal)
            specular = max(0, np.dot(view_dir, reflect_dir)) ** material.shininess
            color += np.array([255, 255, 255]) * specular * material.specular * remaining
        
        return np.clip(color, 0, 255)