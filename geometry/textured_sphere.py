import numpy as np
from dataclasses import dataclass
from geometry.sphere import Sphere
from typing import Optional

@dataclass
class TexturedSphere:
    sphere: Sphere
    texture: Optional[np.ndarray] = None
    texture_width: int = 0
    texture_height: int = 0
    
    def __post_init__(self):
        if self.texture is not None:
            h, w = self.texture.shape[:2]
            self.texture_height = h
            self.texture_width = w
    
    def intersect(self, guiding_vector, start_vector):
        return self.sphere.intersect(guiding_vector, start_vector)
    
    def get_color(self, hit_point: np.ndarray) -> np.ndarray:
        if self.texture is None:
            return self.sphere.material.color
        
        p = (hit_point - self.sphere.center) / self.sphere.radius
        u = 0.5 + np.arctan2(p[0], p[2]) / (2 * np.pi)
        v = 0.5 - np.arcsin(p[1]) / np.pi
        
        x = int(u * (self.texture_width - 1))
        y = int(v * (self.texture_height - 1))
        
        return self.texture[y, x]