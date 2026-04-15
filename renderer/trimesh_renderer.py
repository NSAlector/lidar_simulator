import time
import numpy as np
import trimesh
from core.camera import Camera
from core.plane import Plane
from core.light import Light
from core.material import Material
from PIL import Image
import os
import sys

class TrimeshRenderer:
    def __init__(self, mesh_path, material, camera, plane, light):
        self.mesh = trimesh.load(mesh_path)
        self.material = material
        self.camera = camera
        self.plane = plane
        self.light = light
        self.ray_intersector = self.mesh.ray
    
    def render(self, width, height):
        image = np.full((height, width, 3), [100, 100, 100], dtype=np.uint8)
        total_pixels = width * height
        pixels_done = 0
        
        batch_size = 10000
        
        for batch_start in range(0, total_pixels, batch_size):
            batch_end = min(batch_start + batch_size, total_pixels)
            batch_indices = list(range(batch_start, batch_end))
            
            ray_origins = []
            ray_directions = []
            
            for idx in batch_indices:
                y = idx // width
                x = idx % width
                u = (x + 0.5) / width - 0.5
                v = 0.5 - (y + 0.5) / height
                direction = self.camera.forward + u * self.camera.viewport_width * self.camera.right + v * self.camera.viewport_height * self.camera.real_up
                direction = direction / np.linalg.norm(direction)
                ray_origins.append(self.camera.position)
                ray_directions.append(direction)
            
            ray_origins = np.array(ray_origins)
            ray_directions = np.array(ray_directions)
            
            try:
                locations, index_ray, index_tri = self.ray_intersector.intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                    multiple_hits=False
                )
                
                for i, ray_idx in enumerate(index_ray):
                    original_idx = batch_start + ray_idx
                    y = original_idx // width
                    x = original_idx % width
                    if 0 <= y < height and 0 <= x < width:
                        normal = self.mesh.face_normals[index_tri[i]]
                        light_dir = self.light.position - locations[i]
                        light_dir = light_dir / np.linalg.norm(light_dir)
                        diffuse = max(0.1, np.dot(normal, light_dir))
                        color = self.material.color * diffuse
                        image[y, x] = np.clip(color, 0, 255).astype(np.uint8)
            except:
                pass
            
            pixels_done += len(batch_indices)
            percent = (pixels_done / total_pixels) * 100
            sys.stdout.write(f"\rРендеринг: {pixels_done}/{total_pixels} ({percent:.1f}%)")
            sys.stdout.flush()
        
        print()
        return image