import numpy as np
import struct
from geometry.mesh import Mesh, Triangle
from core.material import Material

def parse_binary_stl(filepath: str, color: np.ndarray) -> Mesh:
    triangles = []
    material = Material(color=color)
    
    with open(filepath, 'rb') as f:
        f.read(80)
        triangle_count = struct.unpack('<I', f.read(4))[0]
        
        for _ in range(triangle_count):
            normal = struct.unpack('<3f', f.read(12))
            normal = np.array(normal, dtype=np.float64)
            
            v0 = struct.unpack('<3f', f.read(12))
            v1 = struct.unpack('<3f', f.read(12))
            v2 = struct.unpack('<3f', f.read(12))
            
            v0 = np.array(v0, dtype=np.float64)
            v1 = np.array(v1, dtype=np.float64)
            v2 = np.array(v2, dtype=np.float64)
            
            f.read(2)
            
            if np.linalg.norm(normal) < 1e-6:
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
            
            triangles.append(Triangle(v0, v1, v2, normal, color, material))
    
    return Mesh(triangles)