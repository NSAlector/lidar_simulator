import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Box:
    min: np.ndarray
    max: np.ndarray
    
    def intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> bool:
        inv_dir = 1.0 / ray_dir
        t1 = (self.min - ray_origin) * inv_dir
        t2 = (self.max - ray_origin) * inv_dir
        
        tmin = np.max(np.minimum(t1, t2))
        tmax = np.min(np.maximum(t1, t2))
        
        return tmax >= max(tmin, 0.0)

class BVHNode:
    def __init__(self, triangles, box: Box, left=None, right=None):
        self.box = box
        self.triangles = triangles
        self.left = left
        self.right = right
        self.is_leaf = left is None and right is None

def build_bvh(triangles, axis=0, max_triangles=10):
    if len(triangles) <= max_triangles:
        box = Box(
            min=np.min([np.min([t.v0, t.v1, t.v2], axis=0) for t in triangles], axis=0),
            max=np.max([np.max([t.v0, t.v1, t.v2], axis=0) for t in triangles], axis=0)
        )
        return BVHNode(triangles, box)
    
    centroids = np.array([(t.v0 + t.v1 + t.v2) / 3 for t in triangles])
    sorted_indices = np.argsort(centroids[:, axis])
    sorted_triangles = [triangles[i] for i in sorted_indices]
    
    mid = len(sorted_triangles) // 2
    left = build_bvh(sorted_triangles[:mid], (axis + 1) % 3)
    right = build_bvh(sorted_triangles[mid:], (axis + 1) % 3)
    
    box = Box(
        min=np.minimum(left.box.min, right.box.min),
        max=np.maximum(left.box.max, right.box.max)
    )
    
    return BVHNode([], box, left, right)

def intersect_bvh(node, ray_origin, ray_dir, epsilon=1e-6):
    if not node.box.intersect(ray_origin, ray_dir):
        return None
    
    if node.is_leaf:
        t_min = float('inf')
        hit_tri = None
        
        for tri in node.triangles:
            if np.sum(ray_dir * tri.normal) > 0:
                continue
                
            edge1 = tri.v1 - tri.v0
            edge2 = tri.v2 - tri.v0
            
            pvec = np.cross(ray_dir, edge2)
            det = np.sum(edge1 * pvec)
            
            if abs(det) < epsilon:
                continue
            
            inv_det = 1.0 / det
            tvec = ray_origin - tri.v0
            
            u = np.sum(tvec * pvec) * inv_det
            if u < 0.0 or u > 1.0:
                continue
            
            qvec = np.cross(tvec, edge1)
            v = np.sum(ray_dir * qvec) * inv_det
            if v < 0.0 or u + v > 1.0:
                continue
            
            t = np.sum(edge2 * qvec) * inv_det
    
            if t > epsilon and t < t_min:
                t_min = t
                hit_tri = tri
        
        if hit_tri:
            return (t_min, hit_tri)
        return None
    
    left_hit = intersect_bvh(node.left, ray_origin, ray_dir)
    right_hit = intersect_bvh(node.right, ray_origin, ray_dir)
    
    if not left_hit and not right_hit:
        return None
    if not left_hit:
        return right_hit
    if not right_hit:
        return left_hit
    
    return left_hit if left_hit[0] < right_hit[0] else right_hit