"""
Module for parallel realization of tof modeling.
"""
import numpy as np
from numba import njit, prange

@njit
def numba_distance_to(
    point1_coords: np.ndarray, 
    point2_coords: np.ndarray
) -> float:
    return np.sqrt(np.sum((point1_coords - point2_coords) ** 2))

@njit
def numba_sphere_intersect(
    ray_start: np.ndarray, 
    ray_direction: np.ndarray, 
    sphere_center: np.ndarray, 
    sphere_R: float
) -> tuple[float | None, float]:
    delta_x = ray_start - sphere_center
    A = np.dot(ray_direction, ray_direction)
    B = 2 * np.dot(ray_direction, delta_x)
    C = np.dot(delta_x, delta_x) - (sphere_R ** 2)
    D = B ** 2 - 4 * A * C
    
    if D < 0:
        return None, -1.0
    
    t1 = (-B + np.sqrt(D)) / (2 * A)
    t2 = (-B - np.sqrt(D)) / (2 * A)
    
    t_min = np.inf
    found = False
    
    if t1 >= 0 and t1 < t_min:
        t_min = t1
        found = True
    if t2 >= 0 and t2 < t_min:
        t_min = t2
        found = True
    
    if found:
        return ray_start + t_min * ray_direction, t_min
    return None, -1.0

@njit
def numba_check_point_in_triangle(
    point_coords: np.ndarray, 
    p1: np.ndarray, 
    p2: np.ndarray, 
    p3: np.ndarray,
    normal: np.ndarray,
    D: float
) -> bool:
    eps = 1e-10

    x = point_coords[0]
    y = point_coords[1]
    z = point_coords[2]
    
    A = normal[0]
    B = normal[1]
    C = normal[2]

    check = A * x + B * y + C * z + D

    if not np.allclose(check, 0, eps):
        return False

    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p1 - p3

    v1_p = point_coords - p1
    v2_p = point_coords - p2
    v3_p = point_coords - p3

    cross_1 = np.cross(v1, v1_p)
    cross_2 = np.cross(v2, v2_p)
    cross_3 = np.cross(v3, v3_p)

    sign_1 = np.dot(cross_1, normal)
    sign_2 = np.dot(cross_2, normal)
    sign_3 = np.dot(cross_3, normal)

    positive = (sign_1 > eps) or (sign_2 > eps) or (sign_3 > eps)
    negative = (sign_1 < -eps) or (sign_2 < -eps) or (sign_3 < -eps)

    if negative and positive:
        return False

    return True


@njit
def numba_triangle_intersect(
    ray_start: np.ndarray, 
    ray_direction: np.ndarray, 
    v0: np.ndarray, 
    v1: np.ndarray, 
    v2: np.ndarray, 
    normal: np.ndarray, 
    D: float
) -> tuple[np.ndarray | None, float]:
    normal = normal.astype(np.float64)
    a = np.dot(ray_direction, normal)
    b = -D - np.dot(ray_start, normal)
    
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            if numba_check_point_in_triangle(ray_start, v0, v1, v2, normal, D):
                return ray_start, 0.0
            
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            t_min = np.inf
            best_point = None
            
            for i in range(3):
                p1, p2 = edges[i]
                vec = p1 - p2
                A = np.column_stack((ray_direction, -vec))
                b_vec = p1 - ray_start
                
                try:
                    t, u = np.linalg.lstsq(A, b_vec)[0]
                    
                    if t >= -1e-10 and 0 <= u <= 1:
                        point = ray_start + t * ray_direction
                        if numba_check_point_in_triangle(point, v0, v1, v2, normal, D):
                            if t < t_min:
                                t_min = t
                                best_point = point
                except:
                    pass
            
            if best_point is not None:
                return best_point, t_min
            return None, -1.0
        else:
            return None, -1.0
    
    t = b / a
    
    if t < 0:
        return None, -1.0
    
    point = ray_start + t * ray_direction
    if numba_check_point_in_triangle(point, v0, v1, v2, normal, D):
        return point, t
    return None, -1.0

@njit(parallel=True)
def numba_process_all_rays(
    rays_start: np.ndarray, 
    rays_direction: np.ndarray, 
    object_type: int, 
    object_params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallel processing of rays with numba.
    
    Args:
        n_rays: total number of rays.
        rays_start: start of all rays.
        rays_direction: array of directions (N, 3).
        object_type: object type (0 - sphere, 1 - triangle, 2 - figure).
        object_params: object parameters.
    
    Returns:
        distances: array of distances.
        points: array of points.

    UPD:
    Triangle in array of triangles is a 13 float numbers:
        [0:3]   - coords p1: (v0_x, v0_y, v0_z)
        [3:6]   - coords p2: (v1_x, v1_y, v1_z)      
        [6:9]   - coords p3: (v2_x, v2_y, v2_z)
        [9:12]  - normal (normal_x, normal_y, normal_z) 
        [12]    - triangle D   
    """
    n_rays = rays_start.shape[0]
    distances = np.full(n_rays, np.nan, dtype=np.float64)
    points = np.full((n_rays, 3), np.nan, dtype=np.float64)
    
    if object_type == 0:
        sphere_center = object_params[:3]
        sphere_R = object_params[3]
        
        for i in prange(n_rays):
            point, t = numba_sphere_intersect(
                rays_start[i], rays_direction[i], 
                sphere_center, sphere_R
            )
            if point is not None:
                distances[i] = t * np.linalg.norm(rays_direction[i])
                points[i] = point
    
    elif object_type == 1:
        v0 = object_params[0:3]
        v1 = object_params[3:6]
        v2 = object_params[6:9]
        normal = object_params[9:12]
        D = object_params[12]
        
        for i in prange(n_rays):
            point, t = numba_triangle_intersect(
                rays_start[i], rays_direction[i],
                v0, v1, v2, normal, D
            )
            if point is not None:
                distances[i] = t * np.linalg.norm(rays_direction[i])
                points[i] = point

    elif object_type == 2:
        n_triangles = object_params.shape[0] // 13
        for i in prange(n_rays):
            min_dist = np.inf
            best_point = None
            
            for j in range(n_triangles):
                idx = j * 13
                v0 = object_params[idx:idx+3]
                v1 = object_params[idx+3:idx+6]
                v2 = object_params[idx+6:idx+9]
                normal = object_params[idx+9:idx+12]
                D = object_params[idx+12]
                
                point, t = numba_triangle_intersect(
                    rays_start[i], rays_direction[i],
                    v0, v1, v2, normal, D
                )
                
                if point is not None:
                    dist = t * np.linalg.norm(rays_direction[i])
                    if dist < min_dist:
                        min_dist = dist
                        best_point = point
            
            if best_point is not None:
                distances[i] = min_dist
                points[i] = best_point
    
    return distances, points

@njit
def numba_generate_rays(
    position: np.ndarray, 
    direction: np.ndarray, 
    width: int, 
    height: int, 
    fov_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(direction)
    e1 = direction / d
    
    a = np.array([1.0, 0.0, 0.0])
    if np.all(np.abs(a - e1) < 1e-10):
        a = np.array([0.0, 1.0, 0.0])
    
    e2 = np.cross(e1, a)
    e2_norm = np.linalg.norm(e2)
    if e2_norm > 1e-10:
        e2 /= e2_norm
    
    e3 = np.cross(e1, e2)
    e3_norm = np.linalg.norm(e3)
    if e3_norm > 1e-10:
        e3 /= e3_norm
    
    center = position + e1
    
    ratio = height / width
    half_width = np.tan(fov_rad / 2)
    half_height = ratio * half_width
    
    abscissa = np.linspace(-half_width, half_width, width)
    ordinate = np.linspace(-half_height, half_height, height)
    
    n_rays = width * height
    rays_start = np.zeros((n_rays, 3))
    rays_direction = np.zeros((n_rays, 3))
    
    idx = 0
    for y in ordinate:
        for x in abscissa:
            pixel_point = center + x * e2 + y * e3
            ray_direction = pixel_point - position
            norm = np.linalg.norm(ray_direction)
            if norm > 1e-10:
                ray_direction /= norm
            
            rays_start[idx] = position
            rays_direction[idx] = ray_direction
            idx += 1
    
    return rays_start, rays_direction