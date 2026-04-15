import numpy as np

from numba import njit

from tof_function_parallel import numba_triangle_intersect

class IncorrectCoordinates(Exception):
    pass

@njit
def numba_check_ray_box_intersect(
    ray_direction: np.ndarray, 
    ray_start: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray
) -> bool:
    """
    Numba realization of check_ray_box_intersect.
    """
    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        i_min = p_min[i]
        i_max = p_max[i]

        if abs(ray_direction[i]) < 1e-11:
            if ray_start[i] < i_min or ray_start[i] > i_max:
                return False
            continue
        
        t_1 = (i_min - ray_start[i]) / ray_direction[i]
        t_2 = (i_max - ray_start[i]) / ray_direction[i]

        t_left = min(t_1, t_2)
        t_right = max(t_1, t_2)
        
        t_min = max(t_min, t_left)
        t_max = min(t_max, t_right)
        
        if t_min > t_max:
            return False
    
    return t_max >= max(t_min, 0.0)

@njit
def numba_check_triangle_in_box(
    box_min: np.ndarray,
    box_max: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray
) -> bool:
    if np.all(v1 >= box_min) and np.all(v1 <= box_max):
        return True
    if np.all(v2 >= box_min) and np.all(v2 <= box_max):
        return True
    if np.all(v3 >= box_min) and np.all(v3 <= box_max):
        return True
    return False

class Box:
    """
    Class for bounding box in 3D.
    """
    def __init__(self, p_min: np.ndarray, p_max: np.ndarray) -> None:
        if np.any(p_min > p_max):
            raise IncorrectCoordinates("Coordinates in p_max are smaller than in p_min")
        self._p_min = p_min
        self._p_max = p_max

    def check_ray_intersect(
        self,
        ray_direction: np.ndarray, 
        ray_start: np.ndarray
    ) -> bool:
        """
        Check intersection of ray and box.

        Args:
            ray_direction: direction vector of ray.
            ray_start: start point of ray.
        
        Returns:
            Bool flag. flag is True if ray intersect box. 
        """
        return numba_check_ray_box_intersect(ray_direction, ray_start, self._p_min, self._p_max)
    
    def get_center(self) -> np.ndarray:
        return (self._p_min + self._p_max) / 2

    @property
    def p_min(self) -> np.ndarray:
        return self._p_min
    
    @property
    def p_max(self) -> np.ndarray:
        return self._p_max

class TreeNode:
    """
    Class for node of Octree.
    """
    def __init__(
        self, 
        box: Box,
        triangles: list,
        depth: int = 0
    ) -> None:
        self._box: Box = box
        self._triangles = triangles
        self._childrens: list[TreeNode] = []
        self._depth = depth
        self._is_leaf = True

    def divide(self, max_triangles: int = 150, max_depth: int = 20) -> None:
        """
        Divide TreeNode into 8 subnodes.

        Args:
            max_triangles: max amount of triangles in one node.
            max_depth: max depth of the node.
        """
        if len(self._triangles) <= max_triangles or self._depth >= max_depth:
            return
        
        self._is_leaf = False

        center = self._box.get_center()

        for i in range(8):
            child_min = np.copy(self._box.p_min)
            child_max = np.copy(self._box.p_max)

            if i & 1:
                child_min[0] = center[0]
            else:
                child_max[0] = center[0]

            if i & 2:
                child_min[1] = center[1]
            else:
                child_max[1] = center[1]

            if i & 4:
                child_min[2] = center[2]
            else:
                child_max[2] = center[2]

            child = Box(child_min, child_max)
            child_node = TreeNode(child, [], self._depth + 1)
            self._childrens.append(child_node)

        for child in self._childrens:
            for triangle in self._triangles:
                if numba_check_triangle_in_box(
                    child._box.p_min, child._box.p_max,
                    triangle._p1.coords, triangle._p2.coords, triangle._p3.coords
                ):
                    child._triangles.append(triangle)

        self._triangles = []

        for child in self._childrens:
            child.divide(max_triangles, max_depth)

    def ray_intersect(
        self,
        ray_start: np.ndarray,
        ray_direction: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Find ray intersection with triangle in the tree node.

        Args:
            ray_start: start point of the ray.
            ray_direction: direction vector of the ray.

        Returns:
            (point of intersection | None, t | -1.0).
        """
        if not numba_check_ray_box_intersect(
            ray_direction, ray_start, self._box.p_min, self._box.p_max
        ):
            return None, -1.0

        if self._is_leaf:
            best_point = None
            t_min = np.inf

            for triangle in self._triangles:
                point, t = numba_triangle_intersect(
                    ray_start,
                    ray_direction,
                    triangle._p1.coords,
                    triangle._p2.coords,
                    triangle._p3.coords,
                    triangle.normal,
                    triangle.D
                )
                if point is not None and t < t_min:
                    best_point = point
                    t_min = t
            
            return best_point, t_min
        else:
            child_best_point = None
            child_t_min = np.inf

            for child in self._childrens:
                

                point, t = child.ray_intersect(ray_start, ray_direction)
                
                if point is not None and t < child_t_min:
                    child_best_point = point
                    child_t_min = t

            return child_best_point, child_t_min

class Octree:
    """
    Class for Octree.
    """
    def __init__(
        self,
        triangles: list,
        max_triangles: int = 150,
        max_depth: int = 20,
    ) -> None:
        self._triangles = triangles
        self._max_depth = max_depth
        self._max_triangles = max_triangles

        total_box = self._calculate_total_box()
        self._root = TreeNode(total_box, triangles)

        self.build_tree()

    def build_tree(self) -> None:
        """
        Build octree.
        """
        self._root.divide(self._max_triangles, self._max_depth)

    def _calculate_total_box(self, margin: float = 0.01) -> Box:
        """
        Calculate general box.

        Args:
            margin: little margin which will be added to the box.

        Returns:
            General box that include all triangles.
        """
        p_min = np.array([np.inf, np.inf, np.inf])
        p_max = np.array([-np.inf, -np.inf, -np.inf])

        for triangle in self._triangles:
            for v in triangle.vertices:
                p_min = np.minimum(p_min, v.coords)
                p_max = np.maximum(p_max, v.coords)

        p_min -= margin
        p_max += margin

        return Box(p_min, p_max)
    
    def ray_intersect(
        self,
        ray_start: np.ndarray,
        ray_direction: np.ndarray
    ) -> tuple[np.ndarray | None, float]:
        """
        Find ray intersection with triangle in octree.

        Args:
            ray_start: start point of the ray.
            ray_direction: direction vector of the ray.

        Returns:
            (point of intersection | None, t | -1.0).
        """
        return self._root.ray_intersect(ray_start, ray_direction)
