import numpy as np
import trimesh
import fast_simplification
import stl_reader

from oct_tree import Octree


class Point:
    """
    Class for representing point in 3D.
    """

    def __init__(self, coords: np.ndarray) -> None:
        if coords.size == 3:
            self._coords = coords
        else:
            self._coords = np.array([0, 0, 0])

    def __str__(self) -> str:
        return f"[{self._coords[0], self._coords[1], self._coords[2]}]"
    
    @property
    def coords(self) -> np.ndarray:
        return self._coords
    
    @coords.setter
    def coords(self, new_coords: np.ndarray) -> None:
        if new_coords.size == 3:
            self._coords = new_coords
        else:
            raise ValueError("Coords.setter: Incorrect size of new coordinates")
        
    def __eq__(self, other: "Point") -> bool:
        return np.all(self.coords == other.coords)


class Sphere:
    """
    Class for representing sphere in 3D.
    """
    def __init__(self, R: float, center: Point) -> None:
        self.R = R
        self.center = center

class Triangle:
    """
    Class for representing triangle in 3D.
    """
    def __init__(self, p1: Point, p2: Point, p3: Point) -> None:
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3

        self._vertices = (p1, p2, p3)

        v1 = p2.coords - p1.coords
        v2 = p3.coords - p1.coords

        self._normal = np.cross(v1, v2)

        if np.allclose(self._normal, np.zeros(3)):
            raise ValueError("Points are on the same line.")

        self._normal = self._normal / np.linalg.norm(self._normal)

        self._A, self._B, self._C = self._normal
        self._D = -np.dot(self._normal, p1.coords)

    @property
    def vertices(self) -> tuple[Point]:
        return self._vertices

    @property
    def normal(self) -> np.ndarray:
        return self._normal
    
    @property
    def coefficients(self) -> tuple[float]:
        return self._A, self._B, self._C, self._D
    
    @property
    def A(self) -> float:
        return self._A
    
    @property
    def B(self) -> float:
        return self._B
    
    @property
    def C(self) -> float:
        return self._C
    
    @property
    def D(self) -> float:
        return self._D

    def __str__(self) -> str:
        return f"{self._A}x + {self._B}y + {self._C}z + {self._D} = 0"

    def check_point_in_triangle(self, p: Point) -> bool:
        """
        Check if point located in triagnle or not.
        """
        eps = 1e-10
        x, y, z = p.coords
        check = self.A * x + self.B * y + self.C * z + self.D

        if not np.allclose(check, 0, eps):
            return False

        v1 = self._p2.coords - self._p1.coords
        v2 = self._p3.coords - self._p2.coords
        v3 = self._p1.coords - self._p3.coords

        v1_p = p.coords - self._p1.coords
        v2_p = p.coords - self._p2.coords
        v3_p = p.coords - self._p3.coords

        cross_1 = np.cross(v1, v1_p)
        cross_2 = np.cross(v2, v2_p)
        cross_3 = np.cross(v3, v3_p)

        sign_1 = np.dot(cross_1, self._normal)
        sign_2 = np.dot(cross_2, self._normal)
        sign_3 = np.dot(cross_3, self._normal)

        positive = (sign_1 > eps) or (sign_2 > eps) or (sign_3 > eps)
        negative = (sign_1 < -eps) or (sign_2 < -eps) or (sign_3 < -eps)

        if negative and positive:
            return False

        return True


class Figure:
    """
    Class for representing figure in 3D, which consists of 
    triangles.
    """
    def __init__(
        self,
        file: str = "",
        triangles: list[Triangle] = [], 
        use_octree: bool = False
    ) -> None:
        self._triangles = None
        self.root = None

        if triangles:
            self._triangles = triangles
        elif file:
            self.read_stl(file)

        if use_octree:
            self.root = Octree(self._triangles)

    @property
    def triangles(self) -> list[Triangle]:
        return self._triangles
    
    def get_center(self) -> Point:
        """
        Get the mean of all points of figure.
        """
        all_vertices = []
        for triangle in self.triangles:
            for vertex in triangle.vertices:
                all_vertices.append(vertex.coords)

        center = np.mean(np.array(all_vertices), axis=0)
        return Point(center)
    
    def reduce_number_of_triangles(self, coeff: float) -> None:
        """
        Reduce number of triangles in Figure.

        Args:
            coeff: reduce coefficient.
        """
        verctices = []
        faces = []
        vertex_index = {}

        for triangle in self.triangles:
            face_vertices = []
            for vertex in triangle.vertices:
                coords_tuple = tuple(vertex.coords)

                if coords_tuple not in vertex_index:
                    vertex_index[coords_tuple] = len(verctices)
                    verctices.append(vertex.coords)

                face_vertices.append(vertex_index[coords_tuple])
            
            faces.append(face_vertices)
        
        mesh = trimesh.Trimesh(verctices, faces)

        reduced_vertices, reduced_faces = fast_simplification.simplify(
            mesh.vertices,
            mesh.faces,
            coeff
        )

        reduced_mesh = trimesh.Trimesh(reduced_vertices, reduced_faces)

        reduced_triangles = []

        for face in reduced_mesh.faces:
            p1 = Point(reduced_mesh.vertices[face[0]])
            p2 = Point(reduced_mesh.vertices[face[1]])
            p3 = Point(reduced_mesh.vertices[face[2]])

            triangle = Triangle(p1, p2, p3)
            reduced_triangles.append(triangle)
    
        self._triangles = reduced_triangles

        
    def read_stl(self, stl_file: str) -> None:
        """
        Read stl to the figure.

        Args:
            stl_file: file name of stl.
        """
        vertices, index = stl_reader.read(stl_file)
        triangles = []

        counter = 0

        for ind in index:
            ind1, ind2, ind3 = ind

            v1 = Point(vertices[ind1])
            v2 = Point(vertices[ind2])
            v3 = Point(vertices[ind3])

            triangle = Triangle(v1, v2, v3)

            counter += 1

            triangles.append(triangle)

        print(f"triangles count = {counter}")

        self._triangles = triangles

def distance_to(p1: Point, p2: Point) -> float:
    """
    Supporting function for calculation 
    Euclidean distance between two points.
    """
    return np.sqrt(np.sum((p1.coords - p2.coords)**2)) 