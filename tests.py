import numpy as np

from tof_modeling import ToFCamera
from geometry_tof import Triangle, Point, Sphere, Figure

from time import perf_counter

from scene_loader import SceneConfig, load_scene


def simple_pyramid(camera: ToFCamera) -> None:
    triangle_1 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([2, 2, 3])),
        Point(np.array([2, -2, 3]))
    )

    triangle_2 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-2, -2, 3])),
        Point(np.array([-2, 2, 3]))
    )

    triangle_3 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([2, 2, 3])),
        Point(np.array([-2, 2, 3]))
    )

    triangle_4 = Triangle(
        Point(np.array([0, 0, 2])),
        Point(np.array([-2, -2, 3])),
        Point(np.array([2, -2, 3]))
    )

    figure = Figure([triangle_1, triangle_2, triangle_3, triangle_4])

    camera.get_points_and_distances_to_object(figure)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def simple_sphere(camera: ToFCamera, sphere: Sphere) -> None:
    camera.get_points_and_distances_to_object(sphere)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def simple_sphere(camera: ToFCamera, triangle: Triangle) -> None:
    camera.get_points_and_distances_to_object(triangle)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()

def difficult_figure(camera: ToFCamera) -> None:
    triangle_1 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, 0, 1]))
    )

    triangle_2 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([1, -1, 1])),
        Point(np.array([3, 0, 3]))
    )

    triangle_3 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([-1, 0, 1])),
        Point(np.array([-1, 2, 3]))
    )

    triangle_4 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, 0, 1])),
        Point(np.array([-1, -2, 3]))
    )
    
    triangle_5 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([3, 0, 3])),
        Point(np.array([3, 3, 4]))
    )

    triangle_6 = Triangle(
        Point(np.array([1, 1, 1])),
        Point(np.array([-1, 2, 3])),
        Point(np.array([3, 3, 4]))
    )

    triangle_7 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([3, 0, 3])),
        Point(np.array([3, -3, 4]))
    )

    triangle_8 = Triangle(
        Point(np.array([1, -1, 1])),
        Point(np.array([-1, -2, 3])),
        Point(np.array([3, -3, 4]))
    )

    triangles = [triangle_1, triangle_2, triangle_3, triangle_4, triangle_5, triangle_6, triangle_7, triangle_8]

    figure = Figure(triangles)

    camera.get_points_and_distances_to_object(figure)
    camera.visualize_depth_map()
    camera.visualize_point_cloud()


if __name__ == "__main__":
    start = perf_counter()

    figure = Figure(file="stl_models/Mig29.stl", use_octree=True)
    figure_center = figure.get_center()

    tof_camera = ToFCamera(
        position=Point(np.array([-50.0, 250.0, 300.0], dtype=np.float64) * 1.2),
        width=150,
        height=150,
        direction=figure_center.coords - np.array([-50.0, 250.0, 300.0], dtype=np.float64) * 1.2,
        fov=60
    )

    sphere = Sphere(15, Point(np.array([20 -50 + 49.98*0.01,-10 + 250 - 123.77*0.01, 300 - 205.02*0.01])))
    scene = [figure, sphere]

    tof_camera.get_points_and_distances_to_scene(scene, use_octree=True)

    end = perf_counter()
    print(f"total time = {end - start}")

    tof_camera.visualize_depth_map()

    """
    Model: Mig29.stl
    Triangles amount: 4348

    Times (in seconds) :
        Sequential program: 2892.355886799982 (~48 min)
        Parallel program: 19.3167747000698
        Sequential octree program: 6.491295399959199

    """
