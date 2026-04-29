import os
import numpy as np
from PIL import Image as PilImage
from PIL.Image import Image

from scene_state import SceneState
from camera_controller import CameraController
from geometry_utils import build_rotation_matrix, compute_mesh_normalization

from stl_parser import parse_binary_stl
from core.camera import Camera as RayCamera
from core.light import Light
from core.material import Material
from renderer.mesh_renderer import MeshRenderer
from geometry.mesh import Mesh, Triangle as RayTriangle

_MESH_ROTATE_X_90 = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)

class RaytraceService:
    def __init__(self):
        self._base_texture_cache: dict[str, np.ndarray] = {}
        self._tinted_texture_cache: dict[tuple[str, tuple[int, int, int]], np.ndarray] = {}

    @staticmethod
    def _project_root() -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def _load_texture_pixels(self, texture_path: str) -> np.ndarray | None:
        full_path = os.path.normpath(os.path.join(self._project_root(), texture_path))
        if full_path in self._base_texture_cache:
            return self._base_texture_cache[full_path]
        if not os.path.exists(full_path):
            return None

        with PilImage.open(full_path) as texture_image:
            pixels = np.array(texture_image.convert("RGB"), dtype=np.float64)

        self._base_texture_cache[full_path] = pixels
        return pixels

    def _get_object_texture(self, scene_state: SceneState, texture_name: str, color: np.ndarray) -> np.ndarray | None:
        scene_config = getattr(scene_state, 'scene_config', None)
        if scene_config is None or not texture_name:
            return None

        texture_path = scene_config.textures.get(texture_name)
        if not texture_path:
            return None

        base_texture = self._load_texture_pixels(texture_path)
        if base_texture is None:
            return None

        color_key = tuple(int(channel) for channel in np.clip(color, 0, 255).astype(np.uint8))
        cache_key = (os.path.normpath(texture_path), color_key)
        if cache_key not in self._tinted_texture_cache:
            tint = np.array(color_key, dtype=np.float64) / 255.0
            self._tinted_texture_cache[cache_key] = np.clip(base_texture * tint, 0, 255)
        return self._tinted_texture_cache[cache_key]

    @staticmethod
    def _attach_texture(material: Material, texture: np.ndarray | None) -> Material:
        if texture is None:
            if hasattr(material, 'texture'):
                delattr(material, 'texture')
            return material
        setattr(material, 'texture', texture)
        return material

    @staticmethod
    def _resolve_object_transform(scene_state: SceneState, obj):
        position = scene_state.airplane_pos if obj.dynamic_pos == 'airplane_pos' else obj.position
        rotation = scene_state.airplane_rot if obj.dynamic_rot == 'airplane_rot' else obj.rotation
        position_arr = np.array(position, dtype=np.float64)
        rotation_arr = np.array(rotation, dtype=np.float64)
        scale_arr = np.array(obj.scale, dtype=np.float64)
        rotation_matrix = build_rotation_matrix(rotation_arr[0], rotation_arr[1], rotation_arr[2])
        return position_arr, rotation_arr, scale_arr, rotation_matrix

    @staticmethod
    def _transform_local_vertex(vertex, scale_arr: np.ndarray, rotation_matrix: np.ndarray, position_arr: np.ndarray) -> np.ndarray:
        local_vertex = np.array(vertex, dtype=np.float64) * scale_arr
        return rotation_matrix @ local_vertex + position_arr

    @staticmethod
    def _append_triangle(
        triangle_list: list,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        color: np.ndarray,
        material: Material,
        uv0: np.ndarray | None = None,
        uv1: np.ndarray | None = None,
        uv2: np.ndarray | None = None,
    ) -> None:
        normal = np.cross(v1 - v0, v2 - v0)
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-6:
            return

        if uv0 is None or uv1 is None or uv2 is None:
            triangle_list.append(RayTriangle(v0, v1, v2, normal / normal_length, color, material))
            return

        triangle_list.append(
            RayTriangle(
                v0,
                v1,
                v2,
                normal / normal_length,
                color,
                material,
                np.array(uv0, dtype=np.float64),
                np.array(uv1, dtype=np.float64),
                np.array(uv2, dtype=np.float64),
            )
        )

    @staticmethod
    def _plane_texture_tiling(texture_name: str, scale_arr: np.ndarray) -> tuple[float, float]:
        if texture_name == 'ground':
            return scale_arr[0] / 3.0, scale_arr[2] / 3.0
        if texture_name == 'road':
            return scale_arr[0] / 4.0, 1.0
        if texture_name == 'field':
            return 1.0, scale_arr[2] / 4.0
        return 1.0, 1.0

    def _build_plane_triangles(
        self,
        triangle_list: list,
        obj,
        color: np.ndarray,
        material: Material,
        position_arr: np.ndarray,
        scale_arr: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> None:
        tiles_x, tiles_y = self._plane_texture_tiling(obj.texture, scale_arr)
        local_vertices = [
            np.array([-1.0, 0.0, -1.0], dtype=np.float64),
            np.array([-1.0, 0.0,  1.0], dtype=np.float64),
            np.array([ 1.0, 0.0,  1.0], dtype=np.float64),
            np.array([ 1.0, 0.0, -1.0], dtype=np.float64),
        ]
        transformed = [
            self._transform_local_vertex(vertex, scale_arr, rotation_matrix, position_arr)
            for vertex in local_vertices
        ]
        texture = getattr(material, 'texture', None)
        uv0 = np.array([0.0, 0.0], dtype=np.float64)
        uv1 = np.array([0.0, tiles_y], dtype=np.float64)
        uv2 = np.array([tiles_x, tiles_y], dtype=np.float64)
        uv3 = np.array([tiles_x, 0.0], dtype=np.float64)
        if texture is None:
            uv0 = uv1 = uv2 = uv3 = None

        self._append_triangle(triangle_list, transformed[0], transformed[1], transformed[2], color, material, uv0, uv1, uv2)
        self._append_triangle(triangle_list, transformed[0], transformed[2], transformed[3], color, material, uv0, uv2, uv3)

    def _build_box_triangles(
        self,
        triangle_list: list,
        obj,
        color: np.ndarray,
        wall_material: Material,
        position_arr: np.ndarray,
        scale_arr: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> None:
        width, height, depth = obj.dimensions
        x0, x1 = -width / 2.0, width / 2.0
        y0, y1 = 0.0, height
        z0, z1 = -depth / 2.0, depth / 2.0

        roof_material = Material(color=np.array([51.0, 51.0, 51.0], dtype=np.float64), diffuse=0.85)
        uv_a = np.array([0.0, 0.0], dtype=np.float64)
        uv_b = np.array([1.0, 0.0], dtype=np.float64)
        uv_c = np.array([1.0, 1.0], dtype=np.float64)
        uv_d = np.array([0.0, 1.0], dtype=np.float64)

        faces = [
            (
                wall_material,
                [
                    np.array([x0, y0, z1], dtype=np.float64),
                    np.array([x1, y0, z1], dtype=np.float64),
                    np.array([x1, y1, z1], dtype=np.float64),
                    np.array([x0, y1, z1], dtype=np.float64),
                ],
                color,
            ),
            (
                wall_material,
                [
                    np.array([x1, y0, z0], dtype=np.float64),
                    np.array([x0, y0, z0], dtype=np.float64),
                    np.array([x0, y1, z0], dtype=np.float64),
                    np.array([x1, y1, z0], dtype=np.float64),
                ],
                color,
            ),
            (
                wall_material,
                [
                    np.array([x0, y0, z0], dtype=np.float64),
                    np.array([x0, y0, z1], dtype=np.float64),
                    np.array([x0, y1, z1], dtype=np.float64),
                    np.array([x0, y1, z0], dtype=np.float64),
                ],
                color,
            ),
            (
                wall_material,
                [
                    np.array([x1, y0, z1], dtype=np.float64),
                    np.array([x1, y0, z0], dtype=np.float64),
                    np.array([x1, y1, z0], dtype=np.float64),
                    np.array([x1, y1, z1], dtype=np.float64),
                ],
                color,
            ),
            (
                roof_material,
                [
                    np.array([x0, y1, z0], dtype=np.float64),
                    np.array([x0, y1, z1], dtype=np.float64),
                    np.array([x1, y1, z1], dtype=np.float64),
                    np.array([x1, y1, z0], dtype=np.float64),
                ],
                roof_material.color,
            ),
        ]

        for material, local_vertices, face_color in faces:
            transformed = [
                self._transform_local_vertex(vertex, scale_arr, rotation_matrix, position_arr)
                for vertex in local_vertices
            ]
            texture = getattr(material, 'texture', None)

            if texture is None:
                self._append_triangle(triangle_list, transformed[0], transformed[1], transformed[2], face_color, material)
                self._append_triangle(triangle_list, transformed[0], transformed[2], transformed[3], face_color, material)
            else:
                self._append_triangle(triangle_list, transformed[0], transformed[1], transformed[2], face_color, material, uv_a, uv_b, uv_c)
                self._append_triangle(triangle_list, transformed[0], transformed[2], transformed[3], face_color, material, uv_a, uv_c, uv_d)

    def _build_sphere_triangles(
        self,
        triangle_list: list,
        obj,
        color: np.ndarray,
        material: Material,
        position_arr: np.ndarray,
        scale_arr: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> None:
        rings = 16
        segments = 24
        for ring in range(rings):
            theta0 = (-np.pi / 2.0) + (np.pi * ring / rings)
            theta1 = (-np.pi / 2.0) + (np.pi * (ring + 1) / rings)
            v0 = ring / rings
            v1 = (ring + 1) / rings

            for segment in range(segments):
                phi0 = 2.0 * np.pi * segment / segments
                phi1 = 2.0 * np.pi * (segment + 1) / segments
                u0 = segment / segments
                u1 = (segment + 1) / segments

                local_v00 = np.array([np.cos(theta0) * np.cos(phi0), np.sin(theta0), np.cos(theta0) * np.sin(phi0)], dtype=np.float64)
                local_v01 = np.array([np.cos(theta0) * np.cos(phi1), np.sin(theta0), np.cos(theta0) * np.sin(phi1)], dtype=np.float64)
                local_v10 = np.array([np.cos(theta1) * np.cos(phi0), np.sin(theta1), np.cos(theta1) * np.sin(phi0)], dtype=np.float64)
                local_v11 = np.array([np.cos(theta1) * np.cos(phi1), np.sin(theta1), np.cos(theta1) * np.sin(phi1)], dtype=np.float64)

                vertex_00 = self._transform_local_vertex(local_v00, scale_arr, rotation_matrix, position_arr)
                vertex_01 = self._transform_local_vertex(local_v01, scale_arr, rotation_matrix, position_arr)
                vertex_10 = self._transform_local_vertex(local_v10, scale_arr, rotation_matrix, position_arr)
                vertex_11 = self._transform_local_vertex(local_v11, scale_arr, rotation_matrix, position_arr)
                texture = getattr(material, 'texture', None)

                if texture is None:
                    self._append_triangle(triangle_list, vertex_00, vertex_10, vertex_11, color, material)
                    self._append_triangle(triangle_list, vertex_00, vertex_11, vertex_01, color, material)
                else:
                    uv00 = np.array([u0, v0], dtype=np.float64)
                    uv01 = np.array([u1, v0], dtype=np.float64)
                    uv10 = np.array([u0, v1], dtype=np.float64)
                    uv11 = np.array([u1, v1], dtype=np.float64)
                    self._append_triangle(triangle_list, vertex_00, vertex_10, vertex_11, color, material, uv00, uv10, uv11)
                    self._append_triangle(triangle_list, vertex_00, vertex_11, vertex_01, color, material, uv00, uv11, uv01)

    @staticmethod
    def _mesh_uv(vertex: np.ndarray, min_x: float, min_z: float, span_x: float, span_z: float) -> np.ndarray:
        rotated_vertex = _MESH_ROTATE_X_90 @ vertex
        norm_x = (rotated_vertex[0] - min_x) / span_x
        norm_z = (rotated_vertex[2] - min_z) / span_z
        return np.array([1.0 - norm_z, 1.0 - norm_x], dtype=np.float64)

    def calculate_raytrace(self, scene_state: SceneState, camera_controller: CameraController, width: int = 400, height: int = 300) -> Image:
        """
        Runs a raytrace render and returns the resulting PIL Image.
        """
        try:
            all_triangles = []
            parsed_stls = {}

            if getattr(scene_state, 'scene_config', None):
                for obj in scene_state.scene_config.objects:
                    color = np.array([int(channel * 255) for channel in obj.color], dtype=np.float64)
                    texture_pixels = self._get_object_texture(scene_state, obj.texture, color)
                    pos_arr, _, scale_arr, rot_mat = self._resolve_object_transform(scene_state, obj)

                    if obj.type == 'mesh':
                        if obj.model_path not in parsed_stls:
                            full_path = os.path.join(self._project_root(), obj.model_path)
                            raw_mesh = parse_binary_stl(full_path, color)
                            all_vertices = np.array([[tri.v0, tri.v1, tri.v2] for tri in raw_mesh.triangles], dtype=np.float64).reshape(-1, 3)
                            center, norm_scale = compute_mesh_normalization(all_vertices)
                            rotated_vertices = (_MESH_ROTATE_X_90 @ all_vertices.T).T
                            min_coords = rotated_vertices.min(axis=0)
                            max_coords = rotated_vertices.max(axis=0)
                            parsed_stls[obj.model_path] = {
                                'mesh': raw_mesh,
                                'center': center,
                                'norm_scale': norm_scale,
                                'min_x': min_coords[0],
                                'min_z': min_coords[2],
                                'span_x': max(max_coords[0] - min_coords[0], 1e-6),
                                'span_z': max(max_coords[2] - min_coords[2], 1e-6),
                            }

                        raw_mesh_data = parsed_stls[obj.model_path]
                        material = self._attach_texture(Material(
                            color=color,
                            diffuse=0.9,
                            specular=0.3,
                            shininess=50.0,
                            reflection=0.1,
                        ), texture_pixels)

                        def scaled_transform(vertex):
                            normalized_vertex = (vertex - raw_mesh_data['center']) * raw_mesh_data['norm_scale']
                            normalized_vertex = _MESH_ROTATE_X_90 @ normalized_vertex
                            scaled_vertex = normalized_vertex * scale_arr
                            return rot_mat @ scaled_vertex + pos_arr

                        for tri in raw_mesh_data['mesh'].triangles:
                            v0 = scaled_transform(tri.v0)
                            v1 = scaled_transform(tri.v1)
                            v2 = scaled_transform(tri.v2)
                            texture = getattr(material, 'texture', None)

                            if texture is None:
                                self._append_triangle(all_triangles, v0, v1, v2, color, material)
                                continue

                            uv0 = self._mesh_uv(tri.v0, raw_mesh_data['min_x'], raw_mesh_data['min_z'], raw_mesh_data['span_x'], raw_mesh_data['span_z'])
                            uv1 = self._mesh_uv(tri.v1, raw_mesh_data['min_x'], raw_mesh_data['min_z'], raw_mesh_data['span_x'], raw_mesh_data['span_z'])
                            uv2 = self._mesh_uv(tri.v2, raw_mesh_data['min_x'], raw_mesh_data['min_z'], raw_mesh_data['span_x'], raw_mesh_data['span_z'])
                            self._append_triangle(all_triangles, v0, v1, v2, color, material, uv0, uv1, uv2)

                    elif obj.type == 'plane':
                        material = self._attach_texture(Material(color=color, diffuse=0.85), texture_pixels)
                        self._build_plane_triangles(all_triangles, obj, color, material, pos_arr, scale_arr, rot_mat)

                    elif obj.type == 'box':
                        material = self._attach_texture(Material(color=color, diffuse=0.85), texture_pixels)
                        self._build_box_triangles(all_triangles, obj, color, material, pos_arr, scale_arr, rot_mat)

                    elif obj.type == 'sphere':
                        material = self._attach_texture(Material(color=color, diffuse=0.85), texture_pixels)
                        self._build_sphere_triangles(all_triangles, obj, color, material, pos_arr, scale_arr, rot_mat)

            if not all_triangles:
                return None

            scene_mesh = Mesh(all_triangles)

            cam_pos = np.array(camera_controller.pos, dtype=np.float64)
            cam_target = np.array(camera_controller.target, dtype=np.float64)

            camera = RayCamera(
                position=cam_pos,
                look_at=cam_target,
                vector_up=np.array([0.0, 1.0, 0.0]),
                fov_vertical=camera_controller.fov,
                fov_horizontal=camera_controller.fov * (width / height)
            )

            light = Light(
                position=np.array([10.0, 20.0, 10.0]),
                color=np.array([255, 255, 220], dtype=np.uint8),
                intensity=2.0
            )

            renderer = MeshRenderer(
                mesh=scene_mesh,
                camera=camera,
                plane=None,
                light=light
            )

            image_array = renderer.render(width, height)

            img = PilImage.fromarray(image_array.astype(np.uint8))
            return img

        except Exception:
            return None
