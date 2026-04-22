import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List


def _parse_vector(values, default: List[float], cast_type=float):
    if not isinstance(values, list) or len(values) != len(default):
        return default.copy()
    return [cast_type(value) for value in values]

@dataclass
class SceneObject:
    id: str
    type: str  # 'mesh', 'plane', 'box', 'sphere'
    position: List[float]
    rotation: List[float]
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    texture: str = ""
    
    # Specifics
    model_path: str = ""
    dimensions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    
    # Dynamic binding keys
    dynamic_pos: str = ""
    dynamic_rot: str = ""

    @staticmethod
    def _build_triangle(v0, v1, v2):
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            return None
        return ((v0, v1, v2), normal / norm)

    def get_triangles(self) -> List:
        """
        Генерирует список треугольников для примитивов (box, plane).
        Возвращает список кортежей: [((v1, v2, v3), normal), ...]
        Каждая вершина - np.array формы (3,)
        """
        tris = []
        if self.type == 'plane':
            sx, sy, sz = self.scale
            px, py, pz = self.position
            
            # Plane is defined in XZ plane at y = py
            v0 = np.array([px - sx, py, pz - sz])
            v1 = np.array([px - sx, py, pz + sz])
            v2 = np.array([px + sx, py, pz + sz])
            v3 = np.array([px + sx, py, pz - sz])
            
            normal = np.array([0.0, 1.0, 0.0])
            tris.append(((v0, v1, v2), normal))
            tris.append(((v0, v2, v3), normal))
            
        elif self.type == 'box':
            w, h, d = self.dimensions
            px, py, pz = self.position
            
            x0, x1 = px - w / 2, px + w / 2
            y0, y1 = py, py + h
            z0, z1 = pz - d / 2, pz + d / 2
            
            # Вспомогательная функция для добавления квада 
            def add_quad(pa, pb, pc, pd, n):
                tris.append(((pa, pb, pc), n))
                tris.append(((pa, pc, pd), n))
            
            # Top (0, 1, 0)
            add_quad(
                np.array([x0, y1, z0]), np.array([x0, y1, z1]),
                np.array([x1, y1, z1]), np.array([x1, y1, z0]),
                np.array([0.0, 1.0, 0.0])
            )
            # Bottom (0, -1, 0) - ignored in GL building but we can add anyway
            # Front (0, 0, 1)
            add_quad(
                np.array([x0, y0, z1]), np.array([x1, y0, z1]),
                np.array([x1, y1, z1]), np.array([x0, y1, z1]),
                np.array([0.0, 0.0, 1.0])
            )
            # Back (0, 0, -1)
            add_quad(
                np.array([x1, y0, z0]), np.array([x0, y0, z0]),
                np.array([x0, y1, z0]), np.array([x1, y1, z0]),
                np.array([0.0, 0.0, -1.0])
            )
            # Left (-1, 0, 0)
            add_quad(
                np.array([x0, y0, z0]), np.array([x0, y0, z1]),
                np.array([x0, y1, z1]), np.array([x0, y1, z0]),
                np.array([-1.0, 0.0, 0.0])
            )
            # Right (1, 0, 0)
            add_quad(
                np.array([x1, y0, z1]), np.array([x1, y0, z0]),
                np.array([x1, y1, z0]), np.array([x1, y1, z1]),
                np.array([1.0, 0.0, 0.0])
            )

        elif self.type == 'sphere':
            px, py, pz = self.position
            rx = max(abs(float(self.scale[0])), 1e-3)
            ry = max(abs(float(self.scale[1])), 1e-3)
            rz = max(abs(float(self.scale[2])), 1e-3)
            rings = 16
            segments = 24

            for ring in range(rings):
                theta0 = (-np.pi / 2.0) + (np.pi * ring / rings)
                theta1 = (-np.pi / 2.0) + (np.pi * (ring + 1) / rings)

                for segment in range(segments):
                    phi0 = 2.0 * np.pi * segment / segments
                    phi1 = 2.0 * np.pi * (segment + 1) / segments

                    v00 = np.array([
                        px + rx * np.cos(theta0) * np.cos(phi0),
                        py + ry * np.sin(theta0),
                        pz + rz * np.cos(theta0) * np.sin(phi0),
                    ])
                    v01 = np.array([
                        px + rx * np.cos(theta0) * np.cos(phi1),
                        py + ry * np.sin(theta0),
                        pz + rz * np.cos(theta0) * np.sin(phi1),
                    ])
                    v10 = np.array([
                        px + rx * np.cos(theta1) * np.cos(phi0),
                        py + ry * np.sin(theta1),
                        pz + rz * np.cos(theta1) * np.sin(phi0),
                    ])
                    v11 = np.array([
                        px + rx * np.cos(theta1) * np.cos(phi1),
                        py + ry * np.sin(theta1),
                        pz + rz * np.cos(theta1) * np.sin(phi1),
                    ])

                    tri_a = self._build_triangle(v00, v10, v11)
                    tri_b = self._build_triangle(v00, v11, v01)
                    if tri_a is not None:
                        tris.append(tri_a)
                    if tri_b is not None:
                        tris.append(tri_b)
            
        return tris


@dataclass
class ToFCameraConfig:
    position: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0])
    target: List[float] = field(default_factory=lambda: [22.0, 0.0, 0.0])
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    resolution: List[int] = field(default_factory=lambda: [100, 100])
    accuracy: float = 0.0

@dataclass
class RenderCameraConfig:
    position: List[float] = field(default_factory=lambda: [100.0, 18.0, 10.0])
    target: List[float] = field(default_factory=lambda: [40.0, 8.0, 0.0])
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    resolution: List[int] = field(default_factory=lambda: [400, 300])


@dataclass
class CameraRouteKeyframe:
    position: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0])
    target: List[float] = field(default_factory=lambda: [22.0, 0.0, 0.0])


@dataclass
class CameraRouteTemplate:
    id: str
    name: str
    default_step: float = 2.0
    keyframes: List[CameraRouteKeyframe] = field(default_factory=list)


@dataclass
class SceneConfig:
    objects: List[SceneObject]
    textures: dict = field(default_factory=dict)
    tof_camera: ToFCameraConfig = field(default_factory=ToFCameraConfig)
    render_camera: RenderCameraConfig = field(default_factory=RenderCameraConfig)
    camera_routes: List[CameraRouteTemplate] = field(default_factory=list)

def load_scene(config_path: str) -> SceneConfig:
    if not os.path.exists(config_path):
        return SceneConfig(objects=[])
        
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    objs = []
    for o_data in data.get("objects", []):
        obj = SceneObject(
            id=o_data.get("id", ""),
            type=o_data.get("type", "mesh"),
            position=list(o_data.get("position", [0.0, 0.0, 0.0])),
            rotation=list(o_data.get("rotation", [0.0, 0.0, 0.0])),
            scale=list(o_data.get("scale", [1.0, 1.0, 1.0])),
            color=list(o_data.get("color", [0.8, 0.8, 0.8])),
            texture=o_data.get("texture", ""),
            model_path=o_data.get("model_path", ""),
            dimensions=list(o_data.get("dimensions", [1.0, 1.0, 1.0])),
            dynamic_pos=o_data.get("dynamic_pos", ""),
            dynamic_rot=o_data.get("dynamic_rot", "")
        )
        objs.append(obj)
        
    textures = data.get("textures", {})
    tof_data = data.get("tof_camera", {})
    tof_camera = ToFCameraConfig(
        position=_parse_vector(tof_data.get("position"), [0.0, 5.0, 10.0]),
        target=_parse_vector(tof_data.get("target"), [22.0, 0.0, 0.0]),
        fov=float(tof_data.get("fov", 45.0)),
        near=float(tof_data.get("near", 0.1)),
        far=float(tof_data.get("far", 100.0)),
        resolution=_parse_vector(tof_data.get("resolution"), [100, 100], int),
        accuracy=max(0.0, float(tof_data.get("accuracy", 0.0))),
    )

    render_data = data.get("render_camera", {})
    render_camera = RenderCameraConfig(
        position=_parse_vector(render_data.get("position"), [100.0, 18.0, 10.0]),
        target=_parse_vector(render_data.get("target"), [40.0, 8.0, 0.0]),
        fov=float(render_data.get("fov", 45.0)),
        near=float(render_data.get("near", 0.1)),
        far=float(render_data.get("far", 100.0)),
        resolution=_parse_vector(render_data.get("resolution"), [400, 300], int)
    )

    camera_routes = []
    for route_index, route_data in enumerate(data.get("camera_routes", []), start=1):
        keyframes = []

        start_data = route_data.get("start")
        end_data = route_data.get("end")
        if isinstance(start_data, dict) and isinstance(end_data, dict):
            source_keyframes = [start_data, end_data]
        else:
            source_keyframes = route_data.get("keyframes", [])

        for keyframe_data in source_keyframes:
            default_position = keyframe_data.get("tof_position", keyframe_data.get("render_position", tof_camera.position))
            default_target = keyframe_data.get("tof_target", keyframe_data.get("render_target", tof_camera.target))
            keyframes.append(CameraRouteKeyframe(
                position=_parse_vector(keyframe_data.get("position"), default_position),
                target=_parse_vector(keyframe_data.get("target"), default_target),
            ))

        if not keyframes:
            continue

        route_id = str(route_data.get("id", f"route_{route_index}"))
        route_name = str(route_data.get("name", route_id))
        camera_routes.append(CameraRouteTemplate(
            id=route_id,
            name=route_name,
            default_step=max(0.1, float(route_data.get("default_step", route_data.get("step", 2.0)))),
            keyframes=keyframes,
        ))

    return SceneConfig(
        objects=objs,
        textures=textures,
        tof_camera=tof_camera,
        render_camera=render_camera,
        camera_routes=camera_routes,
    )
