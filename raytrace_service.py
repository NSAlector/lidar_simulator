import os
import numpy as np
from PIL.Image import Image

from scene_state import SceneState
from camera_controller import CameraController
from geometry_utils import build_rotation_matrix, transform_vertex, compute_mesh_normalization

class RaytraceService:
    def calculate_raytrace(self, scene_state: SceneState, camera_controller: CameraController, width: int = 400, height: int = 300) -> Image:
        """
        Runs a raytrace render and returns the resulting PIL Image.
        """
        try:
            from stl_parser import parse_binary_stl
            from core.camera import Camera as RayCamera
            from core.light import Light
            from core.plane import Plane
            from core.material import Material
            from renderer.mesh_renderer import MeshRenderer
            from geometry.mesh import Mesh, Triangle as RayTriangle
            from PIL import Image as PilImage

            all_triangles = []
            
            # Cache for parsed STLs to avoid parsing the same file multiple times (e.g. Mig29)
            parsed_stls = {}

            if getattr(scene_state, 'scene_config', None):
                for obj in scene_state.scene_config.objects:
                    color = np.array([int(c*255) for c in obj.color], dtype=np.float64)
                    mat = Material(color=color, diffuse=0.85)

                    pos_arr = np.array(scene_state.airplane_pos if obj.dynamic_pos == 'airplane_pos' else obj.position, dtype=np.float64)
                    rot_arr = scene_state.airplane_rot if obj.dynamic_rot == 'airplane_rot' else obj.rotation
                    rot_mat = build_rotation_matrix(rot_arr[0], rot_arr[1], rot_arr[2])
                    scale_arr = np.array(obj.scale)

                    if obj.type == 'mesh':
                        if obj.model_path not in parsed_stls:
                            print(f"[Raytracer] Parsing {obj.model_path}...")
                            # we need an abspath probably or assume it's relative to project root
                            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), obj.model_path)
                            parsed_stls[obj.model_path] = parse_binary_stl(full_path, color)
                        
                        raw_mesh = parsed_stls[obj.model_path]
                        all_verts = np.array([[t.v0, t.v1, t.v2] for t in raw_mesh.triangles]).reshape(-1, 3)
                        # We need center and scale of the original mesh to normalize it
                        center, norm_scale = compute_mesh_normalization(all_verts)
                        
                        m = Material(color=color, diffuse=0.9, specular=0.3, shininess=50.0, reflection=0.1)
                        print(f"[Raytracer] Adding mesh: {obj.id}")
                        for tri in raw_mesh.triangles:
                            # transform_vertex applies: (v - center)/norm_scale, then rot, scale, trans
                            # Wait, the existing transform_vertex does: rot @ ((v - center)/norm_scale) + pos
                            # Let's adjust scale.
                            rot_x_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                            def scaled_transform(v):
                                v_norm = (v - center) * norm_scale
                                v_norm = rot_x_90 @ v_norm
                                v_scaled = v_norm * scale_arr
                                return rot_mat @ v_scaled + pos_arr
                                
                            v0 = scaled_transform(tri.v0)
                            v1 = scaled_transform(tri.v1)
                            v2 = scaled_transform(tri.v2)
                            
                            n = rot_mat @ tri.normal
                            nl = np.linalg.norm(n)
                            if nl < 1e-6: continue
                            all_triangles.append(RayTriangle(v0, v1, v2, n / nl, color, m))
                    
                    elif obj.type in ['plane', 'box']:
                        print(f"[Raytracer] Adding {obj.type}: {obj.id}")
                        tris = obj.get_triangles()
                        for (v0, v1, v2), normal in tris:
                            # Apply dynamic rotation/position if needed (usually primitives don't have dynamic, but just in case)
                            if obj.dynamic_pos or obj.dynamic_rot:
                                v0 = rot_mat @ v0 + pos_arr
                                v1 = rot_mat @ v1 + pos_arr
                                v2 = rot_mat @ v2 + pos_arr
                                n = rot_mat @ normal
                            else:
                                n = normal
                                
                            nl = np.linalg.norm(n)
                            if nl > 1e-6:
                                all_triangles.append(RayTriangle(v0, v1, v2, n/nl, color, mat))

            if not all_triangles:
                print("[Raytracer] Нет треугольников.")
                return None

            print(f"[Raytracer] Строим BVH для {len(all_triangles)} треугольников...")
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

            print(f"[Raytracer] Рендеринг {width}x{height}...")
            image_array = renderer.render(width, height)

            img = PilImage.fromarray(image_array.astype(np.uint8))
            return img

        except Exception as e:
            import traceback
            print(f"[Raytracer] Ошибка: {e}")
            traceback.print_exc()
            return None
