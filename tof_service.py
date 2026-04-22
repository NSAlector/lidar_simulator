import importlib.util
import os
import sys
import numpy as np
from scene_state import SceneState
from geometry_utils import build_rotation_matrix, compute_mesh_normalization

class ToFService:
    def __init__(self):
        self._last_camera = None

    @staticmethod
    def _apply_accuracy(camera_position, points, distances, accuracy, near_plane, far_plane):
        accuracy = max(0.0, float(accuracy))
        if accuracy <= 0.0:
            return points, distances

        valid_mask = ~np.isnan(distances)
        if not np.any(valid_mask):
            return points, distances

        quantized_distances = distances.copy()
        hit_distances = quantized_distances[valid_mask]
        hit_distances = np.round(hit_distances / accuracy) * accuracy
        hit_distances = np.clip(hit_distances, near_plane, far_plane)
        quantized_distances[valid_mask] = hit_distances

        if points is None or getattr(points, "size", 0) == 0:
            return points, quantized_distances

        directions = points - camera_position
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        quantized_points = camera_position + (directions / norms) * hit_distances[:, None]
        return quantized_points, quantized_distances

    @staticmethod
    def _load_tof_dependencies():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        geometry_path = os.path.join(base_dir, "geometry.py")
        tof_modeling_path = os.path.join(base_dir, "tof_modeling.py")

        geometry_spec = importlib.util.spec_from_file_location("_tof_geometry_module", geometry_path)
        if geometry_spec is None or geometry_spec.loader is None:
            raise ImportError(f"Cannot load geometry module from {geometry_path}")
        geometry_module = importlib.util.module_from_spec(geometry_spec)

        geometry_spec.loader.exec_module(geometry_module)

        previous_geometry_module = sys.modules.get("geometry")
        sys.modules["geometry"] = geometry_module

        try:
            tof_spec = importlib.util.spec_from_file_location("_tof_modeling_module", tof_modeling_path)
            if tof_spec is None or tof_spec.loader is None:
                raise ImportError(f"Cannot load ToF module from {tof_modeling_path}")
            tof_module = importlib.util.module_from_spec(tof_spec)

            tof_spec.loader.exec_module(tof_module)
        finally:
            if previous_geometry_module is not None:
                sys.modules["geometry"] = previous_geometry_module
            else:
                sys.modules.pop("geometry", None)

        return (
            geometry_module.Point,
            geometry_module.Triangle,
            geometry_module.Figure,
            tof_module.ToFCamera,
        )

    def calculate_tof(self, scene_state: SceneState):
        if not getattr(scene_state, 'scene_config', None) or not scene_state.scene_config.objects:
            self._last_camera = None
            return
            
        try:
            ToFPoint, ToFTriangle, ToFFigure, ToFCamera = self._load_tof_dependencies()

            import stl_loader
            parsed_stls = {}
            triangles = []
            
            for obj in scene_state.scene_config.objects:
                pos_arr = np.array(scene_state.airplane_pos if obj.dynamic_pos == 'airplane_pos' else obj.position, dtype=np.float64)
                rot_arr = scene_state.airplane_rot if obj.dynamic_rot == 'airplane_rot' else obj.rotation
                rot_mat = build_rotation_matrix(rot_arr[0], rot_arr[1], rot_arr[2])
                scale_arr = np.array(obj.scale)
                
                if obj.type == 'mesh':
                    if obj.model_path not in parsed_stls:
                        # try to load using stl_loader
                        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), obj.model_path)
                        parsed_stls[obj.model_path] = stl_loader.load_stl(full_path)
                        
                    mesh_data = parsed_stls[obj.model_path]
                    if mesh_data is not None:
                        all_points = mesh_data.vectors.reshape(-1, 3)
                        center, norm_scale = compute_mesh_normalization(all_points)
                        
                        def scaled_transform(v):
                            v_norm = (v - center) * norm_scale
                            v_scaled = v_norm * scale_arr
                            return rot_mat @ v_scaled + pos_arr
                            
                        for vec in mesh_data.vectors:
                            try:
                                v1 = scaled_transform(vec[0])
                                v2 = scaled_transform(vec[1])
                                v3 = scaled_transform(vec[2])
                                
                                p1 = ToFPoint(v1)
                                p2 = ToFPoint(v2)
                                p3 = ToFPoint(v3)
                                triangles.append(ToFTriangle(p1, p2, p3))
                            except ValueError:
                                pass
                                
                elif obj.type in ['plane', 'box', 'sphere']:
                    tris = obj.get_triangles()
                    for (v0, v1, v2), normal in tris:
                        if obj.dynamic_pos or obj.dynamic_rot:
                            v0 = rot_mat @ v0 + pos_arr
                            v1 = rot_mat @ v1 + pos_arr
                            v2 = rot_mat @ v2 + pos_arr
                        
                        try:
                            p1 = ToFPoint(v0)
                            p2 = ToFPoint(v1)
                            p3 = ToFPoint(v2)
                            triangles.append(ToFTriangle(p1, p2, p3))
                        except ValueError:
                            pass
            
            if not triangles:
                return
                
            figure = ToFFigure(triangles=triangles, use_octree=True)
            
            tof_camera = scene_state.scene_config.tof_camera
            position = np.array(tof_camera.position, dtype=np.float64)
            target = np.array(tof_camera.target, dtype=np.float64)
            direction = target - position
            if np.linalg.norm(direction) < 1e-5:
                direction = np.array([0.0, 0.0, -1.0])

            width, height = [int(value) for value in tof_camera.resolution[:2]]
            near_plane = min(float(tof_camera.near), float(tof_camera.far))
            far_plane = max(float(tof_camera.near), float(tof_camera.far))
                
            cam = ToFCamera(
                position=ToFPoint(position),
                width=width,
                height=height,
                direction=direction,
                fov=float(tof_camera.fov)
            )
            
            cam.get_points_and_distances_to_object(figure, parallel=False, use_octree=True)
            self._last_camera = cam
            raw_distances = cam.object_distances.copy()
            valid_hits_mask = ~np.isnan(raw_distances)
            in_range_hits_mask = raw_distances[valid_hits_mask] >= near_plane
            in_range_hits_mask &= raw_distances[valid_hits_mask] <= far_plane

            filtered_distances = raw_distances.copy()
            filtered_distances[valid_hits_mask] = np.where(
                in_range_hits_mask,
                raw_distances[valid_hits_mask],
                np.nan
            )
            filtered_points = (
                cam.object_points[in_range_hits_mask]
                if cam.object_points is not None and cam.object_points.size
                else np.array([])
            )
            filtered_points, filtered_distances = self._apply_accuracy(
                position,
                filtered_points,
                filtered_distances,
                getattr(tof_camera, "accuracy", 0.0),
                near_plane,
                far_plane,
            )

            cam.object_distances = filtered_distances
            cam.object_points = filtered_points

            scene_state.tof_distances = filtered_distances
            scene_state.tof_resolution = (width, height)
            scene_state.tof_points = filtered_points
            
        except ImportError:
            self._last_camera = None
        except Exception:
            self._last_camera = None

    def save_depth_map(self, scene_state: SceneState, filename="depth_map.png"):
        if not hasattr(scene_state, 'tof_distances') or scene_state.tof_distances is None:
            return False

        output_dir = os.path.dirname(os.path.abspath(filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        width, height = scene_state.tof_resolution
        depth_map = scene_state.tof_distances.reshape((height, width))

        figure, axis = plt.subplots(figsize=(8, 6))

        if np.all(np.isnan(depth_map)):
            masked_depth = np.ma.masked_where(np.ones_like(depth_map, dtype=bool), depth_map)
            axis.imshow(masked_depth, cmap='plasma_r')
        else:
            axis.imshow(depth_map, cmap='plasma_r', vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))

        axis.axis("image")
        axis.grid(False)
        axis.axis("off")

        plt.savefig(filename)
        plt.close(figure)
        return True

    def save_point_cloud_pcd(self, filename="point_cloud.pcd", points=None):
        if self._last_camera is None and points is None:
            return False

        output_dir = os.path.dirname(os.path.abspath(filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if points is None:
            self._last_camera.write_point_cloud_pcd(filename)
        else:
            import pypcd4

            point_array = np.asarray(points, dtype=np.float32)
            if point_array.size == 0:
                point_array = np.empty((0, 3), dtype=np.float32)
            elif point_array.ndim == 1:
                point_array = point_array.reshape(1, -1)

            if point_array.shape[1] > 3:
                point_array = point_array[:, :3]

            point_cloud = pypcd4.PointCloud.from_xyz_points(point_array)
            point_cloud.save(filename)

        return True

    def save_point_cloud_las(self, filename="point_cloud.las", points=None):
        if self._last_camera is None and points is None:
            return False

        output_dir = os.path.dirname(os.path.abspath(filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if points is None:
            self._last_camera.write_point_cloud_las(filename)
        else:
            import laspy

            point_array = np.asarray(points, dtype=np.float64)
            if point_array.size == 0:
                point_array = np.empty((0, 3), dtype=np.float64)
            elif point_array.ndim == 1:
                point_array = point_array.reshape(1, -1)

            if point_array.shape[1] > 3:
                point_array = point_array[:, :3]

            header = laspy.LasHeader(point_format=3, version="1.2")
            header.scales = np.array([0.0001, 0.0001, 0.0001])
            las = laspy.LasData(header)
            las.x = point_array[:, 0]
            las.y = point_array[:, 1]
            las.z = point_array[:, 2]
            las.write(filename)

        return True
