import datetime
import glob
import json
import os
import re

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from camera_route import build_orbit_route_frames, build_relative_direction_route_frames
from config_loader import ConfigLoader
from raytrace_service import RaytraceService
from scene_loader import load_scene
from tof_service import ToFService


class SimulationDefaults:
    AIRPLANE_POS = [22.0, 0.0, 0.0]
    AIRPLANE_ROT = [-90.0, 0.0, 0.0]
    TOF_POS = [0.0, 5.0, 10.0]
    TOF_TARGET = [22.0, 0.0, 0.0]
    RENDER_POS = [100.0, 18.0, 10.0]
    RENDER_TARGET = [40.0, 8.0, 0.0]


class SimulationController:
    def __init__(self, view):
        self.view = view
        self.gl_scene = self.view.gl_scene
        self.tof_service = ToFService()
        self.raytrace_service = RaytraceService()
        self._route_templates_by_id = {}
        self._scene_paths = []

        self._set_airplane_state(
            SimulationDefaults.AIRPLANE_POS,
            SimulationDefaults.AIRPLANE_ROT,
            update_view=False,
        )
        self.gl_scene.tof_pos = SimulationDefaults.TOF_POS.copy()
        self.gl_scene.tof_dir = [
            SimulationDefaults.TOF_TARGET[index] - SimulationDefaults.TOF_POS[index]
            for index in range(3)
        ]
        self.gl_scene.render_camera_controller.apply_config({
            'position': SimulationDefaults.RENDER_POS,
            'target': SimulationDefaults.RENDER_TARGET,
        })
        self._set_route_output_dir(self._default_route_output_dir())

        self._connect_signals()
        self._load_configs()

    @staticmethod
    def _set_spin_values(spins, values):
        for spin, value in zip(spins, values):
            signals_blocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(signals_blocked)

    @staticmethod
    def _set_spin_value(spin, value):
        signals_blocked = spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(signals_blocked)

    @staticmethod
    def _vector_from_spins(spins):
        return [float(spin.value()) for spin in spins]

    def _set_airplane_state(self, position, rotation, update_view: bool = True):
        safe_position = [float(value) for value in position]
        safe_rotation = [float(value) for value in rotation]
        self.gl_scene.airplane_pos = safe_position
        self.gl_scene.airplane_rot = safe_rotation

        if update_view:
            self._set_spin_values(self.view.airplane_spins, safe_position)
            self._set_spin_values(self.view.airplane_rot_spins, safe_rotation)

    def _dynamic_airplane_initial_state(self):
        scene_config = self._scene_config()
        if scene_config is not None:
            for obj in scene_config.objects:
                if obj.dynamic_pos == 'airplane_pos' or obj.dynamic_rot == 'airplane_rot':
                    return list(obj.position), list(obj.rotation)

        return (
            SimulationDefaults.AIRPLANE_POS.copy(),
            SimulationDefaults.AIRPLANE_ROT.copy(),
        )

    @staticmethod
    def _configs_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

    @classmethod
    def _default_scene_path(cls) -> str:
        return os.path.join(cls._configs_dir(), "scene.json")

    @staticmethod
    def _output_root() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_images")

    @classmethod
    def _build_output_path(cls, category: str, filename_prefix: str, extension: str = "png") -> str:
        category_dir = os.path.join(cls._output_root(), category)
        os.makedirs(category_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        return os.path.join(category_dir, f"{filename_prefix}_{timestamp}.{extension}")

    @classmethod
    def _default_route_output_dir(cls) -> str:
        return os.path.join(cls._output_root(), "flythroughs")

    @staticmethod
    def _sanitize_fragment(value: str) -> str:
        sanitized = re.sub(r"[^\w.-]+", "_", value.strip())
        return sanitized.strip("_") or "route"

    @staticmethod
    def _normalize_directory(path: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser((path or "").strip()))
        if not expanded:
            expanded = SimulationController._default_route_output_dir()
        return os.path.abspath(expanded)

    @staticmethod
    def _try_call(label: str, callback):
        try:
            return bool(callback())
        except Exception:
            return False

    def _set_route_output_dir(self, path: str):
        self.view.route_output_edit.setText(self._normalize_directory(path))

    def _current_route_output_dir(self) -> str:
        normalized = self._normalize_directory(self.view.route_output_edit.text())
        if self.view.route_output_edit.text() != normalized:
            self.view.route_output_edit.setText(normalized)
        return normalized

    @staticmethod
    def _scene_sort_key(path: str):
        normalized = os.path.normcase(os.path.abspath(path))
        default_normalized = os.path.normcase(os.path.abspath(SimulationController._default_scene_path()))
        return (0 if normalized == default_normalized else 1, os.path.basename(normalized))

    @staticmethod
    def _scene_display_name(scene_path: str) -> str:
        fallback = os.path.splitext(os.path.basename(scene_path))[0].replace("_", " ").strip() or "scene"
        try:
            with open(scene_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            return fallback.title()

        name = data.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return fallback.title()

    def _discover_scene_paths(self):
        scene_pattern = os.path.join(self._configs_dir(), "scene*.json")
        return sorted(glob.glob(scene_pattern), key=self._scene_sort_key)

    def _populate_scene_selector(self):
        self._scene_paths = self._discover_scene_paths()

        signals_blocked = self.view.scene_combo.blockSignals(True)
        self.view.scene_combo.clear()
        for scene_path in self._scene_paths:
            self.view.scene_combo.addItem(self._scene_display_name(scene_path), scene_path)
        self.view.scene_combo.blockSignals(signals_blocked)

        self.view.scene_combo.setEnabled(bool(self._scene_paths))

    def _scene_has_dynamic_airplane(self) -> bool:
        scene_config = self._scene_config()
        if scene_config is None:
            return False

        return any(
            obj.dynamic_pos == "airplane_pos" or obj.dynamic_rot == "airplane_rot"
            for obj in scene_config.objects
        )

    def _load_scene_file(self, scene_path: str):
        self.gl_scene.scene_state.scene_config = load_scene(scene_path)

        airplane_position, airplane_rotation = self._dynamic_airplane_initial_state()
        tof_camera = self.gl_scene.scene_state.scene_config.tof_camera
        render_camera = self.gl_scene.scene_state.scene_config.render_camera

        self._set_airplane_state(airplane_position, airplane_rotation)
        self.view.objects_group.setEnabled(self._scene_has_dynamic_airplane())
        self.gl_scene.scene_state.tof_resolution = tuple(tof_camera.resolution)
        self._apply_tof_camera(tof_camera.position, tof_camera.target)
        self._apply_render_camera(render_camera.position, render_camera.target)
        self.gl_scene.render_camera_controller.apply_config({
            'fov': render_camera.fov,
            'near': render_camera.near,
            'far': render_camera.far,
        })
        self.refresh_route_templates()
        self.gl_scene.reload_scene_resources()
        self.gl_scene.update()

    def _scene_config(self):
        return getattr(self.gl_scene.scene_state, 'scene_config', None)

    def _route_scene_ready(self) -> bool:
        scene_config = self._scene_config()
        return bool(
            scene_config is not None
            and scene_config.tof_camera is not None
            and scene_config.render_camera is not None
        )

    def _selected_route_mode(self) -> str:
        return str(self.view.route_mode_combo.currentData() or "linear")

    def _selected_point_cloud_format(self) -> str:
        return str(self.view.route_point_cloud_format_combo.currentData() or "pcd").lower()

    def _selected_route_session_name(self) -> str:
        return "orbit_flythrough" if self._selected_route_mode() == "orbit" else "linear_flythrough"

    def _get_render_resolution(self):
        width, height = 400, 300
        scene_config = self._scene_config()
        if scene_config is not None and scene_config.render_camera is not None:
            resolution = scene_config.render_camera.resolution
            if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                width = max(1, int(resolution[0]))
                height = max(1, int(resolution[1]))
        return width, height

    def _render_current_frame(self):
        width, height = self._get_render_resolution()
        return self.raytrace_service.calculate_raytrace(
            self.gl_scene.scene_state,
            self.gl_scene.render_camera_controller,
            width=width,
            height=height,
        )

    def _save_current_tof_frame(
        self,
        depth_path,
        point_cloud_path=None,
        point_cloud_format: str = "pcd",
    ):
        self.tof_service.calculate_tof(self.gl_scene.scene_state)
        self.gl_scene.update()

        points = getattr(self.gl_scene.scene_state, 'tof_points', [])
        normalized_point_cloud_format = str(point_cloud_format or "pcd").lower()
        point_cloud_saver = {
            "pcd": self.tof_service.save_point_cloud_pcd,
            "las": self.tof_service.save_point_cloud_las,
        }.get(normalized_point_cloud_format)
        if point_cloud_path and point_cloud_saver is None:
            raise ValueError(f"Неподдерживаемый формат облака точек: {point_cloud_format}")

        return {
            "depth_saved": self._try_call(
                "save_depth_map",
                lambda: self.tof_service.save_depth_map(self.gl_scene.scene_state, depth_path),
            ),
            "point_cloud_saved": bool(point_cloud_path) and self._try_call(
                f"save_point_cloud_{normalized_point_cloud_format}",
                lambda: point_cloud_saver(point_cloud_path, points=points),
            ),
        }

    def _save_current_dataset_frame(
        self,
        render_path,
        depth_path,
        point_cloud_path=None,
        point_cloud_format: str = "pcd",
    ):
        tof_result = self._save_current_tof_frame(
            depth_path,
            point_cloud_path,
            point_cloud_format=point_cloud_format,
        )

        render_saved = False
        render_image = self._render_current_frame()
        if render_image is not None:
            render_image.save(render_path)
            render_saved = True

        return {
            "render_saved": render_saved,
            "depth_saved": tof_result["depth_saved"],
            "point_cloud_saved": tof_result["point_cloud_saved"],
        }

    def _apply_tof_camera(self, position, target, update_view: bool = True, persist: bool = True):
        safe_position = [float(value) for value in position]
        safe_target = [float(value) for value in target]
        direction = [safe_target[index] - safe_position[index] for index in range(3)]
        self.gl_scene.tof_pos = safe_position.copy()
        self.gl_scene.tof_dir = direction

        if persist:
            scene_config = self._scene_config()
            if scene_config is not None and scene_config.tof_camera is not None:
                scene_config.tof_camera.position = safe_position.copy()
                scene_config.tof_camera.target = safe_target.copy()

        if update_view:
            self._set_spin_values(self.view.tof_pos_spins, safe_position)
            self._set_spin_values(self.view.tof_target_spins, safe_target)

    def _apply_render_camera(self, position, target, update_view: bool = True, persist: bool = True):
        safe_position = [float(value) for value in position]
        safe_target = [float(value) for value in target]
        self.gl_scene.render_camera_controller.apply_config({
            'position': safe_position,
            'target': safe_target,
        })

        if persist:
            scene_config = self._scene_config()
            if scene_config is not None and scene_config.render_camera is not None:
                scene_config.render_camera.position = safe_position.copy()
                scene_config.render_camera.target = safe_target.copy()

        if update_view:
            self._set_spin_values(self.view.render_pos_spins, safe_position)
            self._set_spin_values(self.view.render_target_spins, safe_target)

    def _restore_camera_state(self, tof_position, tof_target, render_config):
        self._apply_tof_camera(tof_position, tof_target)
        self._apply_render_camera(render_config['position'], render_config['target'])
        self.gl_scene.render_camera_controller.apply_config(render_config)

        scene_config = self._scene_config()
        if scene_config is not None and scene_config.render_camera is not None:
            scene_config.render_camera.position = render_config['position'].copy()
            scene_config.render_camera.target = render_config['target'].copy()
            scene_config.render_camera.fov = float(render_config['fov'])
            scene_config.render_camera.near = float(render_config['near'])
            scene_config.render_camera.far = float(render_config['far'])

        self.gl_scene.update()

    def _set_export_controls_state(self, exporting: bool, text: str = None):
        scene_ready = self._route_scene_ready()
        self.view.route_export_button.setText("Пролет камер")
        self.view.route_export_button.setEnabled((not exporting) and scene_ready)
        self.view.route_mode_combo.setEnabled(not exporting)
        self.view.route_mode_stack.setEnabled(not exporting)
        self.view.route_output_edit.setEnabled(not exporting)
        self.view.route_output_browse_button.setEnabled(not exporting)
        self.view.route_point_cloud_format_combo.setEnabled(not exporting)
        self.view.scene_combo.setEnabled((not exporting) and bool(self._scene_paths))
        self.view.load_camera_button.setEnabled(not exporting)
        self.view.tof_button.setEnabled(not exporting)
        self.view.raytrace_button.setEnabled(not exporting)

    def _selected_route_template(self):
        if not self._route_templates_by_id:
            return None
        return next(iter(self._route_templates_by_id.values()))

    def refresh_route_templates(self):
        scene_config = self._scene_config()
        routes = list(scene_config.camera_routes) if scene_config is not None else []
        self._route_templates_by_id = {route.id: route for route in routes}
        if routes:
            self.update_selected_route_template()
        else:
            self._apply_route_defaults_from_camera()

        self._set_export_controls_state(False)

    def update_selected_route_template(self):
        route = self._selected_route_template()
        if route is None:
            self._apply_route_defaults_from_camera()
            return

        start = route.keyframes[0]
        end = route.keyframes[-1]
        self._set_spin_values(self.view.linear_start_spins, start.position)
        self._set_spin_values(self.view.linear_end_spins, end.position)
        self._set_spin_values(self.view.linear_target_spins, start.target)
        self._set_spin_value(self.view.linear_step_spin, max(0.1, float(route.default_step)))
        self._set_spin_values(self.view.orbit_start_spins, start.position)
        self._set_spin_values(self.view.orbit_target_spins, start.target)

    def _apply_route_defaults_from_camera(self):
        scene_config = self._scene_config()
        if scene_config is not None and scene_config.tof_camera is not None:
            start_position = [float(value) for value in scene_config.tof_camera.position]
            target = [float(value) for value in scene_config.tof_camera.target]
        else:
            start_position = SimulationDefaults.TOF_POS.copy()
            target = SimulationDefaults.TOF_TARGET.copy()

        end_position = start_position.copy()
        end_position[0] += 10.0

        self._set_spin_values(self.view.linear_start_spins, start_position)
        self._set_spin_values(self.view.linear_end_spins, end_position)
        self._set_spin_values(self.view.linear_target_spins, target)
        self._set_spin_values(self.view.orbit_start_spins, start_position)
        self._set_spin_values(self.view.orbit_target_spins, target)

    def _build_route_session(self, session_name: str, base_dir=None):
        base_dir = self._normalize_directory(base_dir or self._default_route_output_dir())
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(
            base_dir,
            f"{self._sanitize_fragment(session_name)}_{timestamp}",
        )
        subdirs = {
            "renders": os.path.join(session_dir, "renders"),
            "depth_maps": os.path.join(session_dir, "depth_maps"),
            "point_clouds": os.path.join(session_dir, "point_clouds"),
        }

        os.makedirs(session_dir, exist_ok=True)
        for path in subdirs.values():
            os.makedirs(path, exist_ok=True)

        return session_dir, subdirs

    def _connect_signals(self):
        for spin in self.view.airplane_spins:
            spin.valueChanged.connect(self.update_airplane_pos)
        for spin in self.view.airplane_rot_spins:
            spin.valueChanged.connect(self.update_airplane_pos)
        for spin in self.view.tof_pos_spins:
            spin.valueChanged.connect(self.update_tof_pos)
        for spin in self.view.tof_target_spins:
            spin.valueChanged.connect(self.update_tof_pos)
        for spin in self.view.render_pos_spins:
            spin.valueChanged.connect(self.update_render_camera)
        for spin in self.view.render_target_spins:
            spin.valueChanged.connect(self.update_render_camera)

        self.view.scene_combo.currentIndexChanged.connect(self.change_scene)
        self.view.load_camera_button.clicked.connect(self.load_camera_config)
        self.view.tof_button.clicked.connect(self.take_tof_snapshot)
        self.view.raytrace_button.clicked.connect(self.take_raytrace_render)
        self.view.route_output_browse_button.clicked.connect(self.choose_route_output_dir)
        self.view.route_export_button.clicked.connect(self.run_camera_flythrough)

    def _build_selected_route_frames(self):
        route_mode = self._selected_route_mode()
        if route_mode == "orbit":
            return build_orbit_route_frames(
                self._vector_from_spins(self.view.orbit_start_spins),
                self._vector_from_spins(self.view.orbit_target_spins),
                self.view.orbit_angle_step_spin.value(),
            )

        return build_relative_direction_route_frames(
            self._vector_from_spins(self.view.linear_start_spins),
            self._vector_from_spins(self.view.linear_end_spins),
            self._vector_from_spins(self.view.linear_target_spins),
            self.view.linear_step_spin.value(),
        )

    def update_airplane_pos(self):
        pos = [spin.value() for spin in self.view.airplane_spins]
        rot = [spin.value() for spin in self.view.airplane_rot_spins]
        self._set_airplane_state(pos, rot, update_view=False)
        self.gl_scene.update()

    def update_tof_pos(self):
        pos = [spin.value() for spin in self.view.tof_pos_spins]
        target = [spin.value() for spin in self.view.tof_target_spins]
        self._apply_tof_camera(pos, target, update_view=False, persist=True)
        self.gl_scene.update()

    def update_render_camera(self):
        pos = [spin.value() for spin in self.view.render_pos_spins]
        target = [spin.value() for spin in self.view.render_target_spins]
        self._apply_render_camera(pos, target, update_view=False, persist=True)
        self.gl_scene.update()

    def change_scene(self, index=None):
        del index

        scene_path = self.view.scene_combo.currentData()
        if not scene_path:
            return

        try:
            self._load_scene_file(str(scene_path))
        except Exception as error:
            QMessageBox.critical(self.view, "Ошибка загрузки", f"Не удалось загрузить сцену:\n{error}")

    def load_camera_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Выберите файл конфигурации камеры",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'),
            "Config files (*.yaml *.yml *.toml);;All files (*.*)"
        )
        if not path:
            return

        try:
            config = ConfigLoader.load(path)
            self.gl_scene.apply_camera_config(config)
            self._set_spin_values(self.view.render_pos_spins, self.gl_scene.render_camera_controller.pos)
            self._set_spin_values(self.view.render_target_spins, self.gl_scene.render_camera_controller.target)

            scene_config = self._scene_config()
            if scene_config is not None and scene_config.render_camera is not None:
                scene_config.render_camera.position = self.gl_scene.render_camera_controller.pos.copy()
                scene_config.render_camera.target = self.gl_scene.render_camera_controller.target.copy()
                scene_config.render_camera.fov = self.gl_scene.render_camera_controller.fov
                scene_config.render_camera.near = self.gl_scene.render_camera_controller.near
                scene_config.render_camera.far = self.gl_scene.render_camera_controller.far
                if 'resolution' in config and isinstance(config['resolution'], (list, tuple)) and len(config['resolution']) >= 2:
                    scene_config.render_camera.resolution = [int(config['resolution'][0]), int(config['resolution'][1])]

            name = config.get('name', os.path.basename(path))
            QMessageBox.information(
                self.view,
                "Конфиг применён",
                (
                    f"Рендер-камера «{name}» загружена.\n"
                    f"FOV: {self.gl_scene.cam_fov}°  "
                    f"Near: {self.gl_scene.cam_near}  "
                    f"Far: {self.gl_scene.cam_far}"
                ),
            )
        except Exception as error:
            QMessageBox.critical(self.view, "Ошибка загрузки", str(error))

    def take_tof_snapshot(self):
        self.view.tof_button.setText("Расчёт... Ждите")
        self.view.tof_button.setEnabled(False)
        QApplication.processEvents()

        try:
            depth_path = self._build_output_path("heatmaps", "depth_map")

            result = self._save_current_tof_frame(
                depth_path,
            )

            if result["depth_saved"]:
                self.view._show_heatmap_dialog(depth_path)
            else:
                QMessageBox.warning(
                    self.view,
                    "ToF снимок",
                    "Не удалось сохранить карту глубин.",
                )
        except Exception as error:
            QMessageBox.critical(self.view, "ToF снимок", f"Не удалось сохранить снимок:\n{error}")
        finally:
            self.view.tof_button.setText("📷 Снимок ToF камерой")
            self.view.tof_button.setEnabled(True)

    def take_raytrace_render(self):
        self.view.raytrace_button.setText("Рендеринг... Ждите")
        self.view.raytrace_button.setEnabled(False)
        QApplication.processEvents()

        render_path = self._build_output_path("raytraces", "raytrace_render")
        img = self._render_current_frame()

        self.view.raytrace_button.setText("🎨 Рендер (Raytracer)")
        self.view.raytrace_button.setEnabled(True)

        if img is not None:
            img.save(render_path)
            self.view._show_render_dialog(render_path)
            return
            QMessageBox.information(
                self.view,
                "Raytracer",
                f"Рендер завершён.\nСохранён в:\n{render_path}"
            )
        else:
            QMessageBox.warning(self.view, "Raytracer", "Рендер не удался.")

    def choose_route_output_dir(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self.view,
            "Выберите каталог для сохранения пролета",
            self._current_route_output_dir(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected_dir:
            self._set_route_output_dir(selected_dir)

    def run_camera_flythrough(self):
        if not self._route_scene_ready():
            QMessageBox.warning(self.view, "Пролет камер", "Сцена не загружена.")
            return

        try:
            frames = self._build_selected_route_frames()
        except ValueError as error:
            QMessageBox.warning(self.view, "Пролет камер", str(error))
            return

        if not frames:
            QMessageBox.warning(self.view, "Пролет камер", "Для выбранного типа не удалось построить кадры.")
            return

        scene_config = self._scene_config()
        point_cloud_format = self._selected_point_cloud_format()

        original_tof_position = scene_config.tof_camera.position.copy()
        original_tof_target = scene_config.tof_camera.target.copy()
        original_render_config = self.gl_scene.render_camera_controller.export_config()

        try:
            session_dir, subdirs = self._build_route_session(
                self._selected_route_session_name(),
                self._current_route_output_dir(),
            )
        except Exception as error:
            QMessageBox.critical(self.view, "Пролет камер", f"Не удалось подготовить каталог сохранения:\n{error}")
            return

        self._set_export_controls_state(True)
        QApplication.processEvents()

        try:
            for index, frame in enumerate(frames):
                self._apply_tof_camera(frame.position, frame.target)
                self._apply_render_camera(frame.position, frame.target)
                self.gl_scene.update()
                QApplication.processEvents()

                render_path = os.path.join(subdirs["renders"], f"frame_{index:04d}.png")
                depth_path = os.path.join(subdirs["depth_maps"], f"frame_{index:04d}.png")
                point_cloud_path = os.path.join(subdirs["point_clouds"], f"frame_{index:04d}.{point_cloud_format}")
                self._save_current_dataset_frame(
                    render_path,
                    depth_path,
                    point_cloud_path,
                    point_cloud_format=point_cloud_format,
                )

                QApplication.processEvents()

            QMessageBox.information(
                self.view,
                "Пролет камер",
                (
                    f"Экспорт завершён.\n"
                    f"Кадров: {len(frames)}\n"
                    f"Каталог:\n{session_dir}"
                ),
            )
        except Exception as error:
            QMessageBox.critical(self.view, "Пролет камер", f"Экспорт прерван:\n{error}")
        finally:
            self._restore_camera_state(
                original_tof_position,
                original_tof_target,
                original_render_config,
            )
            self._set_export_controls_state(False)
            QApplication.processEvents()

    def _load_configs(self):
        config_dir = self._configs_dir()
        yaml_path = os.path.join(config_dir, 'sensor.yaml')
        toml_path = os.path.join(config_dir, 'sensor.toml')

        try:
            lidar_config = ConfigLoader.load(yaml_path)
        except Exception:
            pass

        try:
            camera_config = ConfigLoader.load(toml_path)
        except Exception:
            pass

        self._populate_scene_selector()
        if not self._scene_paths:
            self.view.objects_group.setEnabled(False)
            self._set_export_controls_state(False)
            return

        default_scene_path = os.path.normcase(os.path.abspath(self._default_scene_path()))
        selected_index = 0
        for scene_index, scene_path in enumerate(self._scene_paths):
            if os.path.normcase(os.path.abspath(scene_path)) == default_scene_path:
                selected_index = scene_index
                break

        try:
            signals_blocked = self.view.scene_combo.blockSignals(True)
            self.view.scene_combo.setCurrentIndex(selected_index)
            self.view.scene_combo.blockSignals(signals_blocked)
            self._load_scene_file(self._scene_paths[selected_index])
        except Exception as error:
            QMessageBox.critical(self.view, "Ошибка загрузки", f"Не удалось загрузить сцену:\n{error}")
