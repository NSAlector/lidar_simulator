import datetime
import os
import re

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from camera_route import build_route_frames
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

        self.gl_scene.airplane_pos = SimulationDefaults.AIRPLANE_POS.copy()
        self.gl_scene.airplane_rot = SimulationDefaults.AIRPLANE_ROT.copy()
        self.gl_scene.tof_pos = SimulationDefaults.TOF_POS.copy()
        self.gl_scene.tof_dir = [
            SimulationDefaults.TOF_TARGET[index] - SimulationDefaults.TOF_POS[index]
            for index in range(3)
        ]
        self.gl_scene.render_camera_controller.apply_config({
            'position': SimulationDefaults.RENDER_POS,
            'target': SimulationDefaults.RENDER_TARGET,
        })

        self._connect_signals()
        self._load_configs()

    @staticmethod
    def _set_spin_values(spins, values):
        for spin, value in zip(spins, values):
            signals_blocked = spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(signals_blocked)

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
    def _try_call(label: str, callback):
        try:
            return bool(callback())
        except Exception as error:
            print(f"[{label}] {error}")
            return False

    def _scene_config(self):
        return getattr(self.gl_scene.scene_state, 'scene_config', None)

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

    def _save_current_dataset_frame(self, render_path, depth_path, point_cloud_path):
        self.tof_service.calculate_tof(self.gl_scene.scene_state)
        self.gl_scene.update()

        render_saved = False
        render_image = self._render_current_frame()
        if render_image is not None:
            render_image.save(render_path)
            render_saved = True

        points = getattr(self.gl_scene.scene_state, 'tof_points', [])
        return {
            "render_saved": render_saved,
            "depth_saved": self._try_call(
                "save_depth_map",
                lambda: self.tof_service.save_depth_map(self.gl_scene.scene_state, depth_path),
            ),
            "point_cloud_saved": self._try_call(
                "save_point_cloud_pcd",
                lambda: self.tof_service.save_point_cloud_pcd(point_cloud_path, points=points),
            ),
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
        has_routes = bool(self._route_templates_by_id)
        self.view.route_export_button.setText("Пролет камер")
        self.view.route_export_button.setEnabled((not exporting) and has_routes)
        self.view.route_step_spin.setEnabled((not exporting) and has_routes)
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
        self._set_export_controls_state(False)

        if routes:
            self.update_selected_route_template()

    def update_selected_route_template(self):
        route = self._selected_route_template()
        if route is None:
            return

        signals_blocked = self.view.route_step_spin.blockSignals(True)
        self.view.route_step_spin.setValue(max(0.1, float(route.default_step)))
        self.view.route_step_spin.blockSignals(signals_blocked)

    def _build_route_session(self, route):
        base_dir = self._default_route_output_dir()
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(
            base_dir,
            f"{self._sanitize_fragment(route.id)}_{timestamp}",
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

        self.view.load_camera_button.clicked.connect(self.load_camera_config)
        self.view.tof_button.clicked.connect(self.take_tof_snapshot)
        self.view.raytrace_button.clicked.connect(self.take_raytrace_render)
        self.view.route_export_button.clicked.connect(self.run_camera_flythrough)

    def update_airplane_pos(self):
        pos = [spin.value() for spin in self.view.airplane_spins]
        rot = [spin.value() for spin in self.view.airplane_rot_spins]
        self.gl_scene.airplane_pos = pos
        self.gl_scene.airplane_rot = rot
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
            render_path = self._build_output_path("scene_renders", "scene_render")
            depth_path = self._build_output_path("heatmaps", "depth_map")
            point_cloud_path = self._build_output_path("point_clouds", "point_cloud", extension="pcd")

            result = self._save_current_dataset_frame(
                render_path,
                depth_path,
                point_cloud_path,
            )

            if result["render_saved"]:
                print(f"Рендер сцены сохранён в {render_path}")
        except Exception as error:
            import traceback

            traceback.print_exc()
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
            QMessageBox.information(
                self.view,
                "Raytracer",
                f"Рендер завершён.\nСохранён в:\n{render_path}"
            )
        else:
            QMessageBox.warning(self.view, "Raytracer", "Рендер не удался. Проверьте консоль.")

    def run_camera_flythrough(self):
        route = self._selected_route_template()
        if route is None:
            QMessageBox.warning(self.view, "Пролет камер", "Маршрут не настроен.")
            return

        frames = build_route_frames(route, self.view.route_step_spin.value())
        if not frames:
            QMessageBox.warning(self.view, "Пролет камер", "Для маршрута не удалось построить кадры.")
            return

        scene_config = self._scene_config()
        if scene_config is None or scene_config.tof_camera is None or scene_config.render_camera is None:
            QMessageBox.warning(self.view, "Пролет камер", "Сцена не загружена.")
            return

        original_tof_position = scene_config.tof_camera.position.copy()
        original_tof_target = scene_config.tof_camera.target.copy()
        original_render_config = self.gl_scene.render_camera_controller.export_config()

        session_dir, subdirs = self._build_route_session(route)
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
                point_cloud_path = os.path.join(subdirs["point_clouds"], f"frame_{index:04d}.pcd")
                self._save_current_dataset_frame(
                    render_path,
                    depth_path,
                    point_cloud_path,
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
            import traceback

            traceback.print_exc()
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
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(base_dir, 'configs', 'sensor.yaml')
        toml_path = os.path.join(base_dir, 'configs', 'sensor.toml')
        scene_path = os.path.join(base_dir, 'configs', 'scene.json')

        try:
            lidar_config = ConfigLoader.load(yaml_path)
            print(f"Loaded YAML config (LiDAR): {lidar_config}")
        except Exception as error:
            print(f"Error loading YAML config: {error}")

        try:
            camera_config = ConfigLoader.load(toml_path)
            print(f"Loaded TOML config (Camera): {camera_config}")
        except Exception as error:
            print(f"Error loading TOML config: {error}")

        try:
            self.gl_scene.scene_state.scene_config = load_scene(scene_path)
            print(f"Loaded JSON config (Scene) with {len(self.gl_scene.scene_state.scene_config.objects)} objects.")
            tof_camera = self.gl_scene.scene_state.scene_config.tof_camera
            render_camera = self.gl_scene.scene_state.scene_config.render_camera
            self.gl_scene.scene_state.tof_resolution = tuple(tof_camera.resolution)
            self._apply_tof_camera(tof_camera.position, tof_camera.target)
            self._apply_render_camera(render_camera.position, render_camera.target)
            self.gl_scene.render_camera_controller.apply_config({
                'fov': render_camera.fov,
                'near': render_camera.near,
                'far': render_camera.far,
            })
            self.refresh_route_templates()
            self.gl_scene.update()
        except Exception as error:
            import traceback

            traceback.print_exc()
            print(f"Error loading JSON scene config: {error}")
