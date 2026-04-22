import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout,
    QWidget, QStackedWidget, QDoubleSpinBox, QLabel,
    QHBoxLayout, QGroupBox, QSpacerItem, QSizePolicy, QDialog, QScrollArea,
    QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from gl_widget import SceneGLWidget
from simulation_controller import SimulationController, SimulationDefaults

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LiDAR Simulator")
        self.resize(800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)

        self.menu_page = QWidget()
        self.menu_layout = QVBoxLayout(self.menu_page)
        self.start_button = QPushButton("Открыть симуляцию")
        self.start_button.setFixedSize(250, 50)
        self.start_button.clicked.connect(self.show_simulation)
        self.menu_layout.addWidget(self.start_button)

        self.sim_page = QWidget()
        self.sim_layout = QHBoxLayout(self.sim_page)

        self.control_panel_widget = QWidget()
        self.control_panel_widget.setFixedWidth(360)
        self.control_layout = QVBoxLayout(self.control_panel_widget)
        self.control_layout.setContentsMargins(0, 0, 0, 0)

        self.back_button = QPushButton("Назад")
        self.back_button.clicked.connect(self.show_menu)
        self.control_layout.addWidget(self.back_button)

        self.scene_layout = QHBoxLayout()
        self.scene_combo = QComboBox()
        self.scene_layout.addWidget(self.scene_combo)
        self.control_layout.addLayout(self.scene_layout)

        self.load_camera_button = QPushButton("Загрузить конфиг рендер-камеры")
        self.control_layout.addWidget(self.load_camera_button)

        self.tof_button = QPushButton("📷 ToF-снимок")
        self.tof_button.setStyleSheet("background-color: #1565c0; color: white; font-weight: bold; margin-top: 10px;")
        self.control_layout.addWidget(self.tof_button)

        self.raytrace_button = QPushButton("🎨 Рендер")
        self.raytrace_button.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; margin-top: 5px;")
        self.control_layout.addWidget(self.raytrace_button)

        self.objects_group = QGroupBox("Управление самолётом")
        self.objects_layout = QVBoxLayout(self.objects_group)

        self.airplane_layout = QHBoxLayout()
        self.airplane_layout.addWidget(QLabel("Позиция:"))
        self.airplane_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            spin.setValue(SimulationDefaults.AIRPLANE_POS[i])
            self.airplane_spins.append(spin)
            self.airplane_layout.addWidget(spin)
        self.objects_layout.addLayout(self.airplane_layout)

        self.airplane_rot_layout = QHBoxLayout()
        self.airplane_rot_layout.addWidget(QLabel("Вращение:"))
        self.airplane_rot_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-360.0, 360.0)
            spin.setSingleStep(5.0)
            spin.setValue(SimulationDefaults.AIRPLANE_ROT[i])
            self.airplane_rot_spins.append(spin)
            self.airplane_rot_layout.addWidget(spin)
        self.objects_layout.addLayout(self.airplane_rot_layout)

        self.control_layout.addWidget(self.objects_group)
        
        self.tof_ctrl_group = QGroupBox("Управление ToF камерой")
        self.tof_ctrl_layout = QVBoxLayout(self.tof_ctrl_group)
        
        self.tof_pos_layout = QHBoxLayout()
        self.tof_pos_layout.addWidget(QLabel("Позиция:"))
        self.tof_pos_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            spin.setValue(SimulationDefaults.TOF_POS[i])
            self.tof_pos_spins.append(spin)
            self.tof_pos_layout.addWidget(spin)
        self.tof_ctrl_layout.addLayout(self.tof_pos_layout)

        self.tof_target_layout = QHBoxLayout()
        self.tof_target_layout.addWidget(QLabel("Цель:"))
        self.tof_target_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            spin.setValue(SimulationDefaults.TOF_TARGET[i])
            self.tof_target_spins.append(spin)
            self.tof_target_layout.addWidget(spin)
        self.tof_ctrl_layout.addLayout(self.tof_target_layout)

        self.control_layout.addWidget(self.tof_ctrl_group)

        self.render_ctrl_group = QGroupBox("Управление рендер-камерой")
        self.render_ctrl_layout = QVBoxLayout(self.render_ctrl_group)

        self.render_pos_layout = QHBoxLayout()
        self.render_pos_layout.addWidget(QLabel("Позиция:"))
        self.render_pos_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            spin.setValue(SimulationDefaults.RENDER_POS[i])
            self.render_pos_spins.append(spin)
            self.render_pos_layout.addWidget(spin)
        self.render_ctrl_layout.addLayout(self.render_pos_layout)

        self.render_target_layout = QHBoxLayout()
        self.render_target_layout.addWidget(QLabel("Цель:"))
        self.render_target_spins = []
        for i in range(3):
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.5)
            spin.setValue(SimulationDefaults.RENDER_TARGET[i])
            self.render_target_spins.append(spin)
            self.render_target_layout.addWidget(spin)
        self.render_ctrl_layout.addLayout(self.render_target_layout)

        self.control_layout.addWidget(self.render_ctrl_group)

        self.route_group = QGroupBox("Пролет камеры")
        self.route_layout = QVBoxLayout(self.route_group)

        self.route_mode_layout = QHBoxLayout()
        self.route_mode_layout.addWidget(QLabel("Тип:"))
        self.route_mode_combo = QComboBox()
        self.route_mode_combo.addItem("Линейный пролет", "linear")
        self.route_mode_combo.addItem("Облет вокруг точки", "orbit")
        self.route_mode_layout.addWidget(self.route_mode_combo)
        self.route_layout.addLayout(self.route_mode_layout)

        self.route_mode_stack = QStackedWidget()

        self.linear_route_page = QWidget()
        self.linear_route_layout = QVBoxLayout(self.linear_route_page)
        self.linear_route_layout.setContentsMargins(0, 0, 0, 0)

        self.linear_start_layout, self.linear_start_spins = self._build_vector_spin_row(
            "Старт:", SimulationDefaults.TOF_POS
        )
        self.linear_route_layout.addLayout(self.linear_start_layout)

        self.linear_end_layout, self.linear_end_spins = self._build_vector_spin_row(
            "Финиш:", SimulationDefaults.RENDER_POS
        )
        self.linear_route_layout.addLayout(self.linear_end_layout)

        self.linear_target_layout, self.linear_target_spins = self._build_vector_spin_row(
            "Точка взгляда:", SimulationDefaults.TOF_TARGET
        )
        self.linear_route_layout.addLayout(self.linear_target_layout)

        self.linear_step_layout = QHBoxLayout()
        self.linear_step_layout.addWidget(QLabel("Шаг:"))
        self.linear_step_spin = QDoubleSpinBox()
        self.linear_step_spin.setRange(0.1, 50.0)
        self.linear_step_spin.setDecimals(2)
        self.linear_step_spin.setSingleStep(0.5)
        self.linear_step_spin.setValue(2.0)
        self.linear_step_layout.addWidget(self.linear_step_spin)
        self.linear_route_layout.addLayout(self.linear_step_layout)

        self.route_mode_stack.addWidget(self.linear_route_page)

        self.orbit_route_page = QWidget()
        self.orbit_route_layout = QVBoxLayout(self.orbit_route_page)
        self.orbit_route_layout.setContentsMargins(0, 0, 0, 0)

        self.orbit_start_layout, self.orbit_start_spins = self._build_vector_spin_row(
            "Старт:", SimulationDefaults.TOF_POS
        )
        self.orbit_route_layout.addLayout(self.orbit_start_layout)

        self.orbit_target_layout, self.orbit_target_spins = self._build_vector_spin_row(
            "Точка цели:", SimulationDefaults.TOF_TARGET
        )
        self.orbit_route_layout.addLayout(self.orbit_target_layout)

        self.orbit_step_layout = QHBoxLayout()
        self.orbit_step_layout.addWidget(QLabel("Шаг угла:"))
        self.orbit_angle_step_spin = QDoubleSpinBox()
        self.orbit_angle_step_spin.setRange(0.1, 180.0)
        self.orbit_angle_step_spin.setDecimals(2)
        self.orbit_angle_step_spin.setSingleStep(5.0)
        self.orbit_angle_step_spin.setValue(15.0)
        self.orbit_step_layout.addWidget(self.orbit_angle_step_spin)
        self.orbit_route_layout.addLayout(self.orbit_step_layout)

        self.route_mode_stack.addWidget(self.orbit_route_page)
        self.route_layout.addWidget(self.route_mode_stack)
        self.route_mode_combo.currentIndexChanged.connect(self.route_mode_stack.setCurrentIndex)

        self.route_output_layout = QHBoxLayout()
        self.route_output_layout.addWidget(QLabel("Каталог:"))
        self.route_output_edit = QLineEdit()
        self.route_output_edit.setPlaceholderText("Выберите каталог для сохранения")
        self.route_output_layout.addWidget(self.route_output_edit)
        self.route_output_browse_button = QPushButton("...")
        self.route_output_browse_button.setFixedWidth(36)
        self.route_output_layout.addWidget(self.route_output_browse_button)
        self.route_layout.addLayout(self.route_output_layout)

        self.route_point_cloud_format_layout = QHBoxLayout()
        self.route_point_cloud_format_layout.addWidget(QLabel("Формат облака точек:"))
        self.route_point_cloud_format_combo = QComboBox()
        self.route_point_cloud_format_combo.addItem("PCD", "pcd")
        self.route_point_cloud_format_combo.addItem("LAS", "las")
        self.route_point_cloud_format_layout.addWidget(self.route_point_cloud_format_combo)
        self.route_layout.addLayout(self.route_point_cloud_format_layout)

        self.route_export_button = QPushButton("Пролет камер")
        self.route_export_button.setStyleSheet("background-color: #ef6c00; color: white; font-weight: bold; margin-top: 5px;")
        self.route_export_button.setEnabled(False)
        self.route_layout.addWidget(self.route_export_button)

        self.control_layout.addWidget(self.route_group)
        self.control_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.sim_layout.addWidget(self.control_panel_widget)

        self.gl_scene = SceneGLWidget()
        self.sim_layout.addWidget(self.gl_scene, stretch=1)

        self.stacked_widget.addWidget(self.menu_page)
        self.stacked_widget.addWidget(self.sim_page)

        self.controller = SimulationController(self)

    def show_simulation(self):
        self.stacked_widget.setCurrentWidget(self.sim_page)

    def show_menu(self):
        self.stacked_widget.setCurrentWidget(self.menu_page)

    def _build_vector_spin_row(self, label_text, defaults):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        spins = []
        for value in defaults:
            spin = QDoubleSpinBox()
            spin.setRange(-100.0, 100.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.5)
            spin.setValue(float(value))
            spins.append(spin)
            layout.addWidget(spin)
        return layout, spins

    def _show_heatmap_dialog(self, image_path: str):
        if not os.path.exists(image_path):
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("🌡 Тепловая карта ToF камеры")
        dialog.resize(700, 560)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_label = QLabel("Карта глубин ToF камеры")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #e0e0e0;"
            "background: #1a237e; padding: 6px; border-radius: 4px;"
        )
        layout.addWidget(title_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(660, 480, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(scaled)
        else:
            img_label.setText("Не удалось загрузить изображение")

        scroll.setWidget(img_label)
        layout.addWidget(scroll)

        path_label = QLabel(f"Сохранено: {image_path}")
        path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path_label.setStyleSheet("font-size: 10px; color: #888888;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        close_btn = QPushButton("Закрыть")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet(
            "background-color: #1565c0; color: white; font-weight: bold;"
            "padding: 6px 20px; border-radius: 4px;"
        )
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        dialog.setStyleSheet("background-color: #212121;")
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
