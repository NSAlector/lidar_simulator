import numpy as np
from scene_loader import SceneConfig

class SceneState:
    """State of the 3D scene that is decoupled from rendering operations."""
    def __init__(self):
        # Config properties
        self.scene_config: SceneConfig = None

        # Airplane properties
        self.airplane_pos = [0.0, 0.0, 0.0]
        self.airplane_rot = [0.0, 0.0, 0.0]  # (Yaw, Pitch, Roll)
        
        # ToF Camera properties
        self.tof_pos = [0.0, 0.0, 0.0]
        self.tof_dir = [0.0, 0.0, 0.0]
        
        # ToF results caching
        self.tof_distances = None
        self.tof_resolution = (100, 100)
        self.tof_points = np.array([])
