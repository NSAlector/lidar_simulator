import numpy as np
from dataclasses import dataclass

@dataclass
class Camera:
    position: np.ndarray[np.float64]
    look_at: np.ndarray[np.float64]
    vector_up: np.ndarray[np.float64]
    fov_vertical: float
    fov_horizontal: float

    def __normalize(self, v: np.ndarray) -> np.ndarray: 
        return v/np.linalg.norm(v)

    def __post_init__(self):
        self.forward = self.__normalize(self.look_at - self.position)
        self.right = self.__normalize(np.cross(self.forward, self.vector_up))
        self.real_up = self.__normalize(np.cross(self.right, self.forward))
        self.viewport_height = 2 * np.tan(np.radians(self.fov_vertical) / 2)
        self.viewport_width = 2 * np.tan(np.radians(self.fov_horizontal) / 2)