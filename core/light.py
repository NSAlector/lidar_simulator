import numpy as np
from dataclasses import dataclass

@dataclass
class Light:
    position: np.ndarray[np.float64]
    color: np.ndarray[np.uint8]
    intensity: float