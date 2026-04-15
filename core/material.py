from dataclasses import dataclass
import numpy as np

@dataclass
class Material:
    color: np.ndarray
    refractive_index: float = 1.0
    reflection: float = 0.0
    refraction: float = 0.0
    diffuse: float = 1.0
    specular: float = 0.0
    shininess: float = 30.0