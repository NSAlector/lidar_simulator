import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Plane:
    point: np.ndarray
    normal: np.ndarray
    color: np.ndarray

    def intersect(
        self,
        guiding_vector: np.ndarray[np.float64],
        start_vector: np.ndarray[np.float64],
    ) -> Optional[float]:
        a = np.sum(guiding_vector * self.normal)
        b = np.sum(self.normal * (start_vector - self.point))

        if abs(a) < 10e-6:
            return None
        else:
            return -b / a