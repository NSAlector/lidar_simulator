from dataclasses import dataclass
from typing import List

from scene_loader import CameraRouteTemplate


@dataclass
class CameraRouteFrame:
    position: List[float]
    target: List[float]


def _lerp_vector(start: List[float], end: List[float], t: float) -> List[float]:
    return [
        float(start[index] + (end[index] - start[index]) * t)
        for index in range(3)
    ]


def _distance(first: List[float], second: List[float]) -> float:
    total = 0.0
    for index in range(3):
        delta = first[index] - second[index]
        total += delta * delta
    return total ** 0.5


def build_route_frames(route: CameraRouteTemplate, delta_step: float) -> List[CameraRouteFrame]:
    if route is None or not route.keyframes:
        return []

    if len(route.keyframes) == 1:
        keyframe = route.keyframes[0]
        return [
            CameraRouteFrame(
                position=keyframe.position.copy(),
                target=keyframe.target.copy(),
            )
        ]

    safe_step = max(float(delta_step), 0.001)
    start = route.keyframes[0]
    end = route.keyframes[-1]
    distance = max(
        _distance(start.position, end.position),
        _distance(start.target, end.target),
    )
    step_count = max(1, int((distance / safe_step) + 0.999999))
    frames: List[CameraRouteFrame] = []

    for step_index in range(step_count + 1):
        t = step_index / step_count
        frames.append(
            CameraRouteFrame(
                position=_lerp_vector(start.position, end.position, t),
                target=_lerp_vector(start.target, end.target, t),
            )
        )

    return frames
