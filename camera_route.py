import math
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


def _is_zero_vector(values: List[float], epsilon: float = 1e-9) -> bool:
    return all(abs(float(value)) <= epsilon for value in values)


def _offset_vector(base: List[float], delta: List[float]) -> List[float]:
    return [
        float(base[index] + delta[index])
        for index in range(3)
    ]


def build_relative_direction_route_frames(
    start_position: List[float],
    end_position: List[float],
    start_target: List[float],
    delta_step: float,
) -> List[CameraRouteFrame]:
    direction = [
        float(start_target[index] - start_position[index])
        for index in range(3)
    ]
    if _is_zero_vector(direction):
        raise ValueError("Направление камеры не может быть нулевым.")

    safe_step = max(float(delta_step), 0.001)
    distance = _distance(start_position, end_position)
    if distance <= 1e-9:
        return [
            CameraRouteFrame(
                position=[float(value) for value in start_position],
                target=[float(value) for value in start_target],
            )
        ]

    step_count = max(1, int((distance / safe_step) + 0.999999))
    frames: List[CameraRouteFrame] = []
    for step_index in range(step_count + 1):
        t = step_index / step_count
        position = _lerp_vector(start_position, end_position, t)
        frames.append(
            CameraRouteFrame(
                position=position,
                target=_offset_vector(position, direction),
            )
        )

    return frames


def build_orbit_route_frames(
    start_position: List[float],
    target: List[float],
    angle_step_degrees: float,
) -> List[CameraRouteFrame]:
    horizontal_offset = [
        float(start_position[0] - target[0]),
        0.0,
        float(start_position[2] - target[2]),
    ]
    radius = math.hypot(horizontal_offset[0], horizontal_offset[2])
    if radius <= 1e-9:
        raise ValueError("Для облета стартовая позиция должна отличаться от точки цели по X или Z.")

    safe_angle_step = max(abs(float(angle_step_degrees)), 0.1)
    start_angle = math.atan2(horizontal_offset[2], horizontal_offset[0])
    fixed_height = float(start_position[1])

    frames: List[CameraRouteFrame] = []
    angle_degrees = 0.0
    while angle_degrees < 360.0 - 1e-9:
        angle = start_angle + math.radians(angle_degrees)
        position = [
            float(target[0] + radius * math.cos(angle)),
            fixed_height,
            float(target[2] + radius * math.sin(angle)),
        ]
        frames.append(
            CameraRouteFrame(
                position=position,
                target=[float(value) for value in target],
            )
        )
        angle_degrees += safe_angle_step

    if not frames:
        frames.append(
            CameraRouteFrame(
                position=[float(value) for value in start_position],
                target=[float(value) for value in target],
            )
        )

    return frames


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
