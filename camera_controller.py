import math

class CameraController:
    """Manages the logic and state for the OpenGL 3D Camera."""
    def __init__(self):
        self.target = [40.0, 8.0, 0.0]
        self.distance = 15.0
        self.yaw = 45.0
        self.pitch = 30.0
        self.pos = [100.0, 18.0, 10.0]
        
        self.fov = 45.0
        self.near = 0.1
        self.far = 100.0
        
        self.update_from_angles()

    def update_from_angles(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        
        x = self.target[0] + self.distance * math.cos(rad_pitch) * math.sin(rad_yaw)
        y = self.target[1] + self.distance * math.sin(rad_pitch)
        z = self.target[2] + self.distance * math.cos(rad_pitch) * math.cos(rad_yaw)
        self.pos = [x, y, z]

    def update_from_camera(self):
        dx = self.pos[0] - self.target[0]
        dy = self.pos[1] - self.target[1]
        dz = self.pos[2] - self.target[2]
        
        self.distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if self.distance > 0.0001:
            self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, dy / self.distance))))
            self.yaw = math.degrees(math.atan2(dx, dz))

    def process_mouse_pan(self, dx: float, dy: float):
        rad_yaw = math.radians(self.yaw)
        right_x = math.cos(rad_yaw)
        right_z = -math.sin(rad_yaw)
        
        pan_speed = self.distance * 0.002
        
        self.target[0] -= right_x * dx * pan_speed
        self.target[2] -= right_z * dx * pan_speed
        self.target[1] += dy * pan_speed
        
        self.update_from_angles()

    def process_mouse_rotate(self, dx: float, dy: float):
        self.yaw -= dx * 0.5
        self.pitch += dy * 0.5
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.update_from_angles()

    def process_mouse_zoom(self, delta: float):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        
        dir_x = -math.cos(rad_pitch) * math.sin(rad_yaw)
        dir_y = -math.sin(rad_pitch)
        dir_z = -math.cos(rad_pitch) * math.cos(rad_yaw)
        
        move_speed = 1.5
        
        if delta > 0:
            self.target[0] += dir_x * move_speed
            self.target[1] += dir_y * move_speed
            self.target[2] += dir_z * move_speed
        else:
            self.target[0] -= dir_x * move_speed
            self.target[1] -= dir_y * move_speed
            self.target[2] -= dir_z * move_speed
            
        self.update_from_angles()

    def export_config(self) -> dict:
        return {
            'position': self.pos.copy(),
            'target': self.target.copy(),
            'fov': float(self.fov),
            'near': float(self.near),
            'far': float(self.far),
        }

    def apply_config(self, config: dict):
        if 'fov' in config:
            self.fov = float(config['fov'])
        if 'near' in config:
            self.near = float(config['near'])
        if 'far' in config:
            self.far = float(config['far'])
        if 'position' in config:
            pos = config['position']
            if isinstance(pos, (list, tuple)) and len(pos) == 3:
                self.pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        if 'target' in config:
            tgt = config['target']
            if isinstance(tgt, (list, tuple)) and len(tgt) == 3:
                self.target = [float(tgt[0]), float(tgt[1]), float(tgt[2])]

        self.update_from_camera()
        print(f"[Camera] pos={self.pos} fov={self.fov} near={self.near} far={self.far}")
