import numpy as np
import laspy

from pathlib import Path
from abc import ABC, abstractmethod
from pypcd4 import pypcd4


class PointCloudLoaderMeta(ABC):
    _registry = {}

    def __init_subclass__(cls, format_name=None, **kw):
        super().__init_subclass__(**kw)

        if format_name:
            PointCloudLoaderMeta._registry[format_name] = cls

    @abstractmethod
    def load(self, filename: str) -> np.ndarray | None:
        ...

    @abstractmethod
    def validate(self, filename: str) -> bool:
        ...

    @classmethod
    def get_plugin(cls, format_name: str):
        plugin_cls = cls._registry.get(format_name)
        if not plugin_cls:
            raise ValueError(f"Unknown format: {format_name}")
        return plugin_cls()


class PCDLoader(PointCloudLoaderMeta, format_name='pcd'):
    def validate(self, filename) -> bool:
        p = Path(filename)
        if not p.exists():
            raise FileNotFoundError(f'File is not found: {p}')
        if p.stat().st_size == 0:
            raise ValueError(f'File is empty: {p}')
        
        ext = p.suffix.lower()
        
        if ext == ".pcd":
            return True
        return False

    def load(self, filename: str) -> np.ndarray | None:
        if self.validate(filename):
            pcd = pypcd4.PointCloud.from_path(filename)
            return np.column_stack((pcd.pc_data['x'], pcd.pc_data['y'], pcd.pc_data['z']))
        return None


class LASLoader(PointCloudLoaderMeta, format_name='las'):
    def validate(self, filename) -> bool:
        p = Path(filename)
        if not p.exists():
            raise FileNotFoundError(f'File is not found: {p}')
        if p.stat().st_size == 0:
            raise ValueError(f'File is empty: {p}')
        
        ext = p.suffix.lower()
        
        if ext == ".las":
            return True
        return False

    def load(self, filename: str) -> np.ndarray | None:
        if self.validate(filename):
            las = laspy.read(filename)
            return np.column_stack((las.x, las.y, las.z))
        return None


class PointCloudLoader:
    @staticmethod
    def load(filename: str) -> np.ndarray:
        ext = Path(filename).suffix.lower()[1:]

        try:
            loader = PointCloudLoaderMeta.get_plugin(ext)
            return loader.load(filename)
        except ValueError as e:
            raise ValueError(f'Unsupported file format: {ext}') from e

    @staticmethod
    def _add_suffix(filename: str, expected_ext: str) -> str:
        path = Path(filename)

        if path.suffix.lower() == expected_ext:
            return filename

        return str(path.with_suffix(expected_ext))


    @staticmethod
    def save_las(object_points: np.ndarray, filename: str) -> None:

        filename = PointCloudLoader._add_suffix(filename, '.las')

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.0001, 0.0001, 0.0001])

        las = laspy.LasData(header)

        las.x = object_points[:, 0]
        las.y = object_points[:, 1]
        las.z = object_points[:, 2]

        las.write(filename)

    @staticmethod
    def save_pcd(object_points: np.ndarray, filename: str) -> None:
        filename = PointCloudLoader._add_suffix(filename, '.pcd')

        pc = pypcd4.PointCloud.from_xyz_points(object_points)
        pc.save(filename)
