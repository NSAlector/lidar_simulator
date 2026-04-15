"""
Модуль для загрузки и подготовки STL-моделей.
"""
import os
import numpy as np
from stl import mesh as stl_mesh
from OpenGL.GL import (
    glGenLists, glNewList, glEndList, glBegin, glEnd,
    glNormal3fv, glVertex3fv, glTexCoord2f,
    GL_COMPILE, GL_TRIANGLES,
)


import math

def load_stl(filename: str) -> stl_mesh.Mesh | None:
    """
    Загружает STL-файл

    :param filename: Имя STL-файла.
    :return: Объект Mesh или None при ошибке.
    """
    stl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        loaded_mesh = stl_mesh.Mesh.from_file(stl_path)
        loaded_mesh.rotate([1.0, 0.0, 0.0], math.radians(90.0))
        print(f"STL модель загружена: {stl_path}")
        return loaded_mesh
    except Exception as e:
        print(f"Ошибка загрузки STL: {e}")
        return None


def build_display_list(mesh_data: stl_mesh.Mesh) -> int:
    """
    Создаёт OpenGL display list для переданного меша.
    Модель центрируется и масштабируется так, чтобы вписаться в куб [-1, 1].

    :param mesh_data: Объект Mesh, полученный из load_stl().
    :return: Идентификатор display list.
    """
    all_points = mesh_data.vectors.reshape(-1, 3)
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    size = (max_coords - min_coords).max()
    scale = 2.0 / size if size > 0 else 1.0

    span_x = max_coords[0] - min_coords[0]
    span_z = max_coords[2] - min_coords[2]
    if span_x == 0: span_x = 1.0
    if span_z == 0: span_z = 1.0

    dl = glGenLists(1)
    glNewList(dl, GL_COMPILE)
    glBegin(GL_TRIANGLES)
    for i in range(len(mesh_data.vectors)):
        normal = mesh_data.normals[i]
        n_len = np.linalg.norm(normal)
        if n_len > 0:
            normal = normal / n_len
        glNormal3fv(normal)
        for vertex in mesh_data.vectors[i]:
            v = (vertex - center) * scale
            # Изначально нормализуем координаты
            norm_x = (vertex[0] - min_coords[0]) / span_x
            norm_z = (vertex[2] - min_coords[2]) / span_z
            
            # Поворот на 180 градусов (разворачиваем обе оси)
            u = 1.0 - norm_z
            v_tex = 1.0 - norm_x
            
            glTexCoord2f(u, v_tex)
            glVertex3fv(v)
    glEnd()
    glEndList()

    return dl
