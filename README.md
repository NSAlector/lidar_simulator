# Ray Tracing Renderer

Система трассировки лучей (ray tracing) для 3D-визуализации. Реализована с нуля на Python с использованием NumPy.

## Возможности

-  Трассировка лучей с рекурсивным отражением и преломлением
-  Поддержка сфер и треугольных мешей (загрузка STL файлов)
-  Материалы: диффузные, зеркальные, стеклянные
-  BVH (Bounding Volume Hierarchy) для ускорения рендеринга
-  Камера с настраиваемым FOV
-  Источники света и тени
-  Текстурирование (UV-развертка для сфер, процедурное для мешей)

## Структура проекта
```
raytracer/
├── core/                      # Базовые классы
│   ├── camera.py             # Камера
│   ├── light.py              # Источник света
│   ├── material.py           # Материалы
│   └── plane.py              # Плоскость (пол)
├── geometry/                  # Геометрические объекты
│   ├── sphere.py             # Сфера
│   ├── mesh.py               # Меш (треугольники)
│   ├── mesh_group.py         # Группа мешей
│   ├── bvh.py                # BVH ускорение
│   └── textured_sphere.py    # Текстурированная сфера
├── renderer/                  # Рендереры
│   ├── base.py               # Базовый рендерер
│   ├── mesh_renderer.py      # Рендерер мешей
│   ├── sphere_renderer.py    # Рендерер сфер
│   ├── mesh_group_renderer.py # Рендерер для группы мешей 
│   └── unified_renderer.py   # Универсальный рендерер
└── requirements.txt
```
## Примеры:
<img width="347" height="279" alt="refraction_sphere_behind" src="https://github.com/user-attachments/assets/f051b77a-74a5-47a9-9237-e7b86dbea36b" />
<img width="800" height="600" alt="mirror_cube_angled_45" src="https://github.com/user-attachments/assets/72deccfb-106e-4c92-b1f1-d5ae67dae004" />
<img width="672" height="565" alt="mig29_with_reflection" src="https://github.com/user-attachments/assets/0f470fc8-a1e3-44bd-a8c6-bbbf691683ef" />
<img width="800" height="600" alt="textured_earth_sphere_mesh" src="https://github.com/user-attachments/assets/a198433a-4f6d-41be-aadd-11b888715ccb" />


## Контактные данные:
- Telegram: @dorohovgleb
- Vk: https://vk.com/id222779271
- Phone: +79026370655
