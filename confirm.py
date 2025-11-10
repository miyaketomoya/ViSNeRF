import numpy as np
import math

import json
import torch


def look_at_matrix(origin: np.ndarray,
                   center: np.ndarray = np.zeros(3),
                   world_up: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
    f = center - origin
    f /= np.linalg.norm(f)
    r = np.cross(f, world_up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, u, -f], axis=1)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = origin
    return T

# JSON から読み込んだ Blender 座標系の pose（4×4）
pose_blender = np.array([
    [ 0.25000013, -0.18163509, -0.95105659, -0.951057  ],
    [-0.0       ,  0.98224705, -0.18759192, -0.187592  ],
    [ 0.96824580,  0.04689800,  0.24556189,  0.245562  ],
    [ 0.0       ,  0.0       ,  0.0       ,  1.0       ],
], dtype=float)

# 学習時と同じ Blender→OpenCV 変換
blender2opencv = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=float)

pose = pose_blender @ blender2opencv

# 1) カメラ原点 (x,y,z) を取り出し
origin = pose[:3, 3]   # shape=(3,)
x, y, z = origin
print(f"origin = ({x:.6f}, {y:.6f}, {z:.6f})")

# 2) 半径 r
r = np.linalg.norm(origin)
print(f"distance r = {r:.6f}")

# 3) 球面座標への変換
#   φ = arccos(z / r)          （極からの偏角）
#   θ = atan2(y, x)            （XY 平面上の方位角）
phi = math.degrees(math.acos(z / r))
theta = math.degrees(math.atan2(y, x)) % 360.0

print(f"theta = {theta:.2f}°")
print(f"phi   = {phi:.2f}°")

theta = math.radians(theta)
phi   = math.radians(phi)
x = math.sin(phi) * math.cos(theta)
y = math.sin(phi) * math.sin(theta)
z = math.cos(phi)
origin = np.array([x, y, z], dtype=float)
print(f"origin: {origin}")
c2w_cpu = torch.from_numpy(look_at_matrix(origin)).float()  # (4,4)
print(f"c2w_cpu = {c2w_cpu}")