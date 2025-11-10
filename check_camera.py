import numpy as np
import json

def ray_passes_center(origin: np.ndarray, forward: np.ndarray, tol: float = 1e-3):
    """
    origin: (3,) カメラ位置
    forward: (3,) 視線方向ベクトル（正規化済み推奨）
    tol: 原点からの最短距離の許容値
    """
    # 単位ベクトル化
    dir_norm = forward / np.linalg.norm(forward)
    # レイと原点の最短距離 = |origin × dir_norm|
    distance = np.linalg.norm(np.cross(origin, dir_norm))
    # レイが原点に一番近づく t パラメータ（t>=0 がカメラ前方）
    t = - np.dot(origin, dir_norm)
    passes = (distance <= tol) and (t >= 0)
    return passes, distance, t

# JSON 読み込み
with open("/home/tomoyam/Work/GitHub/ViSNeRF/data/nyx_vol/transforms_train.json", 'r') as f:
    meta = json.load(f)

blender2opencv = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1]
])

results = []
for i, frame in enumerate(meta["frames"]):
    # Blender→OpenCV 座標系補正
    pose = np.array(frame["transform_matrix"]) @ blender2opencv
    
    # カメラ位置（並進ベクトル）
    origin = pose[:3, 3]
    
    # 視線方向ベクトルは回転行列の 3 列目
    R = pose[:3, :3]
    forward = R[:, 2]
    
    # 判定
    ok, dist, tval = ray_passes_center(origin, forward, tol=1e-2)
    results.append({
        "frame": i,
        "origin": origin.tolist(),
        "forward": forward.tolist(),
        "passes": ok,
        "distance": float(dist),
        "t": float(tval)
    })

# 結果表示
for r in results:
    print(f"Frame {r['frame']}: passes={r['passes']}, "
          f"distance={r['distance']:.4f}, t={r['t']:.4f}")
