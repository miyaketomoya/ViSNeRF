import math
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ViSNeRF import ViSNeRF
from opt import config_parser
from data_loader import get_rays, get_ray_directions
from tqdm import tqdm
from renderer import OctreeRender_trilinear_fast as renderer

def convert_image_to_uint8(rgb: np.ndarray) -> np.ndarray:
    img = np.clip(rgb, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)

def look_at_matrix(
                origin: np.ndarray,
                center: np.ndarray = np.zeros(3),
                world_up: np.ndarray = np.array([0, 1, 0])
                ) -> np.ndarray:
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

@torch.no_grad()
def evaluate_one_view(visnerf, args, device):
    """
    θ, φ, param の組を args から読み、
    1 枚の画像を生成して disk に保存します。
    """
    # 1) カメラ行列（CPU）
    theta = math.radians(args.theta_deg)
    phi   = math.radians(args.phi_deg)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    origin = np.array([x, y, z], dtype=float)
    print(f"sp: {theta}-{phi}, origin: {origin}")
    pose = look_at_matrix(origin)
    print(f"pose: {pose}")

    blender2opencv = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ])

    # 2) レイ生成（CPU）
    # 出力解像度
    W, H = args.output_res
    # data_loader と同じ 30° FOV から focal を算出
    focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * W
    directions = get_ray_directions(H, W, [focal, focal])           # (H, W, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    intrinsics = torch.tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]]).float()
    downsample = 1.0

    pose = pose @ blender2opencv
    c2w = torch.FloatTensor(pose)
    rays = get_rays(directions, c2w)  # (H*W, 6)

    params = np.array(args.param)
    min_params = np.array(args.min_params)
    max_params = np.array(args.max_params)
    params = (params - min_params) / (max_params - min_params) * 2.0 - 1.0
    params = torch.ones(rays.size(0),1) * torch.FloatTensor(params)

    print(f"rays: {rays[0]}, params: {params[0]}")

    # 4) モデルロード（GPU）
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt['kwargs'].update({'vecSize_params': args.vecSize_params})
    vis = ViSNeRF(**{**ckpt['kwargs'], 'device': device})
    vis.load(ckpt)
    vis.eval()

    # 5) チャンク分割レンダリング
    chunk_size = 512
    torch.cuda.empty_cache()
    rgb_chunks = []

    print("Rendering...")
    for i in tqdm(range(0, rays.shape[0], chunk_size)):
        rays_chunk   = rays[i:i+chunk_size].to(device, non_blocking=True)
        params_chunk = params[i:i+chunk_size].to(device, non_blocking=True)
        rgb_flat, _, _ ,_, _ = renderer(
            rays_chunk, params_chunk, vis,
            chunk=chunk_size,
            N_samples=args.nSamples,
            white_bg=True,
            ndc_ray=args.ndc_ray,
            device=device
        )
        rgb_chunks.append(rgb_flat.cpu())
        del rays_chunk, params_chunk, rgb_flat
        torch.cuda.empty_cache()

    rgb_flat = torch.cat(rgb_chunks, dim=0)  # (H*W, 3)

    # 6) 画像化＆保存
    rgb = rgb_flat.reshape(H, W, 3).numpy()
    img = convert_image_to_uint8(rgb)
    out_name = f"view_t{args.theta_deg:03d}_p{args.phi_deg:03d}_param{args.param}.png"
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(out_path))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    args = config_parser()
    # コマンドライン or config ファイルに以下を追加しておくこと:
    #   theta_deg: 水平角度 (deg)
    #   phi_deg:   垂直角度 (deg)
    #   param:     パラメータスカラー
    # 例: python viewer.py --config cube.txt --theta_deg 45 --phi_deg 30 --param 50

    args.theta_deg = 191.16
    args.phi_deg = 75.78
    args.param = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_one_view(None, args, device)