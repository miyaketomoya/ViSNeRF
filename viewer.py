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

class ImageSynthetizer:
    def __init__(self, args, device='cuda'):
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.visnerf = self.setup_model(args)
        self.args = args
        self.device = device
        self.blender2opencv = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ])
        self.w ,self.h = args.output_res
        self.focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * self.w
        self.directions = get_ray_directions(self.h, self.w, [self.focal, self.focal])           # (H, W, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.min_params = np.array(args.min_params)
        self.max_params = np.array(args.max_params)
        self.chunk = 8192 * 2 * 2

    @torch.no_grad()
    def visualize(self, theta_deg, phi_deg, param):
        """
        θ, φ, param の組から、
        1 枚の画像を生成して disk に保存します。
        """
        # 1) 学習データと同じtranform_matrixを生成
        print(torch.cuda.memory_allocated(self.device))
        theta = math.radians(theta_deg)
        phi   = math.radians(phi_deg)
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        origin = np.array([x, y, z], dtype=float)
        print(f"sp: {theta}-{phi}, origin: {origin}")
        pose = self.look_at_matrix(origin)

        # 2) レイ生成
        pose = pose @ self.blender2opencv
        c2w = torch.FloatTensor(pose)
        rays = get_rays(self.directions, c2w)  # (H*W, 6)
        rays = rays.to(self.device, non_blocking=True) 

        # 3) パラメータスカラーを正規化
        params = np.array(param)
        params = (params - self.min_params) / (self.max_params - self.min_params) * 2.0 - 1.0
        params = (torch.ones(rays.size(0),1) * torch.FloatTensor(params)).to(self.device, non_blocking=True)

        print(torch.cuda.memory_allocated(self.device))
        
        # print("Rendering...")

        nSamples = min(self.args.nSamples, self.visnerf.nSamples)
        # print(f"nSamples: {nSamples}")
        rgb_flat, _, depth_flat, _, _ = self.renderer(
                rays, params,
                N_samples=nSamples,
                white_bg=True,
                ndc_ray=self.args.ndc_ray,
                device=self.device
            )

        # 6) 画像化＆保存 (RGB)
        rgb = rgb_flat.cpu().reshape(self.h, self.w, 3).numpy()
        rgb_img = self.convert_image_to_uint8(rgb)
        # rgb_name = f"view_t{theta_deg:06.2f}_p{phi_deg:06.2f}_param{param}.png"
        # rgb_path = Path(rgb_name)
        # rgb_path.parent.mkdir(parents=True, exist_ok=True)
        # Image.fromarray(rgb_img).save(str(rgb_path))
        # print(f"Saved RGB → {rgb_path}")

        # 7) 深度マップの可視化＆保存
        #    same as evaluation: normalize to [0,255] with cmap
        from utils import visualize_depth_numpy
        # depth_flat is (H*W,), reshape then to numpy
        depth = depth_flat.cpu().reshape(self.h, self.w).numpy()
        # visualize_depth_numpy returns (depth_viz, raw_depth) according to your utils
        depth_viz, _ = visualize_depth_numpy(depth, self.visnerf.near_far)
        # depth_viz is already uint8 HxW
        # depth_name = f"view_t{theta_deg:06.2f}_p{phi_deg:06.2f}_param{param}_depth.png"
        # depth_path = Path(depth_name)
        # Image.fromarray(depth_viz).save(str(depth_path))
        # print(f"Saved Depth → {depth_path}")

        return rgb_img, depth_viz, x, y, z

    def renderer(self, rays, params, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
        rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
        N_rays_all = rays.shape[0]
        for chunk_idx in tqdm(range(N_rays_all // self.chunk + int(N_rays_all % self.chunk > 0))):
            rays_chunk = rays[chunk_idx * self.chunk:(chunk_idx + 1) * self.chunk]
            params_chunk = params[chunk_idx * self.chunk:(chunk_idx + 1) * self.chunk]

            rgb_map, depth_map = self.visnerf(rays_chunk, params_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
            rgbs.append(rgb_map)
            depth_maps.append(depth_map)
        
        return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


    def convert_image_to_uint8(self, rgb: np.ndarray) -> np.ndarray:
        img = np.clip(rgb, 0.0, 1.0)
        return (img * 255.0).round().astype(np.uint8)

    def look_at_matrix(
                    self,
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

    def setup_model(self, args):
        """
        args からモデルをセットアップします。
        """
        # 1) モデルロード（GPU）
        ckpt = torch.load(args.ckpt, map_location=self.device)
        ckpt['kwargs'].update({'vecSize_params': args.vecSize_params})
        self.visnerf = ViSNeRF(**{**ckpt['kwargs'], 'device': self.device})
        self.visnerf.load(ckpt)
        self.visnerf.eval()
        return self.visnerf

if __name__ == "__main__":
    args = config_parser()
    #   theta_deg: 水平角度 (deg)
    #   phi_deg:   垂直角度 (deg)
    #   param:     パラメータスカラー
    theta_deg = 191.16
    phi_deg   = 75.78
    param     = [50] 

    image_synthesizer = ImageSynthetizer(args)
    image_synthesizer.visualize(theta_deg, phi_deg, param)

