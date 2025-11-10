import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
import math
import numpy as np
from kornia import create_meshgrid
from tqdm import tqdm
import json
import re

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3]  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    rays = torch.cat([rays_o, rays_d], dim=-1)
    return rays

def get_ray_directions(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def create_cam2world_matrix(forward_vector, origin):
    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.FloatTensor([0, 1, 0]).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

class ViSNeRFDataset(Dataset):
    def __init__(self, args, split='train'):
        self.resolution = args.input_res if split == 'train' else args.output_res
        self.N_vis = -1
        self.split = split
        self.is_stack = False if split == 'train' else True
        self.img_wh = self.resolution
        print('{} resolution: {}'.format(split, self.img_wh))
        self.transform = T.ToTensor()
        self.scene_bbox = torch.tensor(args.bbox)
        self.n_params = args.nParams
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


        # evalが指定されている場合は、学習済みのモデルが対象と判断する
        if self.split=="eval":
            if not args.eval_datadir:
                raise ValueError("eval_datadirを指定してください")
        self.root_dir = args.eval_datadir if split=="eval" else args.datadir
        self.trained_dataset = args.datadir if split=="eval" else None
        
        self.white_bg = True
        # self.near_far = [2.0,6.0]
        self.near_far = [0.1,5.0]
        # self.near_far = [0.5,4.0]
        # self.near_far = [0.01, 6.0]
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.load_all()

    
    def load_all(self):
        """
        CUBEVis:
        root_dir/
            cam_info.json
            sim_param_info.json
            mu*/cam*/frame_00001.jpg ...
        を読み、self.all_rays / self.all_rgbs / self.all_params を構築する。
        params は [mu, time]（ともに [-1,1] 正規化）。
        """
        # --- メタ読込 ---
        cam_info_path = os.path.join(self.root_dir, "cam_info.json")
        sim_info_path = os.path.join(self.root_dir, "sim_param_info.json")
        with open(cam_info_path, "r") as f:
            cam_info = json.load(f)
        with open(sim_info_path, "r") as f:
            sim_info = json.load(f)
    

        intr = cam_info["intrinsics"]
        width  = int(intr["width"])
        height = int(intr["height"])
        fx = float(intr["fx"])
        fy = float(intr["fy"])
        cx = float(intr["cx"])
        cy = float(intr["cy"])

        # 画像サイズは self.img_wh が指定されていればそれを優先
        if getattr(self, "img_wh", None) is None:
            self.img_wh = (width, height)
        w, h = self.img_wh

        # 射影行列（必要なら利用）
        self.intrinsics = torch.tensor([[fx, 0, cx],
                                        [0,  fy, cy],
                                        [0,   0,  1]], dtype=torch.float32)

        # ray directions （get_ray_directions は [fx,fy] を受けられる想定）
        self.directions = get_ray_directions(h, w, [fx, fy],center=[cx, cy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        # 正規化レンジ
        mu_min = float(sim_info["mu"]["min"])
        mu_max = float(sim_info["mu"]["max"])
        t_min  = float(sim_info["time"]["min"])
        t_max  = float(sim_info["time"]["max"])
        timestep_map = {int(k): int(v) for k, v in sim_info["timestep_mapping"].items()}

        if self.split=="eval":
            # もしtrained_datasetが指定されている場合、学習時の正規化方法を使用する
            # 学習時のmin,maxに上書き.
            # timestep_mapは元のものを使用しなければいけない
            trained_dataset_sim_param = os.path.join(self.trained_dataset, "sim_param_info.json")
            with open(trained_dataset_sim_param, "r") as f:
                sim_info = json.load(f)
            mu_min = float(sim_info["mu"]["min"])
            mu_max = float(sim_info["mu"]["max"])
            t_min  = float(sim_info["time"]["min"])
            t_max  = float(sim_info["time"]["max"]) 

        # ViSNeRF 互換のため保持（ベクトル演算で使えるよう 2 次元）
        self.min_params = torch.tensor([mu_min, t_min], dtype=torch.float32)
        self.max_params = torch.tensor([mu_max, t_max], dtype=torch.float32)

        # cam_info のテーブル（R/T は world->camera）
        cam_table = {}
        # for e in cam_info.get("entries", []):
        #     cam_table[e["name"]] = dict(
        #         R_wc=torch.tensor(e["R"], dtype=torch.float32),   # 3x3
        #         T_wc=torch.tensor(e["T"], dtype=torch.float32)    # 3,
        #     )

        for e in cam_info.get("entries", []):
            cam_table[e["name"]] = dict(
                R_c2w=torch.tensor(e["R"], dtype=torch.float32),  # JSONのRは c2w
                T_wc =torch.tensor(e["T"], dtype=torch.float32),  # JSONのTは w2c
            )

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_params = []
        self.downsample = 1.0
        self.all_image_paths = []

        # mu ディレクトリ列挙
        mu_dirs = sorted(
            d for d in os.listdir(self.root_dir)
            if d.startswith("mu") and os.path.isdir(os.path.join(self.root_dir, d))
        )

        # ユーティリティ: frame_XXXXX.jpg から番号を取る
        frame_pat = re.compile(r"frame_(\d+)\.jpg$")

        for mu_dirname in mu_dirs:
            mu_val = float(mu_dirname[2:])  # "mu0.5" → 0.5
            mu_dir = os.path.join(self.root_dir, mu_dirname)

            # cam ディレクトリ列挙
            cam_dirs = sorted(
                d for d in os.listdir(mu_dir)
                if d.startswith("cam") and os.path.isdir(os.path.join(mu_dir, d))
            )

            for cam_name in cam_dirs:
                if cam_name not in cam_table:
                    continue
                # R_wc = cam_table[cam_name]["R_wc"]
                # T_wc = cam_table[cam_name]["T_wc"]
                R_c2w = cam_table[cam_name]["R_c2w"]
                T_wc  = cam_table[cam_name]["T_wc"]
                C_w   = -R_c2w @ T_wc

                # c2w
                # R_c2w = R_wc.t()
                # C_w = -R_c2w @ T_wc
                c2w = torch.eye(4, dtype=torch.float32)
                c2w[:3, :3] = R_c2w
                c2w[:3, 3] = C_w
                self.poses.append(c2w)

                cam_dir = os.path.join(mu_dir, cam_name)
                frame_files = sorted(
                    f for f in os.listdir(cam_dir)
                    if f.startswith("frame_") and f.endswith(".jpg")
                )
                image_length = len(frame_files)

                # ★ ここで枚数選定（test は常に3枚、eval は総枚数が多いときだけ3枚に間引く）
                if self.split == "test" and image_length >= 3:
                    idx_selected = [0, image_length // 3, (2 * image_length) // 3]
                # elif self.split == "eval" and (len(cam_dirs) * image_length > 300) and image_length >= 3:
                #     idx_selected = [0, image_length // 3, (2 * image_length) // 3]
                else:
                    idx_selected = list(range(image_length))

                for idx in idx_selected:
                    fname = frame_files[idx]
                    m = frame_pat.match(fname)
                    if not m:
                        continue
                    frame_idx = int(m.group(1))  # 1-based
                    if frame_idx not in timestep_map:
                        continue
                    real_t = timestep_map[frame_idx]

                    # 画像ロード
                    image_path = os.path.join(cam_dir, fname)
                    self.image_paths.append(image_path)
                    img = Image.open(image_path).convert("RGB")
                    if img.size != (w, h):
                        img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img).view(3, -1).permute(1, 0)
                    self.all_rgbs.append(img)

                    # rays
                    rays = get_rays(self.directions, c2w)
                    self.all_rays.append(rays)

                    # params = [mu, time] を [-1,1]
                    params_np = np.array([mu_val, real_t], dtype=np.float32)
                    params_norm = (params_np - self.min_params.numpy()) / \
                                (self.max_params.numpy() - self.min_params.numpy()) * 2.0 - 1.0
                    params = torch.ones(rays.size(0), 1, dtype=torch.float32) * torch.from_numpy(params_norm)
                    self.all_params.append(params)

                    self.all_image_paths.append(image_path)

        # stack/cat は元の実装に合わせる
        self.poses = torch.stack(self.poses) if len(self.poses) else torch.empty(0, 4, 4)

        if not getattr(self, "is_stack", False):
            self.all_rays   = torch.cat(self.all_rays, 0) if self.all_rays else torch.empty(0)
            self.all_rgbs   = torch.cat(self.all_rgbs, 0) if self.all_rgbs else torch.empty(0)
            self.all_params = torch.cat(self.all_params, 0) if self.all_params else torch.empty(0)
        else:
            # (F,H*W, ·) 形式で揃える場合
            self.all_rays   = torch.stack(self.all_rays, 0) if self.all_rays else torch.empty(0)
            self.all_params = torch.stack(self.all_params, 0) if self.all_params else torch.empty(0)
            # 画像は (F,H,W,3) に戻す
            if self.all_rgbs:
                F = len(self.all_rgbs)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(F, h, w, 3)
            else:
                self.all_rgbs = torch.empty(0)

        print(self.all_rays.size(), self.all_params.size())


# class ParaviewDataset(Dataset):
#     def __init__(self, args, split='train'):
#         self.resolution = args.input_res+args.input_res if split == 'train' else args.output_res+args.output_res
#         self.N_vis = -1
#         self.root_dir = args.train_path if split == 'train' else args.test_path
#         self.split = split
#         self.is_stack = False if split == 'train' else True
#         self.img_wh = self.resolution
#         print('{} resolution: {}'.format(split, self.img_wh))
#         self.transform = T.ToTensor()
#         self.scene_bbox = torch.tensor(args.bbox)
#         self.min_params = np.array(args.min_params)
#         self.max_params = np.array(args.max_params)
#         self.n_params = args.nParams
#         print(self.scene_bbox)
#         self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
#         self.white_bg = True
#         # self.near_far = [2.0,6.0]
#         self.near_far = [0.1,2.0]
#         # self.near_far = [0.5,4.0]
#         # self.near_far = [0.01, 6.0]
#         self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
#         self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
#         self.load_all()

    
#     def load_all(self):
#         w, h = self.img_wh
#         self.focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * self.img_wh[0]  # original focal length


#         self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
#         self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
#         self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

#         self.image_paths = []
#         self.poses = []
#         self.all_rays = []
#         self.all_rgbs = []
#         self.all_params = []

#         img_files = os.listdir('{}'.format(self.root_dir))
#         # img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[2]))
#         img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[self.n_params+1]))


#         for i, filename in enumerate(img_files):
#             # _,timestep,theta,phi, x, y, z = 
#             parse_list = filename.split('.png')[0].split('_')[1:]
#             params = []
#             for j in range(self.n_params):
#                 params.append(float(parse_list[j]))
#             theta,phi, x, y, z = parse_list[self.n_params:]
#             # timestep = int(timestep)
#             theta = float(theta) / 90.0
#             phi = float(phi) / 180.0
#             x = float(x)
#             y = float(y)
#             z = float(z)

#             camera_pivot = torch.FloatTensor([0,0,0])
#             # camera_origins = torch.FloatTensor([x,y,z]) * 4.0311
#             camera_origins = torch.FloatTensor([x,y,z])

#             swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
#             mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])
#             blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

#             c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins)
#             c2w = c2w[0]
#             c2w = swap_row @ c2w
#             c2w = c2w * mask
#             c2w = c2w + 1e-8
#             print(c2w)
#             exit()
#             c2w = c2w @ blender2opencv

#             self.poses += [c2w]

#             image_path = '{}/{}'.format(self.root_dir, filename)
#             self.image_paths += [image_path]
#             img = Image.open(image_path)

#             if img.size != self.resolution:
#                 img = img.resize(self.img_wh, Image.LANCZOS)
#             img = self.transform(img)  # (3, h, w)
#             img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

#             rays = get_rays(self.directions, c2w)

#             self.all_rgbs.append(img)
#             self.all_rays.append(rays)

#             param_list = []
#             for j in range(self.n_params):
#                 p = (params[j] - self.min_params[j]) / (self.max_params[j] - self.min_params[j]) * 2.0 - 1.0
#                 param_j = torch.ones(rays.size(0),1) * p
#                 # print(param.size())
#                 param_list.append(param_j)
#             # time = torch.ones(rays.size(0),1) * (timestep - self.time_range[0]) / (self.time_range[1] - self.time_range[0]) * 2.0 - 1.0
#             # if self.n_params > 1:
#             param = torch.cat(param_list, -1)
#                 # print(param.size())
#                 # print(param.min(), param.max())
#             self.all_params += [param]
#             # print(self.all_params)
#             # else:
#                 # self.all_params += [param_list]

#             # self.all_params = torch.tensor(self.all_params)

#         self.poses = torch.stack(self.poses)
#         if not self.is_stack:
#             self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
#             self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
#             self.all_params = torch.cat(self.all_params, 0)  # (len(self.meta['frames])*h*w, 3)
#         else:
#             self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
#             self.all_params = torch.stack(self.all_params, 0)  # (len(self.meta['frames]),h*w, 3)
#             self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)

# class VisNeRFSmokeRingDataset(Dataset):
#     def __init__(self, args, split='train'):
#         # ここでデータセットの初期化を行う
#         pass



# 以下はbugfixのまえ
def get_rays(directions, c2w): rays_d = directions @ c2w[:3, :3].T # (H, W, 3) rays_o = c2w[:3, 3].expand(rays_d.shape) # (H, W, 3) rays_d = rays_d.view(-1, 3) rays_o = rays_o.view(-1, 3) rays = torch.cat([rays_o, rays_d], dim=-1) return rays def get_ray_directions(H, W, focal, center=None): grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5 i, j = grid.unbind(-1) cent = center if center is not None else [W / 2, H / 2] directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1) # (H, W, 3) return directions def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor: return vectors / (torch.norm(vectors, dim=-1, keepdim=True)) def create_cam2world_matrix(forward_vector, origin): forward_vector = normalize_vecs(forward_vector) up_vector = torch.FloatTensor([0, 1, 0]).expand_as(forward_vector) right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1)) up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1)) rotation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1) rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1) translation_matrix = torch.eye(4).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1) translation_matrix[:, :3, 3] = origin cam2world = (translation_matrix @ rotation_matrix)[:, :, :] assert(cam2world.shape[1:] == (4, 4)) return cam2world class ViSNeRFDataset(Dataset): def __init__(self, args, split='train'): self.resolution = args.input_res if split == 'train' else args.output_res self.N_vis = -1 self.split = split self.is_stack = False if split == 'train' else True self.img_wh = self.resolution print('{} resolution: {}'.format(split, self.img_wh)) self.transform = T.ToTensor() self.scene_bbox = torch.tensor(args.bbox) self.n_params = args.nParams self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # evalが指定されている場合は、学習済みのモデルが対象と判断する if self.split=="eval": if not args.eval_datadir: raise ValueError("eval_datadirを指定してください") self.root_dir = args.eval_datadir if split=="eval" else args.datadir self.trained_dataset = args.datadir if split=="eval" else None self.white_bg = True # self.near_far = [2.0,6.0] self.near_far = [0.1,5.0] # self.near_far = [0.5,4.0] # self.near_far = [0.01, 6.0] self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3) self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3) self.load_all() def load_all(self): """ CUBEVis: root_dir/ cam_info.json sim_param_info.json mu*/cam*/frame_00001.jpg ... を読み、self.all_rays / self.all_rgbs / self.all_params を構築する。 params は [mu, time]（ともに [-1,1] 正規化）。 """ # --- メタ読込 --- cam_info_path = os.path.join(self.root_dir, "cam_info.json") sim_info_path = os.path.join(self.root_dir, "sim_param_info.json") with open(cam_info_path, "r") as f: cam_info = json.load(f) with open(sim_info_path, "r") as f: sim_info = json.load(f) intr = cam_info["intrinsics"] width = int(intr["width"]) height = int(intr["height"]) fx = float(intr["fx"]) fy = float(intr["fy"]) cx = float(intr["cx"]) cy = float(intr["cy"]) # 画像サイズは self.img_wh が指定されていればそれを優先 if getattr(self, "img_wh", None) is None: self.img_wh = (width, height) w, h = self.img_wh # 射影行列（必要なら利用） self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32) # ray directions （get_ray_directions は [fx,fy] を受けられる想定） self.directions = get_ray_directions(h, w, [fx, fy]) # (h, w, 3) self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True) # 正規化レンジ mu_min = float(sim_info["mu"]["min"]) mu_max = float(sim_info["mu"]["max"]) t_min = float(sim_info["time"]["min"]) t_max = float(sim_info["time"]["max"]) timestep_map = {int(k): int(v) for k, v in sim_info["timestep_mapping"].items()} if self.split=="eval": # もしtrained_datasetが指定されている場合、学習時の正規化方法を使用する # 学習時のmin,maxに上書き. # timestep_mapは元のものを使用しなければいけない trained_dataset_sim_param = os.path.join(self.trained_dataset, "sim_param_info.json") with open(trained_dataset_sim_param, "r") as f: sim_info = json.load(f) mu_min = float(sim_info["mu"]["min"]) mu_max = float(sim_info["mu"]["max"]) t_min = float(sim_info["time"]["min"]) t_max = float(sim_info["time"]["max"]) # ViSNeRF 互換のため保持（ベクトル演算で使えるよう 2 次元） self.min_params = torch.tensor([mu_min, t_min], dtype=torch.float32) self.max_params = torch.tensor([mu_max, t_max], dtype=torch.float32) # cam_info のテーブル（R/T は world->camera） cam_table = {} for e in cam_info.get("entries", []): cam_table[e["name"]] = dict( R_wc=torch.tensor(e["R"], dtype=torch.float32), # 3x3 T_wc=torch.tensor(e["T"], dtype=torch.float32) # 3, ) self.image_paths = [] self.poses = [] self.all_rays = [] self.all_rgbs = [] self.all_params = [] self.downsample = 1.0 self.all_image_paths = [] # mu ディレクトリ列挙 mu_dirs = sorted( d for d in os.listdir(self.root_dir) if d.startswith("mu") and os.path.isdir(os.path.join(self.root_dir, d)) ) # ユーティリティ: frame_XXXXX.jpg から番号を取る frame_pat = re.compile(r"frame_(\d+)\.jpg$") for mu_dirname in mu_dirs: mu_val = float(mu_dirname[2:]) # "mu0.5" → 0.5 mu_dir = os.path.join(self.root_dir, mu_dirname) # cam ディレクトリ列挙 cam_dirs = sorted( d for d in os.listdir(mu_dir) if d.startswith("cam") and os.path.isdir(os.path.join(mu_dir, d)) ) for cam_name in cam_dirs: if cam_name not in cam_table: continue R_wc = cam_table[cam_name]["R_wc"] T_wc = cam_table[cam_name]["T_wc"] # c2w R_c2w = R_wc.t() C_w = -R_c2w @ T_wc c2w = torch.eye(4, dtype=torch.float32) c2w[:3, :3] = R_c2w c2w[:3, 3] = C_w self.poses.append(c2w) cam_dir = os.path.join(mu_dir, cam_name) frame_files = sorted( f for f in os.listdir(cam_dir) if f.startswith("frame_") and f.endswith(".jpg") ) image_length = len(frame_files) # ★ ここで枚数選定（test は常に3枚、eval は総枚数が多いときだけ3枚に間引く） if self.split == "test" and image_length >= 3: idx_selected = [0, image_length // 3, (2 * image_length) // 3] # elif self.split == "eval" and (len(cam_dirs) * image_length > 300) and image_length >= 3: # idx_selected = [0, image_length // 3, (2 * image_length) // 3] else: idx_selected = list(range(image_length)) for idx in idx_selected: fname = frame_files[idx] m = frame_pat.match(fname) if not m: continue frame_idx = int(m.group(1)) # 1-based if frame_idx not in timestep_map: continue real_t = timestep_map[frame_idx] # 画像ロード image_path = os.path.join(cam_dir, fname) self.image_paths.append(image_path) img = Image.open(image_path).convert("RGB") if img.size != (w, h): img = img.resize(self.img_wh, Image.LANCZOS) img = self.transform(img).view(3, -1).permute(1, 0) self.all_rgbs.append(img) # rays rays = get_rays(self.directions, c2w) self.all_rays.append(rays) # params = [mu, time] を [-1,1] params_np = np.array([mu_val, real_t], dtype=np.float32) params_norm = (params_np - self.min_params.numpy()) / \ (self.max_params.numpy() - self.min_params.numpy()) * 2.0 - 1.0 params = torch.ones(rays.size(0), 1, dtype=torch.float32) * torch.from_numpy(params_norm) self.all_params.append(params) self.all_image_paths.append(image_path) # stack/cat は元の実装に合わせる self.poses = torch.stack(self.poses) if len(self.poses) else torch.empty(0, 4, 4) if not getattr(self, "is_stack", False): self.all_rays = torch.cat(self.all_rays, 0) if self.all_rays else torch.empty(0) self.all_rgbs = torch.cat(self.all_rgbs, 0) if self.all_rgbs else torch.empty(0) self.all_params = torch.cat(self.all_params, 0) if self.all_params else torch.empty(0) else: # (F,H*W, ·) 形式で揃える場合 self.all_rays = torch.stack(self.all_rays, 0) if self.all_rays else torch.empty(0) self.all_params = torch.stack(self.all_params, 0) if self.all_params else torch.empty(0) # 画像は (F,H,W,3) に戻す if self.all_rgbs: F = len(self.all_rgbs) self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(F, h, w, 3) else: self.all_rgbs = torch.empty(0) print(self.all_rays.size(), self.all_params.size()) class ParaviewDataset(Dataset): def __init__(self, args, split='train'): self.resolution = args.input_res+args.input_res if split == 'train' else args.output_res+args.output_res self.N_vis = -1 self.root_dir = args.train_path if split == 'train' else args.test_path self.split = split self.is_stack = False if split == 'train' else True self.img_wh = self.resolution print('{} resolution: {}'.format(split, self.img_wh)) self.transform = T.ToTensor() self.scene_bbox = torch.tensor(args.bbox) self.min_params = np.array(args.min_params) self.max_params = np.array(args.max_params) self.n_params = args.nParams print(self.scene_bbox) self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) self.white_bg = True # self.near_far = [2.0,6.0] self.near_far = [0.1,2.0] # self.near_far = [0.5,4.0] # self.near_far = [0.01, 6.0] self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3) self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3) self.load_all() def load_all(self): w, h = self.img_wh self.focal = 0.5 / np.tan(0.5 * 30/180*math.pi) * self.img_wh[0] # original focal length self.directions = get_ray_directions(h, w, [self.focal,self.focal]) # (h, w, 3) self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True) self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float() self.image_paths = [] self.poses = [] self.all_rays = [] self.all_rgbs = [] self.all_params = [] img_files = os.listdir('{}'.format(self.root_dir)) # img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[2])) img_files.sort(key=lambda x:float(x.split('.png')[0].split('_')[self.n_params+1])) for i, filename in enumerate(img_files): # _,timestep,theta,phi, x, y, z = parse_list = filename.split('.png')[0].split('_')[1:] params = [] for j in range(self.n_params): params.append(float(parse_list[j])) theta,phi, x, y, z = parse_list[self.n_params:] # timestep = int(timestep) theta = float(theta) / 90.0 phi = float(phi) / 180.0 x = float(x) y = float(y) z = float(z) camera_pivot = torch.FloatTensor([0,0,0]) # camera_origins = torch.FloatTensor([x,y,z]) * 4.0311 camera_origins = torch.FloatTensor([x,y,z]) swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]]) blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins) c2w = c2w[0] c2w = swap_row @ c2w c2w = c2w * mask c2w = c2w + 1e-8 print(c2w) exit() c2w = c2w @ blender2opencv self.poses += [c2w] image_path = '{}/{}'.format(self.root_dir, filename) self.image_paths += [image_path] img = Image.open(image_path) if img.size != self.resolution: img = img.resize(self.img_wh, Image.LANCZOS) img = self.transform(img) # (3, h, w) img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB rays = get_rays(self.directions, c2w) self.all_rgbs.append(img) self.all_rays.append(rays) param_list = [] for j in range(self.n_params): p = (params[j] - self.min_params[j]) / (self.max_params[j] - self.min_params[j]) * 2.0 - 1.0 param_j = torch.ones(rays.size(0),1) * p # print(param.size()) param_list.append(param_j) # time = torch.ones(rays.size(0),1) * (timestep - self.time_range[0]) / (self.time_range[1] - self.time_range[0]) * 2.0 - 1.0 # if self.n_params > 1: param = torch.cat(param_list, -1) # print(param.size()) # print(param.min(), param.max()) self.all_params += [param] # print(self.all_params) # else: # self.all_params += [param_list] # self.all_params = torch.tensor(self.all_params) self.poses = torch.stack(self.poses) if not self.is_stack: self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3) self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3) self.all_params = torch.cat(self.all_params, 0) # (len(self.meta['frames])*h*w, 3) else: self.all_rays = torch.stack(self.all_rays, 0) # (len(self.meta['frames]),h*w, 3) self.all_params = torch.stack(self.all_params, 0) # (len(self.meta['frames]),h*w, 3) self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3) # (len(self.meta['frames]),h,w,3) class VisNeRFSmokeRingDataset(Dataset): def __init__(self, args, split='train'): # ここでデータセットの初期化を行う pass