import torch,os,imageio,sys
from tqdm.auto import tqdm
from ray_utils import get_rays
from utils import *
from ray_utils import ndc_rays_blender
from ViSNeRF import ViSNeRF
import wandb

from pathlib import PurePosixPath, Path

def OctreeRender_trilinear_fast(rays, params, visnerf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk]
        params_chunk = params[chunk_idx * chunk:(chunk_idx + 1) * chunk]
        rgb_map, depth_map = visnerf(rays_chunk, params_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

@torch.no_grad()
def evaluation(test_dataset,visnerf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    torch.cuda.empty_cache()

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1]).cuda()
        params_samples = test_dataset.all_params[0::img_eval_interval][idx]
        params = params_samples.view(-1,params_samples.shape[-1]).cuda()

        rgb_map, _, depth_map, _, _ = renderer(rays, params, visnerf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', visnerf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', visnerf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
        
        combined = np.concatenate((rgb_map, depth_map), axis=1)
        wandb.log({
            "eval/frame_rgb": wandb.Image(rgb_map, caption=f"step_{idx}"),
            "eval/frame_rgbd": wandb.Image(combined, caption=f"step_{idx}")
        })

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    wandb.log({
        "eval/video": wandb.Video(f"{savePath}/{prtx}video.mp4", fps=30, caption="RGB Video"),
        "eval/depth_video": wandb.Video(f"{savePath}/{prtx}depthvideo.mp4", fps=30, caption="Depth Video"),
    })

    torch.cuda.empty_cache()

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        metrics = {"eval/psnr": psnr}
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))

            metrics.update({
                "eval/ssim": float(np.mean(ssims)),
                "eval/lpips_alex": float(np.mean(l_alex)),
                "eval/lpips_vgg": float(np.mean(l_vgg)),
            })
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))

        wandb.log(metrics)
    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,visnerf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _ = renderer(rays, visnerf, chunk=8192, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

# そのまま使う名前変換
def path_to_png_key(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    parts = list(p.parts)
    if parts and parts[0] == "/":
        parts = parts[1:]

    def after_token(token: str):
        idxs = [i for i, s in enumerate(parts) if s == token]
        return parts[idxs[-1] + 1:] if idxs else None

    tail = after_token("CUBEVis")
    if tail is None:
        tail = after_token("data")
    if tail is None:
        tail = parts[:]

    if not tail:
        base = p.stem or p.name or "output"
        return f"{base}.png"

    tail = tail[:]
    tail[-1] = PurePosixPath(tail[-1]).stem or tail[-1]
    return "-".join(tail) + ".png"


@torch.no_grad()
def evaluation_with_gt(
    test_dataset, visnerf, args, renderer,
    N_vis=5, prtx='', N_samples=-1,
    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'
):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]

    # 追加: 出力ルート（要求仕様）
    ds_tag = os.path.basename(str(test_dataset.root_dir).rstrip("/"))
    base_out = os.path.join(args.basedir, args.expname, "render", f"eval__{ds_tag}", f"iters_{args.n_iters}")
    out_gt_dir = os.path.join(base_out, "gt")
    out_rd_dir = os.path.join(base_out, "renders")
    os.makedirs(out_gt_dir, exist_ok=True)
    os.makedirs(out_rd_dir, exist_ok=True)

    # 画像パスの参照（all_image_paths が無ければ image_paths を使う）
    all_img_paths = getattr(test_dataset, "all_image_paths", None)
    assert all_img_paths is not None, "test_dataset に image_paths/all_image_paths が見つかりません"

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    torch.cuda.empty_cache()

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))

    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1]).to(device)
        params_samples = test_dataset.all_params[0::img_eval_interval][idx]
        params = params_samples.view(-1, params_samples.shape[-1]).to(device)

        rgb_map, _, depth_map, _, _ = renderer(
            rays, params, visnerf, chunk=4096, N_samples=N_samples,
            ndc_ray=ndc_ray, white_bg=white_bg, device=device
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        # 可視化深度
        depth_vis, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        # 元フレームのインデックス（サブサンプリング前の位置）
        orig_i = idxs[idx]
        # 元画像のパス -> 保存ファイル名へ
        gt_src_path = all_img_paths[orig_i]
        out_name = path_to_png_key(str(gt_src_path))
        gt_save_path = os.path.join(out_gt_dir, out_name)
        rd_save_path = os.path.join(out_rd_dir, out_name)

        # PSNR/SSIM 等
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[orig_i].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', visnerf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', visnerf.device)
                ssims.append(ssim); l_alex.append(l_a); l_vgg.append(l_v)

            # 正解画像の保存（要求仕様のパス）
            gt_uint8 = (gt_rgb.numpy() * 255).astype('uint8')
            imageio.imwrite(gt_save_path, gt_uint8)

        # 生成画像の保存（要求仕様のパス）
        pred_uint8 = (rgb_map.numpy() * 255).astype('uint8')
        imageio.imwrite(rd_save_path, pred_uint8)

        rgb_maps.append(pred_uint8)
        depth_maps.append(depth_vis)

        # ロギング（任意）
        combined = np.concatenate((pred_uint8, depth_vis), axis=1)
        wandb.log({
            "eval/frame_rgb":   wandb.Image(pred_uint8, caption=f"step_{idx}"),
            "eval/frame_rgbd":  wandb.Image(combined, caption=f"step_{idx}")
        })

    torch.cuda.empty_cache()