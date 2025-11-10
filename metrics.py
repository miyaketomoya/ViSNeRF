#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from metrics_utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from metrics_utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
import importlib.util

from opt import config_parser


def _load_defaults_py(py_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    defaults = getattr(mod, "DEFAULTS", None)
    if not isinstance(defaults, dict):
        raise RuntimeError(f"{py_path} に DEFAULTS dict が見つかりません")
    return defaults


def parse_args():
    p = ArgumentParser("Render with InSituNet Generator")
    p.add_argument("--config", type=str, required=True,
                   help="default.py 等の設定ファイルへのパス（DEFAULTS を含む）")

    # 1st pass で config を読む
    pre, _ = p.parse_known_args()
    defaults = _load_defaults_py(pre.config)

    # デフォルト値を反映
    p.set_defaults(**defaults)

    args = p.parse_args()
    return args


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1-ms_ssims[-1])/2)

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                            }
                                                        )

                with open(scene_dir + "/test" + "/" + method + "/results.json", 'w') as fp:
                    json.dump(full_dict[scene_dir], fp, indent=True)
                with open(scene_dir + "/test" + "/" + method + "/per_view.json", 'w') as fp:
                    json.dump(per_view_dict[scene_dir], fp, indent=True)
        
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

def evaluate_eval(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    print(model_paths)

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # eval用に取得
            # scene_dir 直下で eval__* のディレクトリを取得
            eval_dirs = sorted(
                [p for p in Path(scene_dir).iterdir() if p.is_dir() and p.name.startswith("eval__")],
                key=lambda p: p.name
            )
            eval_names = [p.name for p in eval_dirs]  # 例: ["eval__A", "eval__B"]
            print("find_eval_dir",eval_names)

            for eval_name in eval_names:
                print("dir:",eval_name)
                eval_dir = Path(scene_dir) / eval_name

                for method in os.listdir(eval_dir):
                    print("Method:", method)

                    full_dict[scene_dir][method] = {}
                    per_view_dict[scene_dir][method] = {}
                    full_dict_polytopeonly[scene_dir][method] = {}
                    per_view_dict_polytopeonly[scene_dir][method] = {}

                    # method_dir = eval_dir
                    method_dir = eval_dir / method
                    gt_dir = method_dir/ "gt"
                    renders_dir = method_dir / "renders"
                    renders, gts, image_names = readImages(renders_dir, gt_dir)

                    ssims = []
                    psnrs = []
                    lpipss = []
                    lpipsa = []
                    ms_ssims = []
                    Dssims = []
                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                        ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                        lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                        Dssims.append((1-ms_ssims[-1])/2)

                    print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                    print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                    full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                            "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                            "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                            "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                        )
                    per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                                "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                                "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                                "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                                "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                                }
                                                            )

                    with open(scene_dir + "/" + str(eval_name) + "/" + method + "/results.json", 'w') as fp:
                        json.dump(full_dict[scene_dir], fp, indent=True)
                    with open(scene_dir + "/" + str(eval_name) + "/" + method + "/per_view.json", 'w') as fp:
                        json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

if __name__ == "__main__":
    args = config_parser()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    eval_dir = [os.path.join(args.basedir,args.expname,"render")]

    # print("==evaluationg test data==")
    # evaluate(eval_dir)

    print("==evaluationg eval data==")
    evaluate_eval(eval_dir)
