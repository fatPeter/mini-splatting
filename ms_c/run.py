import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np

from Haar3D_torch import haar3D, inv_haar3D
from utils.sh_utils import SH2RGB
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json



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

            test_dir = Path(scene_dir) / "test_compressed"

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

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

    return torch.tensor(ssims).mean().item(), torch.tensor(psnrs).mean().item(), torch.tensor(lpipss).mean().item()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))



def compress(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        dir_path=os.path.join(scene.model_path,
                    "point_cloud",
                    "iteration_" + str(scene.loaded_iter),
                    "compressed")
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)
        

        depth=16
        Qstep=0.02


        # voxlize
        pos=gaussians._xyz
        rgb=SH2RGB(gaussians._features_dc+0)[:,0]
        pos_voxlized=(pos-pos.min())/(pos.max()-pos.min())
        pos_voxlized=torch.round(pos_voxlized*(2**depth-1))
        pos_voxlized, pos_idx = np.unique(pos_voxlized.detach().cpu().numpy(), axis=0, return_index=True)

        pos_remain = pos[pos_idx]
        rgb=rgb[pos_idx]

        pos_voxlized=(pos_remain-pos_remain.min())/(pos_remain.max()-pos_remain.min())
        pos_voxlized=torch.round(pos_voxlized*(2**depth-1))
        pos_voxlized=pos_voxlized.cpu().numpy()

        num_g = pos.shape[0]



        feat=torch.cat((gaussians._features_dc.reshape(num_g, -1),
                       gaussians._features_rest.reshape(num_g, -1),
                       gaussians._scaling.reshape(num_g, -1),
                       gaussians._rotation.reshape(num_g, -1),
                       gaussians._opacity.reshape(num_g, -1),), 1
                       )
        feat=feat[pos_idx]


        # RAHT transform
        res = haar3D(pos_voxlized, feat, depth)
        CT = res['CT']
        CT_q = torch.round(CT/Qstep)


        # zip compression
        np.savez_compressed(dir_path+'/compressed_gs.npz', pos_remain.cpu().numpy(), CT_q.cpu().numpy(), Qstep, depth)
        
        file_size=0
        import glob
        file_list = glob.glob(dir_path+r'/*')
        for file_path in file_list:
            file_size+=os.path.getsize(file_path)
        print('filezie: %fmb'%(file_size/1024/1024))

        return file_size/1024/1024



def decompress(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        dir_path=os.path.join(scene.model_path,
                    "point_cloud",
                    "iteration_" + str(iteration),
                    "compressed")

        data = np.load(dir_path+'/compressed_gs.npz')

        pos=data['arr_0']
        CT_q=data['arr_1']
        Qstep=data['arr_2']
        depth=data['arr_3']

        pos=torch.tensor(pos).cuda()
        CT_q=torch.tensor(CT_q).cuda()
        Qstep=torch.tensor(Qstep).cuda()


        # voxlize
        pos_voxlized=(pos-pos.min())/(pos.max()-pos.min())
        pos_voxlized=torch.round(pos_voxlized*(2**depth-1))
        pos_voxlized, pos_idx = np.unique(pos_voxlized.detach().cpu().numpy(), axis=0, return_index=True)     

        # inverse RAHT
        feat_rec = inv_haar3D(pos_voxlized, CT_q*(Qstep).item(), int(depth))

        num_g_voxlized=pos_voxlized.shape[0]


        gaussians._xyz = pos
        gaussians._features_dc = feat_rec[:, :3].reshape(num_g_voxlized, -1, 3).float()
        gaussians._features_rest = feat_rec[:, 3:48].reshape(num_g_voxlized, -1, 3)

        gaussians._scaling = feat_rec[:, 48:51].reshape(num_g_voxlized, 3).float()
        gaussians._rotation = feat_rec[:, 51:55].reshape(num_g_voxlized, 4).float()
        gaussians._opacity = feat_rec[:, 55:56].reshape(num_g_voxlized, 1).float()
        gaussians.active_sh_degree = dataset.sh_degree



        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")



        render_set(dataset.model_path, "test_compressed", iteration, scene.getTestCameras(), gaussians, pipeline, background)

    
        print()






if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)

    safe_state(False)


    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    
    # compress
    file_size = compress(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test) 
    # decompress and render       
    decompress(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    #evaluate
    val_ssim, val_psnr, val_lpips = evaluate([model.extract(args).model_path])

    print()
    print('ssim: ', val_ssim)
    print('psnr: ', val_psnr)
    print('lpips: ', val_lpips)
    print('file_size: %f mb'%file_size)


