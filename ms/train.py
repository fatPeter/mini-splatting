import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))


import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
from gaussian_renderer import render_imp, render_depth
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import numpy as np
from lpipsPyTorch import lpips
from utils.sh_utils import SH2RGB


def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask







def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(sh_degree=0)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
         
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_imp(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        simp_iteration1=args.simp_iteration1 
        if iteration<simp_iteration1:
            gaussians.update_learning_rate(iteration)
        else:
            gaussians.update_learning_rate(iteration-simp_iteration1+5000)


        if iteration % 1000 == 0 and iteration>simp_iteration1:
            gaussians.oneupSHdegree()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_imp, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)



            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                area_max = render_pkg["area_max"]
                mask_blur = torch.logical_or(mask_blur, area_max>(image.shape[1]*image.shape[2]/5000))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration % 5000!=0 and gaussians._xyz.shape[0]<args.num_max:  
                
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None


                    gaussians.densify_and_prune_split(opt.densify_grad_threshold, 
                                                    0.005, scene.cameras_extent, 
                                                    size_threshold, mask_blur)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    
                    
                if iteration%5000==0:

                    
                    out_pts_list=[]
                    gt_list=[]
                    views=scene.getTrainCameras()
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        render_depth_pkg = render_depth(view, gaussians, pipe, background)
                        out_pts = render_depth_pkg["out_pts"]
                        accum_alpha = render_depth_pkg["accum_alpha"]


                        prob=1-accum_alpha

                        prob = prob/prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()


                        factor=1/(image.shape[1]*image.shape[2]*len(views)/args.num_depth)


                        N_xyz=prob.shape[0]
                        num_sampled=int(N_xyz*factor)

                        indices = np.random.choice(N_xyz, size=num_sampled, 
                                                   p=prob,replace=False)
                        
                        out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                        gt = gt.permute(1,2,0).reshape(-1,3)

                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])       

    

                    out_pts_merged=torch.cat(out_pts_list)
                    gt_merged=torch.cat(gt_list)

                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device='cuda')
                    torch.cuda.empty_cache()
                    viewpoint_stack = scene.getTrainCameras().copy()



            if iteration == simp_iteration1:

                imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
                views=scene.getTrainCameras()
                for view in views:
                    # print(idx)
                    render_pkg = render_imp(view, gaussians, pipe, background)
                    accum_weights = render_pkg["accum_weights"]
                    area_proj = render_pkg["area_proj"]
                    area_max = render_pkg["area_max"]

                    accum_area_max = accum_area_max+area_max

                    if args.imp_metric=='outdoor':
                        mask_t=area_max!=0
                        temp=imp_score+accum_weights/area_proj
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score=imp_score+accum_weights
                
                imp_score[accum_area_max==0]=0
                prob = imp_score/imp_score.sum()
                prob = prob.cpu().numpy()


                factor=args.sampling_factor
                N_xyz=gaussians._xyz.shape[0]
                num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
                indices = np.random.choice(N_xyz, size=num_sampled, 
                                           p=prob, replace=False)
    
                mask = np.zeros(N_xyz, dtype=bool)
                mask[indices] = True


                # print(mask.sum(), mask.sum()/mask.shape[0])                 
                gaussians.prune_points(mask==False) 
                
                gaussians.max_sh_degree=dataset.sh_degree
                gaussians.reinitial_pts(gaussians._xyz, 
                                    SH2RGB(gaussians._features_dc+0)[:,0])
                
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()
                viewpoint_stack = scene.getTrainCameras().copy()


            if iteration == args.simp_iteration2:

                imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
                accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
                views=scene.getTrainCameras()
                for view in views:
                    gt = view.original_image[0:3, :, :]
                    
                    render_pkg = render_imp(view, gaussians, pipe, background)
                    accum_weights = render_pkg["accum_weights"]
                    area_proj = render_pkg["area_proj"]
                    area_max = render_pkg["area_max"]

                    accum_area_max = accum_area_max+area_max

                    if args.imp_metric=='outdoor':
                        mask_t=area_max!=0
                        temp=imp_score+accum_weights/area_proj
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score=imp_score+accum_weights
                    
                imp_score[accum_area_max==0]=0
                non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 
                                
                gaussians.prune_points(non_prune_mask==False)
                gaussians.training_setup(opt)
                torch.cuda.empty_cache()   

                

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")  

    print(gaussians._xyz.shape)

    return 









def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims = []
                lpipss = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    ssims.append(ssim(image, gt_image))
                    lpipss.append(lpips(image, gt_image, net_type='vgg'))                    


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras']) 

                ssims_test=torch.tensor(ssims).mean()
                lpipss_test=torch.tensor(lpipss).mean()

                print("\n[ITER {}] Evaluating {}: ".format(iteration, config['name']))
                print("  SSIM : {:>12.7f}".format(ssims_test.mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(psnr_test.mean(), ".5"))
                print("  LPIPS : {:>12.7f}".format(lpipss_test.mean(), ".5"))
                print("")
                
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--simp_iteration1", type=int, default = 15_000)
    parser.add_argument("--simp_iteration2", type=int, default = 20_000)
    parser.add_argument("--num_depth", type=int, default = 3_500_000)
    parser.add_argument("--num_max", type=int, default = 4_500_000)
    parser.add_argument("--sampling_factor", type=float, default = 0.5)

    parser.add_argument("--imp_metric", required=True, type=str, default = None)



    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)


    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)


    # All done
    print("\nTraining complete.")
