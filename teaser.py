import torch
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import numpy as np
from utils.sh_utils import SH2RGB
from scene.cameras import Camera
from PIL import Image, ImageDraw


def transformPoint4x4(p_orig, projmatrix):
    p_hom = torch.matmul(p_orig, projmatrix[:3])+projmatrix[3:4]
    return p_hom

def transformPoint4x3(p_orig, viewmatrix):
    p_view = torch.matmul(p_orig, viewmatrix[:3,:3])+viewmatrix[3:4, :3]
    return p_view

def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5

def draw_points_on_image(points, colors, image, size=1):

    image[image>1]=1
    image[image<0]=0
    image = Image.fromarray((image*255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    
    for point, color in zip(points, colors):
        x = point[0]
        y = point[1]
        
        r, g, b = color
        draw.ellipse((x-size,y-size,x+size,y+size), fill=(int(r), int(g), int(b)))
    
    image.show()
    return image





def proj_points(view, gaussians, pipeline, background):
   
    # tune cam_center and cam_rot to find a suitable camera pose
    cam_center=np.array([0.1, -0.5,  0.7])
    cam_rot=view.R

    # tune aabb box mask to extract foreground points
    aabb = np.array([[-1.6, -1.6, -1.6],
                        [1.6, 1.6, 1.6]])


    # render image
    view_reset = Camera(colmap_id=view.colmap_id, R=cam_rot, T=cam_center, 
                FoVx=view.FoVx, FoVy=view.FoVy, 
                image=view.original_image, gt_alpha_mask=None,
                image_name=view.image_name, uid=view.uid)

    render_pkg = render(view_reset, gaussians, pipeline, background)
    rendering = render_pkg["render"]


    # extract foreground
    aabb_min = aabb[0]
    aabb_max = aabb[1]

    xyz = gaussians._xyz+0
    rgb = SH2RGB(gaussians._features_dc+0)[:,0]

    aabb_mask = (xyz[:, 0] >= aabb_min[0]) & (xyz[:, 0] <= aabb_max[0]) & \
                (xyz[:, 1] >= aabb_min[1]) & (xyz[:, 1] <= aabb_max[1]) & \
                (xyz[:, 2] >= aabb_min[2]) & (xyz[:, 2] <= aabb_max[2])
    
    xyz = xyz[aabb_mask]
    rgb = rgb[aabb_mask]


    # perspective projection (modified from cuda code)
    full_proj_transform = view_reset.full_proj_transform
    p_hom = transformPoint4x4(xyz, full_proj_transform)
    p_w = 1.0 / (p_hom[:,3] + 0.0000001)
    p_proj = p_hom[:,:3]*p_w[:,None]

    world_view_transform = view_reset.world_view_transform
    p_view = transformPoint4x3(xyz, world_view_transform)
    mask = p_view[:,2].cpu().numpy()>0.2

    point_image = ndc2Pix(p_proj[:,0], rendering.shape[2]), \
        ndc2Pix(p_proj[:,1], rendering.shape[1])
    
    point_image=torch.cat((point_image[0][:,None], point_image[1][:,None]), -1)

    points = point_image.detach().cpu().numpy()[mask]
    colors = rgb.detach().cpu().numpy()[mask]


    # tune point size for better visualization 0.3, 0.3, 1.2
    image_proj = draw_points_on_image(points, np.zeros(colors.shape)+[0,0,255], rendering.permute(1,2,0).detach().cpu().numpy(), size=0.3)
    image_proj.save(r'./output.jpg')

    return 




def render_teaser(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        view = scene.getTrainCameras()[0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        proj_points(view, gaussians, pipeline, background)







if __name__ == "__main__":
    # Set up command line argument pars
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    print("Rendering source_path " + args.source_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_teaser(model.extract(args), args.iteration, pipeline.extract(args))