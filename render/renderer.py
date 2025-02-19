import math
import time

from pytorch3d.renderer.cameras import FoVPerspectiveCameras
import torch
import torch.nn as nn

from model import (
    BMeshObject, BMeshRenderer,
    BPrimitiveBase,
    GaussianRenderer, GaussianObject,
    GaussianModel
)
from brasterizer import BMeshRasterizationSettings

class Renderer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bmesh_renderer = BMeshRenderer()
        self.gaussian_renderer = GaussianRenderer()

    def forward(self,
        bprimitive_object: BPrimitiveBase,
        camera,
        bg,
        server,
        num_segments_per_bprimitive_edge,
        scale_value,
        scale_gaussian,
        render_type
    ):
        # B-mesh rasterization
        bmesh_rasterizer_settings = BMeshRasterizationSettings(
            image_size=(camera.image_height, camera.image_width), bin_size=32, max_faces_per_bin=500000,
        )
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        bmesh_object = BMeshObject.from_bprimitive_version_1(bprimitive_object, num_segments_per_bprimitive_edge)
        boundary_points_3d, boundary_bp_id, debug_info = self.bmesh_renderer(
            camera, bprimitive_object, bmesh_object, bmesh_rasterizer_settings, render_type
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Display rasterization time in viser GUI
        rasterization_time = end_time - start_time
        if server is not None:
            if not hasattr(server, "rasterization_time_text"):
                server.rasterization_time_text = server.add_text("Rasterization Time", "0")
            server.rasterization_time_text.value = f"{rasterization_time:.4f} s"

        # Gaussian rendering
        boundary_params = {}
        boundary_params["blur_radius"] = math.exp(scale_value)
        boundary_params["boundary_points_3d"] = boundary_points_3d
        boundary_params["boundary_points_bprimitive_id"] = boundary_bp_id
        boundary_params["bprimitive_image"] = debug_info["bprimitive_image"]
        gaussian_bprimitive_id = debug_info["valid_bids"]
        if bprimitive_object.boundary_mode==1:
            gaussian_object = GaussianObject.from_bprimitive_version_1(debug_info["valid_bids"], debug_info["valid_uvws"], bprimitive_object)
            gaussian_object.scaling *= scale_gaussian
        elif bprimitive_object.boundary_mode==2:
            gaussian_object = GaussianObject.from_bprimitive_version_3(debug_info["valid_bids"], debug_info["valid_uvws"], bprimitive_object)
        else:
            gaussian_object, gaussian_bprimitive_id = GaussianObject.from_bprimitive_version_2(bprimitive_object)
            
        boundary_params["gaussian_bprimitive_id"] = gaussian_bprimitive_id

        

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        rendered_image, radii, means2D = self.gaussian_renderer(gaussian_object, camera, bg, image_size=bmesh_rasterizer_settings.image_size, boundary_params=boundary_params)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        debug_info['screen_space_gaussians'] = means2D
        debug_info['screen_space_gaussians_id'] = gaussian_bprimitive_id
        # Display Gaussian time in viser GUI
        render_time = end_time - start_time
        if server is not None:
            if not hasattr(server, "gaussian_time_text"):
                server.gaussian_time_text = server.add_text("Gaussian Time", "0")
            server.gaussian_time_text.value = f"{render_time:.4f} s"
            rendered_image = rendered_image.clamp(0, 1)
        return rendered_image, debug_info



def render_3dgs(camera, pc : GaussianModel,  bg_color : torch.Tensor):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass


    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree= pc.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False
        )



    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None


    scales = pc.get_scaling
    rotations = pc.get_rotation

    colors_precomp = None
    shs = pc.get_features


    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}