import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)*0.5
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)*0.5
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def generate_random_unit_triangles(N):
   
    base_triangle = torch.tensor([
        [-0.5, -math.sqrt(3) / 6, 0.0],
        [0.5, -math.sqrt(3) / 6, 0.0],
        [0.0, math.sqrt(3) / 3, 0.0]
    ])  # shape: (3, 3)

    
    theta_x = torch.rand(N) * 2 * math.pi
    theta_y = torch.rand(N) * 2 * math.pi
    theta_z = torch.rand(N) * 2 * math.pi

   
    R_x = torch.stack([
        torch.ones(N), torch.zeros(N), torch.zeros(N),
        torch.zeros(N), torch.cos(theta_x), -torch.sin(theta_x),
        torch.zeros(N), torch.sin(theta_x), torch.cos(theta_x)
    ], dim=1).view(N, 3, 3)

   
    R_y = torch.stack([
        torch.cos(theta_y), torch.zeros(N), torch.sin(theta_y),
        torch.zeros(N), torch.ones(N), torch.zeros(N),
        -torch.sin(theta_y), torch.zeros(N), torch.cos(theta_y)
    ], dim=1).view(N, 3, 3)


    R_z = torch.stack([
        torch.cos(theta_z), -torch.sin(theta_z), torch.zeros(N),
        torch.sin(theta_z), torch.cos(theta_z), torch.zeros(N),
        torch.zeros(N), torch.zeros(N), torch.ones(N)
    ], dim=1).view(N, 3, 3)

    
    rotation_matrices = torch.bmm(R_z, torch.bmm(R_y, R_x))  # shape: (N, 3, 3)

    
    base_triangle_expanded = base_triangle.unsqueeze(0).expand(N, -1, -1)  # shape: (N, 3, 3)
    rotated_triangles = torch.bmm(base_triangle_expanded, rotation_matrices.transpose(1, 2))

    return rotated_triangles