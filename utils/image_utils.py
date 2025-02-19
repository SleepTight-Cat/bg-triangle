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

import torch
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def edge_detection_rgb(image_tensor):
 

   
    sobel_kernel_x = torch.tensor(
        [[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]],
        dtype=torch.float32, device = image_tensor.device
    ).repeat(3, 1, 1, 1)  

    sobel_kernel_y = torch.tensor(
        [[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]],
        dtype=torch.float32, device = image_tensor.device
    ).repeat(3, 1, 1, 1)  

   
    edge_x = F.conv2d(image_tensor, sobel_kernel_x, padding=1, groups=3)
    edge_y = F.conv2d(image_tensor, sobel_kernel_y, padding=1, groups=3)

   
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze()

    return edges


def grayscale_dilation(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
  
    
    if image.dim() != 4 or image.size(1) != 1:
        image = image.unsqueeze(0).unsqueeze(0)

    
    dilated_image = F.max_pool2d(image, kernel_size, stride=1, padding=kernel_size // 2)
    return dilated_image


def median_filter_2d(depth_map, kernel_size=11):
    #  (H, W) -> (1, 1, H, W)
    depth_map = depth_map.unsqueeze(0).unsqueeze(0)
    
    
    padding = kernel_size // 2
    depth_map_padded = F.pad(depth_map, (padding, padding, padding, padding), mode='reflect')
    
  
    unfolded = depth_map_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    unfolded = unfolded.contiguous().view(*unfolded.shape[:4], -1)
    

    median_filtered, _ = unfolded.median(dim=-1)
    

    return median_filtered.squeeze(0).squeeze(0)

def gaussian_blur(depth_map, kernel_size=5, sigma=10):

    depth_data = depth_map

    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=depth_map.dtype)
    x = torch.exp(-0.5 * (x / sigma)**2)
    kernel_1d = x / x.sum()  
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  
    kernel_2d = kernel_2d.expand(1, 1, -1, -1) 

    # 应用高斯平滑
    depth_data = depth_data.unsqueeze(0).unsqueeze(0)  
    smoothed_depth = F.conv2d(depth_data, kernel_2d.cuda(), padding=kernel_size // 2)
    return smoothed_depth.squeeze()  

@torch.no_grad()
def calculate_normals(depth_map):

    depth_map = gaussian_blur(depth_map)


    H, W = depth_map.shape
    

    Gx = F.pad(depth_map[:, 1:] - depth_map[:, :-1], (0, 1, 0, 0))
    
    
    Gy = F.pad(depth_map[1:, :] - depth_map[:-1, :], (0, 0, 0, 1))
    

    normal_map = torch.stack((-Gx, -Gy, torch.ones_like(depth_map)), dim=-1)
    

    normal_map = F.normalize(normal_map, p=2, dim=-1)
    
    return normal_map  