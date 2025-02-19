/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>
#include <tuple>

// Arguments are the same as RasterizeMeshesCoarse from
// rasterize_meshes/rasterize_meshes.h

std::tuple<at::Tensor, at::Tensor> RasterizeMeshesCoarseCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& mesh_to_face_first_idx,
    const torch::Tensor& num_faces_per_mesh,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin);



std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
RasterizeMeshesFineCuda(
    const at::Tensor& face_verts,
    const at::Tensor& bin_faces,
    const at::Tensor& elems_per_bin,
    const at::Tensor& clipped_faces_neighbor_idx,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int faces_per_pixel,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const bool cull_backfaces);


std::tuple<torch::Tensor, torch::Tensor> DetectOutlineVerticesCuda(
    const torch::Tensor& face_verts,
    const torch::Tensor& bprimitive_image
);


std::tuple<torch::Tensor, torch::Tensor> DetectOutlinePixelsCuda(
    const torch::Tensor& bprimitive_image
);