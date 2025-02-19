/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <math.h>
#include <tuple>
#include "bitmask.cuh"
#include "float_math.cuh"
#include "geometry_utils.cuh" // For kEpsilon -- gross
#include "rasterization_utils.h"


namespace {
// A structure for holding details about a pixel.
struct Pixel {
  float z; 
  int64_t idx; // idx of face
  float dist; // abs distance of pixel to face
  float3 bary;
};

__device__ bool operator<(const Pixel& a, const Pixel& b) {
  return a.z < b.z || (a.z == b.z && a.idx < b.idx);
}

// Get the xyz coordinates of the three vertices for the face given by the
// index face_idx into face_verts.
__device__ thrust::tuple<float3, float3, float3> GetSingleFaceVerts(
    const float* face_verts,
    int face_idx) {
  const float x0 = face_verts[face_idx * 9 + 0];
  const float y0 = face_verts[face_idx * 9 + 1];
  const float z0 = face_verts[face_idx * 9 + 2];
  const float x1 = face_verts[face_idx * 9 + 3];
  const float y1 = face_verts[face_idx * 9 + 4];
  const float z1 = face_verts[face_idx * 9 + 5];
  const float x2 = face_verts[face_idx * 9 + 6];
  const float y2 = face_verts[face_idx * 9 + 7];
  const float z2 = face_verts[face_idx * 9 + 8];

  const float3 v0xyz = make_float3(x0, y0, z0);
  const float3 v1xyz = make_float3(x1, y1, z1);
  const float3 v2xyz = make_float3(x2, y2, z2);

  return thrust::make_tuple(v0xyz, v1xyz, v2xyz);
}

// Get the min/max x/y/z values for the face given by vertices v0, v1, v2.
__device__ thrust::tuple<float2, float2, float2>
GetFaceBoundingBox(float3 v0, float3 v1, float3 v2) {
  const float xmin = FloatMin3(v0.x, v1.x, v2.x);
  const float ymin = FloatMin3(v0.y, v1.y, v2.y);
  const float zmin = FloatMin3(v0.z, v1.z, v2.z);
  const float xmax = FloatMax3(v0.x, v1.x, v2.x);
  const float ymax = FloatMax3(v0.y, v1.y, v2.y);
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);

  return thrust::make_tuple(
      make_float2(xmin, xmax),
      make_float2(ymin, ymax),
      make_float2(zmin, zmax));
}

// Check if the point (px, py) lies outside the face bounding box face_bbox.
// Return true if the point is outside.
__device__ bool CheckPointOutsideBoundingBox(
    float3 v0,
    float3 v1,
    float3 v2,
    float blur_radius,
    float2 pxy) {
  const auto bbox = GetFaceBoundingBox(v0, v1, v2);
  const float2 xlims = thrust::get<0>(bbox);
  const float2 ylims = thrust::get<1>(bbox);
  const float2 zlims = thrust::get<2>(bbox);

  const float x_min = xlims.x - blur_radius;
  const float y_min = ylims.x - blur_radius;
  const float x_max = xlims.y + blur_radius;
  const float y_max = ylims.y + blur_radius;

  // Faces with at least one vertex behind the camera won't render correctly
  // and should be removed or clipped before calling the rasterizer
  const bool z_invalid = zlims.x < kEpsilon;

  // Check if the current point is oustside the triangle bounding box.
  return (
      pxy.x > x_max || pxy.x < x_min || pxy.y > y_max || pxy.y < y_min ||
      z_invalid);
}

// This function checks if a pixel given by xy location pxy lies within the
// face with index face_idx in face_verts. One of the inputs is a list (q)
// which contains Pixel structs with the indices of the faces which intersect
// with this pixel sorted by closest z distance. If the point pxy lies in the
// face, the list (q) is updated and re-orderered in place. In addition
// the auxiliary variables q_size, q_max_z and q_max_idx are also modified.
// This code is shared between RasterizeMeshesNaiveCudaKernel and
// RasterizeMeshesFineCudaKernel.
template <typename FaceQ>
__device__ void CheckPixelInsideFace(
    const float* face_verts, // (F, 3, 3)
    const int64_t* clipped_faces_neighbor_idx, // (F,)
    const int face_idx,
    int& q_size,
    float& q_max_z,
    int& q_max_idx,
    FaceQ& q,
    const float blur_radius,
    const float2 pxy, // Coordinates of the pixel
    const int K,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const bool cull_backfaces) {
  const auto v012 = GetSingleFaceVerts(face_verts, face_idx);
  const float3 v0 = thrust::get<0>(v012);
  const float3 v1 = thrust::get<1>(v012);
  const float3 v2 = thrust::get<2>(v012);

  // Only need xy for barycentric coordinates and distance calculations.
  const float2 v0xy = make_float2(v0.x, v0.y);
  const float2 v1xy = make_float2(v1.x, v1.y);
  const float2 v2xy = make_float2(v2.x, v2.y);

  // Perform checks and skip if:
  // 1. the face is behind the camera
  // 2. the face is facing away from the camera
  // 3. the face has very small face area
  // 4. the pixel is outside the face bbox
  const float zmax = FloatMax3(v0.z, v1.z, v2.z);
  const bool outside_bbox = CheckPointOutsideBoundingBox(
      v0, v1, v2, sqrt(blur_radius), pxy); // use sqrt of blur for bbox
  const float face_area = EdgeFunctionForward(v0xy, v1xy, v2xy);
  // Check if the face is visible to the camera.
  const bool back_face = face_area < 0.0;
  const bool zero_face_area =
      (face_area <= kEpsilon && face_area >= -1.0f * kEpsilon);

  if (zmax < 0 || (cull_backfaces && back_face) || outside_bbox ||
      zero_face_area) {
    return;
  }

  // Calculate barycentric coords and euclidean dist to triangle.
  const float3 p_bary0 = BarycentricCoordsForward(pxy, v0xy, v1xy, v2xy);
  const float3 p_bary = !perspective_correct
      ? p_bary0
      : BarycentricPerspectiveCorrectionForward(p_bary0, v0.z, v1.z, v2.z);
  const float3 p_bary_clip =
      !clip_barycentric_coords ? p_bary : BarycentricClipForward(p_bary);

  const float pz =
      p_bary_clip.x * v0.z + p_bary_clip.y * v1.z + p_bary_clip.z * v2.z;

  if (pz < 0) {
    return; // Face is behind the image plane.
  }

  // Get abs squared distance
  //const float dist = PointTriangleDistanceForward(pxy, v0xy, v1xy, v2xy);
  const float dist = 0; // No need dist for us !!!!
  // Use the unclipped bary coordinates to determine if the point is inside the
  // face.
  const bool inside = p_bary.x > 0.0f && p_bary.y > 0.0f && p_bary.z > 0.0f;
  const float signed_dist = inside ? -dist : dist;
  // Check if pixel is outside blur region
  if (!inside && dist >= blur_radius) {
    return;
  }

  // Handle the case where a face (f) partially behind the image plane is
  // clipped to a quadrilateral and then split into two faces (t1, t2). In this
  // case we:
  // 1. Find the index of the neighboring face (e.g. for t1 need index of t2)
  // 2. Check if the neighboring face (t2) is already in the top K faces
  // 3. If yes, compare the distance of the pixel to t1 with the distance to t2.
  // 4. If dist_t1 < dist_t2, overwrite the values for t2 in the top K faces.
  const int neighbor_idx = clipped_faces_neighbor_idx[face_idx];
  int neighbor_idx_top_k = -1;

  // Check if neighboring face is already in the top K.
  // -1 is the fill value in clipped_faces_neighbor_idx
  if (neighbor_idx != -1) {
    // Only need to loop until q_size.
    for (int i = 0; i < q_size; i++) {
      if (q[i].idx == neighbor_idx) {
        neighbor_idx_top_k = i;
        break;
      }
    }
  }
  // If neighbor idx is not -1 then it is in the top K struct.
  if (neighbor_idx_top_k != -1) {
    // If dist of current face is less than neighbor then overwrite the
    // neighbor face values in the top K struct.
    float neighbor_dist = abs(q[neighbor_idx_top_k].dist);
    if (dist < neighbor_dist) {
      // Overwrite the neighbor face values
      q[neighbor_idx_top_k] = {pz, face_idx, signed_dist, p_bary_clip};

      // If pz > q_max then overwrite the max values and index of the max.
      // q_size stays the same.
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = neighbor_idx_top_k;
      }
    }
  } else {
    // Handle as a normal face
    if (q_size < K) {
      // Just insert it.
      q[q_size] = {pz, face_idx, signed_dist, p_bary_clip};
      if (pz > q_max_z) {
        q_max_z = pz;
        q_max_idx = q_size;
      }
      q_size++;
    } else if (pz < q_max_z) {
      // Overwrite the old max, and find the new max.
      q[q_max_idx] = {pz, face_idx, signed_dist, p_bary_clip};
      q_max_z = pz;
      for (int i = 0; i < K; i++) {
        if (q[i].z > q_max_z) {
          q_max_z = q[i].z;
          q_max_idx = i;
        }
      }
    }
  }
}

} // namespace


__global__ void TriangleBoundingBoxKernel(
    const float* face_verts, // (F, 3, 3)
    const int F,
    const float blur_radius,
    float* bboxes, // (4, F)
    bool* skip_face) { // (F,)
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  const float sqrt_radius = sqrt(blur_radius);
  for (int f = tid; f < F; f += num_threads) {
    const float v0x = face_verts[f * 9 + 0 * 3 + 0];
    const float v0y = face_verts[f * 9 + 0 * 3 + 1];
    const float v0z = face_verts[f * 9 + 0 * 3 + 2];
    const float v1x = face_verts[f * 9 + 1 * 3 + 0];
    const float v1y = face_verts[f * 9 + 1 * 3 + 1];
    const float v1z = face_verts[f * 9 + 1 * 3 + 2];
    const float v2x = face_verts[f * 9 + 2 * 3 + 0];
    const float v2y = face_verts[f * 9 + 2 * 3 + 1];
    const float v2z = face_verts[f * 9 + 2 * 3 + 2];
    const float xmin = FloatMin3(v0x, v1x, v2x) - sqrt_radius;
    const float xmax = FloatMax3(v0x, v1x, v2x) + sqrt_radius;
    const float ymin = FloatMin3(v0y, v1y, v2y) - sqrt_radius;
    const float ymax = FloatMax3(v0y, v1y, v2y) + sqrt_radius;
    const float zmin = FloatMin3(v0z, v1z, v2z);
    const bool skip = zmin < kEpsilon;
    bboxes[0 * F + f] = xmin;
    bboxes[1 * F + f] = xmax;
    bboxes[2 * F + f] = ymin;
    bboxes[3 * F + f] = ymax;
    skip_face[f] = skip;
  }
}

__global__ void PointBoundingBoxKernel(
    const float* points, // (P, 3)
    const float* radius, // (P,)
    const int P,
    float* bboxes, // (4, P)
    bool* skip_points) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * gridDim.x;
  for (int p = tid; p < P; p += num_threads) {
    const float x = points[p * 3 + 0];
    const float y = points[p * 3 + 1];
    const float z = points[p * 3 + 2];
    const float r = radius[p];
    // TODO: change to kEpsilon to match triangles?
    const bool skip = z < 0;
    bboxes[0 * P + p] = x - r;
    bboxes[1 * P + p] = x + r;
    bboxes[2 * P + p] = y - r;
    bboxes[3 * P + p] = y + r;
    skip_points[p] = skip;
  }
}

// ****************************************************************************
// *                            Coarse RASTERIZATION                            *
// ****************************************************************************

__global__ void RasterizeCoarseCudaKernel(
    const float* bboxes, // (4, E) (xmin, xmax, ymin, ymax)
    const bool* should_skip, // (E,)
    const int64_t* elem_first_idxs,
    const int64_t* elems_per_batch,
    const int N,
    const int E,
    const int H,
    const int W,
    const int bin_size,
    const int chunk_size,
    const int max_elem_per_bin,
    int* elems_per_bin,
    int* bin_elems) {
  extern __shared__ char sbuf[];
  const int M = max_elem_per_bin;
  // Integer divide round up
  const int num_bins_x = 1 + (W - 1) / bin_size;
  const int num_bins_y = 1 + (H - 1) / bin_size;

  // NDC range depends on the ratio of W/H
  // The shorter side from (H, W) is given an NDC range of 2.0 and
  // the other side is scaled by the ratio of H:W.
  const float NDC_x_half_range = NonSquareNdcRange(W, H) / 2.0f;
  const float NDC_y_half_range = NonSquareNdcRange(H, W) / 2.0f;

  // Size of half a pixel in NDC units is the NDC half range
  // divided by the corresponding image dimension
  const float half_pix_x = NDC_x_half_range / W;
  const float half_pix_y = NDC_y_half_range / H;

  // This is a boolean array of shape (num_bins_y, num_bins_x, chunk_size)
  // stored in shared memory that will track whether each elem in the chunk
  // falls into each bin of the image.
  BitMask binmask((unsigned int*)sbuf, num_bins_y, num_bins_x, chunk_size);

  // Have each block handle a chunk of elements
  const int chunks_per_batch = 1 + (E - 1) / chunk_size;
  const int num_chunks = N * chunks_per_batch;

  for (int chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
    const int batch_idx = chunk / chunks_per_batch; // batch index
    const int chunk_idx = chunk % chunks_per_batch;
    const int elem_chunk_start_idx = chunk_idx * chunk_size;

    binmask.block_clear();
    const int64_t elem_start_idx = elem_first_idxs[batch_idx];
    const int64_t elem_stop_idx = elem_start_idx + elems_per_batch[batch_idx];

    // Have each thread handle a different face within the chunk
    for (int e = threadIdx.x; e < chunk_size; e += blockDim.x) {
      const int e_idx = elem_chunk_start_idx + e;

      // Check that we are still within the same element of the batch
      if (e_idx >= elem_stop_idx || e_idx < elem_start_idx) {
        continue;
      }

      if (should_skip[e_idx]) {
        continue;
      }
      const float xmin = bboxes[0 * E + e_idx];
      const float xmax = bboxes[1 * E + e_idx];
      const float ymin = bboxes[2 * E + e_idx];
      const float ymax = bboxes[3 * E + e_idx];

      // Brute-force search over all bins; TODO(T54294966) something smarter.
      for (int by = 0; by < num_bins_y; ++by) {
        // Y coordinate of the top and bottom of the bin.
        // PixToNdc gives the location of the center of each pixel, so we
        // need to add/subtract a half pixel to get the true extent of the bin.
        // Reverse ordering of Y axis so that +Y is upwards in the image.
        const float bin_y_min =
            PixToNonSquareNdc(by * bin_size, H, W) - half_pix_y;
        const float bin_y_max =
            PixToNonSquareNdc((by + 1) * bin_size - 1, H, W) + half_pix_y;
        const bool y_overlap = (ymin <= bin_y_max) && (bin_y_min < ymax);

        for (int bx = 0; bx < num_bins_x; ++bx) {
          // X coordinate of the left and right of the bin.
          // Reverse ordering of x axis so that +X is left.
          const float bin_x_max =
              PixToNonSquareNdc((bx + 1) * bin_size - 1, W, H) + half_pix_x;
          const float bin_x_min =
              PixToNonSquareNdc(bx * bin_size, W, H) - half_pix_x;

          const bool x_overlap = (xmin <= bin_x_max) && (bin_x_min < xmax);
          if (y_overlap && x_overlap) {
            binmask.set(by, bx, e);
          }
        }
      }
    }
    __syncthreads();
    // Now we have processed every elem in the current chunk. We need to
    // count the number of elems in each bin so we can write the indices
    // out to global memory. We have each thread handle a different bin.
    for (int byx = threadIdx.x; byx < num_bins_y * num_bins_x;
         byx += blockDim.x) {
      const int by = byx / num_bins_x;
      const int bx = byx % num_bins_x;
      const int count = binmask.count(by, bx);
      const int elems_per_bin_idx =
          batch_idx * num_bins_y * num_bins_x + by * num_bins_x + bx;

      // This atomically increments the (global) number of elems found
      // in the current bin, and gets the previous value of the counter;
      // this effectively allocates space in the bin_faces array for the
      // elems in the current chunk that fall into this bin.
      const int start = atomicAdd(elems_per_bin + elems_per_bin_idx, count);
      if (start + count > M) {
        // The number of elems in this bin is so big that they won't fit.
        // We print a warning using CUDA's printf. This may be invisible
        // to notebook users, but apparent to others. It would be nice to
        // also have a Python-friendly warning, but it is not obvious
        // how to do this without slowing down the normal case.
        const char* warning =
            "Error:Bin size was too small in the coarse rasterization phase. "
            "This caused an overflow, meaning output may be incomplete. "
            "To solve, "
            "try increasing max_faces_per_bin / max_points_per_bin, "
            "decreasing bin_size, "
            "or setting bin_size to 0 to use the naive rasterization.\n";
        printf(warning);
        continue;
      }

      // Now loop over the binmask and write the active bits for this bin
      // out to bin_faces.
      int next_idx = batch_idx * num_bins_y * num_bins_x * M +
          by * num_bins_x * M + bx * M + start;
      for (int e = 0; e < chunk_size; ++e) {
        if (binmask.get(by, bx, e)) {
          // TODO(T54296346) find the correct method for handling errors in
          // CUDA. Throw an error if num_faces_per_bin > max_faces_per_bin.
          // Either decrease bin size or increase max_faces_per_bin
          bin_elems[next_idx] = elem_chunk_start_idx + e;
          next_idx++;
        }
      }
    }
    __syncthreads();
  }
}

std::tuple<at::Tensor, at::Tensor> RasterizeCoarseCuda(
    const at::Tensor& bboxes,
    const at::Tensor& should_skip,
    const at::Tensor& elem_first_idxs,
    const at::Tensor& elems_per_batch,
    const std::tuple<int, int> image_size,
    const int bin_size,
    const int max_elems_per_bin) {
  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(bboxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  const int E = bboxes.size(1);
  const int N = elems_per_batch.size(0);
  const int M = max_elems_per_bin;

  // Integer divide round up
  const int num_bins_y = 1 + (H - 1) / bin_size;
  const int num_bins_x = 1 + (W - 1) / bin_size;

  if (num_bins_y >= kMaxItemsPerBin || num_bins_x >= kMaxItemsPerBin) {
    std::stringstream ss;
    ss << "In RasterizeCoarseCuda got num_bins_y: " << num_bins_y
       << ", num_bins_x: " << num_bins_x << ", " << "; that's too many!" <<", "<<kMaxItemsPerBin;
    AT_ERROR(ss.str());
  }
  auto opts = elems_per_batch.options().dtype(at::kInt);
  at::Tensor elems_per_bin = at::zeros({N, num_bins_y, num_bins_x}, opts);
  at::Tensor bin_elems = at::full({N, num_bins_y, num_bins_x, M}, -1, opts);

  if (bin_elems.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(bin_elems,elems_per_bin);
  }

  const int chunk_size = 512;
  const size_t shared_size = num_bins_y * num_bins_x * chunk_size / 8;
  const size_t blocks = 64;
  const size_t threads = 512;

  RasterizeCoarseCudaKernel<<<blocks, threads, shared_size, stream>>>(
      bboxes.contiguous().data_ptr<float>(),
      should_skip.contiguous().data_ptr<bool>(),
      elem_first_idxs.contiguous().data_ptr<int64_t>(),
      elems_per_batch.contiguous().data_ptr<int64_t>(),
      N,
      E,
      H,
      W,
      bin_size,
      chunk_size,
      M,
      elems_per_bin.data_ptr<int32_t>(),
      bin_elems.data_ptr<int32_t>());

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(bin_elems,elems_per_bin);
}

std::tuple<at::Tensor, at::Tensor> RasterizeMeshesCoarseCuda(
    const at::Tensor& face_verts,
    const at::Tensor& mesh_to_face_first_idx,
    const at::Tensor& num_faces_per_mesh,
    const std::tuple<int, int> image_size,
    const float blur_radius,
    const int bin_size,
    const int max_faces_per_bin) {
  TORCH_CHECK(
      face_verts.ndimension() == 3 && face_verts.size(1) == 3 &&
          face_verts.size(2) == 3,
      "face_verts must have dimensions (num_faces, 3, 3)");

  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      mesh_to_face_first_idx_t{
          mesh_to_face_first_idx, "mesh_to_face_first_idx", 2},
      num_faces_per_mesh_t{num_faces_per_mesh, "num_faces_per_mesh", 3};
  at::CheckedFrom c = "RasterizeMeshesCoarseCuda";
  at::checkAllSameGPU(
      c, {face_verts_t, mesh_to_face_first_idx_t, num_faces_per_mesh_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Allocate tensors for bboxes and should_skip
  const int F = face_verts.size(0);
  auto float_opts = face_verts.options().dtype(at::kFloat);
  auto bool_opts = face_verts.options().dtype(at::kBool);
  at::Tensor bboxes = at::empty({4, F}, float_opts);
  at::Tensor should_skip = at::empty({F}, bool_opts);

  // Launch kernel to compute triangle bboxes
  const size_t blocks = 128;
  const size_t threads = 256;
  TriangleBoundingBoxKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      F,
      blur_radius,
      bboxes.contiguous().data_ptr<float>(),
      should_skip.contiguous().data_ptr<bool>());
  AT_CUDA_CHECK(cudaGetLastError());

  return RasterizeCoarseCuda(
      bboxes,
      should_skip,
      mesh_to_face_first_idx,
      num_faces_per_mesh,
      image_size,
      bin_size,
      max_faces_per_bin);
}

// ****************************************************************************
// *                            FINE RASTERIZATION                            *
// ****************************************************************************
__global__ void RasterizeMeshesFineCudaKernel(
    const float* face_verts, // (F, 3, 3)
    const int32_t* bin_faces, // (N, BH, BW, T)
    const int32_t* elems_per_bin, // (N, BH, BW)
    const int64_t* clipped_faces_neighbor_idx, // (F,)
    const float blur_radius,
    const int bin_size,
    const bool perspective_correct,
    const bool clip_barycentric_coords,
    const bool cull_backfaces,
    const int N,
    const int BH,
    const int BW,
    const int M,
    const int H,
    const int W,
    const int K,
    int64_t* face_idxs, // (N, H, W, K)
    float* zbuf, // (N, H, W, K)
    float* pix_dists, // (N, H, W, K)
    float* bary // (N, H, W, K, 3)
) {
  // This can be more than H * W if H or W are not divisible by bin_size.
  int num_pixels = N * BH * BW * bin_size * bin_size;
  int num_threads = gridDim.x * blockDim.x;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pid = tid; pid < num_pixels; pid += num_threads) {
    // Convert linear index into bin and pixel indices. We make the within
    // block pixel ids move the fastest, so that adjacent threads will fall
    // into the same bin; this should give them coalesced memory reads when
    // they read from faces and bin_faces.
    int i = pid;
    const int n = i / (BH * BW * bin_size * bin_size);
    i %= BH * BW * bin_size * bin_size;
    // bin index y
    const int by = i / (BW * bin_size * bin_size);
    i %= BW * bin_size * bin_size;
    // bin index y
    const int bx = i / (bin_size * bin_size);
    // pixel within the bin
    i %= bin_size * bin_size;

    // Pixel x, y indices
    const int yi = i / bin_size + by * bin_size;
    const int xi = i % bin_size + bx * bin_size;

    if (yi >= H || xi >= W)
      continue;

    const float xf = PixToNonSquareNdc(xi, W, H);
    const float yf = PixToNonSquareNdc(yi, H, W);

    const float2 pxy = make_float2(xf, yf);

    // This part looks like the naive rasterization kernel, except we use
    // bin_faces to only look at a subset of faces already known to fall
    // in this bin. TODO abstract out this logic into some data structure
    // that is shared by both kernels?
    Pixel q[kMaxPointsPerPixel];
    int q_size = 0;
    float q_max_z = -1000;
    int q_max_idx = -1;


    int num_faces_in_bin = elems_per_bin[n * BH * BW  + by * BW  + bx ];

    for (int m = 0; m < num_faces_in_bin; m+=1) {
      const int f = bin_faces[n * BH * BW * M + by * BW * M + bx * M + m];
      if (f < 0) {
        continue; // bin_faces uses -1 as a sentinal value.
      }
      // Check if the pixel pxy is inside the face bounding box and if it is,
      // update q, q_size, q_max_z and q_max_idx in place.
      
      
      CheckPixelInsideFace(
          face_verts,
          clipped_faces_neighbor_idx,
          f,
          q_size,
          q_max_z,
          q_max_idx,
          q,
          blur_radius,
          pxy,
          K,
          perspective_correct,
          clip_barycentric_coords,
          cull_backfaces);
     
    }

    // Now we've looked at all the faces for this bin, so we can write
    // output for the current pixel.
    // TODO: make sorting an option as only top k is needed, not sorted values.
    BubbleSort(q, q_size);

    // Reverse ordering of the X and Y axis so that
    // in the image +Y is pointing up and +X is pointing left.
    const int yidx = H - 1 - yi;
    const int xidx = W - 1 - xi;

    const int pix_idx = n * H * W * K + yidx * W * K + xidx * K;
    for (int k = 0; k < q_size; k++) {
      face_idxs[pix_idx + k] = q[k].idx;
      zbuf[pix_idx + k] = q[k].z;
      pix_dists[pix_idx + k] = q[k].dist;
      bary[(pix_idx + k) * 3 + 0] = q[k].bary.x;
      bary[(pix_idx + k) * 3 + 1] = q[k].bary.y;
      bary[(pix_idx + k) * 3 + 2] = q[k].bary.z;
    }
  }
}

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
    const bool cull_backfaces) {
  TORCH_CHECK(
      face_verts.ndimension() == 3 && face_verts.size(1) == 3 &&
          face_verts.size(2) == 3,
      "face_verts must have dimensions (num_faces, 3, 3)");
  TORCH_CHECK(bin_faces.ndimension() == 4, "bin_faces must have 4 dimensions");
  TORCH_CHECK(
      clipped_faces_neighbor_idx.size(0) == face_verts.size(0),
      "clipped_faces_neighbor_idx must have the same first dimension as face_verts");

  // Check inputs are on the same device
  at::TensorArg face_verts_t{face_verts, "face_verts", 1},
      bin_faces_t{bin_faces, "bin_faces", 2},
      clipped_faces_neighbor_idx_t{
          clipped_faces_neighbor_idx, "clipped_faces_neighbor_idx", 3};
  at::CheckedFrom c = "RasterizeMeshesFineCuda";
  at::checkAllSameGPU(
      c, {face_verts_t, bin_faces_t, clipped_faces_neighbor_idx_t});

  // Set the device for the kernel launch based on the device of the input
  at::cuda::CUDAGuard device_guard(face_verts.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // bin_faces shape (N, BH, BW, M)
  const int N = bin_faces.size(0);
  const int BH = bin_faces.size(1);
  const int BW = bin_faces.size(2);
  const int M = bin_faces.size(3);
  const int K = faces_per_pixel;

  const int H = std::get<0>(image_size);
  const int W = std::get<1>(image_size);

  if (K > kMaxPointsPerPixel) {
    AT_ERROR("Must have num_closest <= 150");
  }
  auto long_opts = bin_faces.options().dtype(at::kLong);
  auto float_opts = face_verts.options().dtype(at::kFloat);

  at::Tensor face_idxs = at::full({N, H, W, K}, -1, long_opts);
  at::Tensor zbuf = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor pix_dists = at::full({N, H, W, K}, -1, float_opts);
  at::Tensor bary = at::full({N, H, W, K, 3}, -1, float_opts);

  if (face_idxs.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
  }

  
  const size_t threads = 256;
  const size_t blocks = (H*W)/threads + 1;

  
  RasterizeMeshesFineCudaKernel<<<blocks, threads, 0, stream>>>(
      face_verts.contiguous().data_ptr<float>(),
      bin_faces.contiguous().data_ptr<int32_t>(),
      elems_per_bin.contiguous().data_ptr<int32_t>(),
      clipped_faces_neighbor_idx.contiguous().data_ptr<int64_t>(),
      blur_radius,
      bin_size,
      perspective_correct,
      clip_barycentric_coords,
      cull_backfaces,
      N,
      BH,
      BW,
      M,
      H,
      W,
      K,
      face_idxs.data_ptr<int64_t>(),
      zbuf.data_ptr<float>(),
      pix_dists.data_ptr<float>(),
      bary.data_ptr<float>());
  
  return std::make_tuple(face_idxs, zbuf, bary, pix_dists);
}



// ****************************************************************************
// *                            DetectOutlines                            *
// ****************************************************************************


__global__ void DetectOutlineVerticesCudaKernel(
    const float* face_verts,  // (F, 3, 3)
    const int32_t* bprimitive_image,        // (H, W)
    const int H, 
    const int W,
    const int F,
    bool* outline_vertices,  // (F, 3)
    unsigned char* outline_image  // (H, W)
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vertices = F * 3;
    
    for (int vid = tid; vid < num_vertices; vid += blockDim.x * gridDim.x) {
        const int f = vid / 3;  // face index
        const int v = vid % 3;  // vertex index within the face

        const float x = face_verts[f * 9 + v * 3];
        const float y = face_verts[f * 9 + v * 3 + 1];
        
        // Convert NDC coordinates to pixel coordinates
        const int px = NonSquareNdcToPix(-x, W, H);
        const int py = NonSquareNdcToPix(-y, H, W);  // Note the negation of y

        
        bool is_outline = false;
        
        // Check neighboring pixels
        for (int dy = -1; dy <= 1 && !is_outline; ++dy) {
            for (int dx = -1; dx <= 1 && !is_outline; ++dx) {
                if (dx == 0 && dy == 0) continue;  // Skip the center pixel
                
                int nx = px + dx;
                int ny = py + dy;
                
                // Check if neighbor is within image bounds
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    // If any neighboring pixel is background, this is an outline vertex
                    if (bprimitive_image[ny * W + nx] != bprimitive_image[py * W + px]) {
                        is_outline = true;
                    }
                }
            }
        }
        
        outline_vertices[vid] = is_outline;

        // Mark the outline vertex on the image
        if (is_outline) {
            for (int dy = -1; dy <= 1 ; ++dy) 
              for (int dx = -1; dx <= 1; ++dx)
              {
                int nx = px + dx;
                int ny = py + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) 
                    outline_image[ny * W + nx] = 255;  // Set to white
              }
                
        }
    }
}

// Host function to launch the kernel
std::tuple<at::Tensor, at::Tensor> DetectOutlineVerticesCuda(
    const at::Tensor& face_verts,
    const at::Tensor& bprimitive_image
) {
    const int H = bprimitive_image.size(0);
    const int W = bprimitive_image.size(1);
    const int F = face_verts.size(0);

    auto outline_vertices = at::zeros({F, 3}, face_verts.options().dtype(at::kLong));
    auto outline_image = at::zeros({H, W}, face_verts.options().dtype(at::kByte));

    const int threads = 1024;
    const int blocks = (F * 3 + threads - 1) / threads;

    DetectOutlineVerticesCudaKernel<<<blocks, threads>>>(
        face_verts.data_ptr<float>(),
        bprimitive_image.data_ptr<int32_t>(),
        H, W, F,
        outline_vertices.data_ptr<bool>(),
        outline_image.data_ptr<unsigned char>()
    );

    return std::make_tuple(outline_vertices, outline_image);
}


// CUDA kernel to detect outline pixels
__global__ void DetectOutlinePixelsKernel(
    const int32_t* bprimitive_image,
    int* outline_mask,
    int* outline_coords,
    int * coord_count,
    int W,
    int H
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < W && y < H) {
        int32_t current = bprimitive_image[y * W + x];
        bool is_outline = false;

        // Check neighboring pixels
        for (int dy = -1; dy <= 1 && !is_outline; ++dy) {
            for (int dx = -1; dx <= 1 && !is_outline; ++dx) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    if (bprimitive_image[ny * W + nx] != current && current!=-1) {
                        is_outline = true;
                    }
                }
            }
        }

        if (is_outline) {
            outline_mask[y * W + x] = 255;
            int idx = atomicAdd(coord_count, 1);
            outline_coords[idx * 2] = x;
            outline_coords[idx * 2 + 1] = y;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> DetectOutlinePixelsCuda(
    const at::Tensor& bprimitive_image
){
    const int H = bprimitive_image.size(0);
    const int W = bprimitive_image.size(1);

    auto outline_mask = at::zeros({H, W}, bprimitive_image.options().dtype(at::kInt));
    auto outline_coords = at::zeros({H * W, 2}, bprimitive_image.options().dtype(at::kInt));

    // const int threads = 32;
    // const dim3 blocks((W + threads - 1) / threads, (H + threads - 1) / threads);


    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);


    // Allocate and initialize coord_count on device
    auto coord_count = at::zeros({1}, bprimitive_image.options().dtype(at::kInt));
    
    // Launch kernel
    DetectOutlinePixelsKernel<<<gridSize, blockSize>>>(
        bprimitive_image.data_ptr<int32_t>(),
        outline_mask.data_ptr<int32_t>(),
        outline_coords.data_ptr<int32_t>(),
        coord_count.data_ptr<int32_t>(),
        W,
        H
    );

    // Get the actual count of outline pixels
    int actual_count = coord_count.item<int>();

    // Resize outline_coords to actual size
    outline_coords = outline_coords.slice(0, 0, actual_count);

    return std::make_tuple(outline_mask, outline_coords);
}