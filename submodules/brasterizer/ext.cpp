/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <torch/extension.h>

#include "csrc/rasterize.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_rasterize_meshes_coarse", &RasterizeMeshesCoarseCuda);
  m.def("_rasterize_meshes_fine", &RasterizeMeshesFineCuda);
  m.def("_detect_outline_vertices", &DetectOutlineVerticesCuda);
  m.def("_detect_outline_pixels", &DetectOutlinePixelsCuda);
}
