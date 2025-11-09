/*
Copyright 2024 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

struct MeshBlockOptimizerOptions {
  bool enabled = false;
  float min_triangle_area_m2 = 0.0f;
  float max_edge_length_m = 0.0f;
  float max_aspect_ratio = 0.0f;
  float small_component_area_m2 = 0.0f;
  float simplify_target_ratio = 1.0f;
  float simplify_abs_error_m = 0.0f;
  float simplify_relative_error = 1e-2f;
  bool simplify_use_sloppy = false;
  bool simplify_lock_border = true;
  bool optimize_overdraw = true;
  float overdraw_threshold = 1.05f;
};

template <typename AppearanceType>
void optimizeMeshBlock(MeshBlock<AppearanceType>* block,
                       const MeshBlockOptimizerOptions& options,
                       const CudaStream& cuda_stream);

}  // namespace nvblox
