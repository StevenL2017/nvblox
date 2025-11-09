/*
Copyright 2023 NVIDIA CORPORATION

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

#include "nvblox/integrators/esdf_integrator.h"
#include "nvblox/integrators/esdf_integrator_params.h"
#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"
#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_integrator_params.h"
#include "nvblox/integrators/projective_occupancy_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/integrators/tsdf_decay_integrator.h"
#include "nvblox/mesh/mesh_integrator.h"
#include "nvblox/utils/params.h"

namespace nvblox {

// ======= DEPTH PRE-PROCESSING =======
constexpr Param<bool>::Description kDoDepthPrepocessingParamDesc{
    "do_depth_preprocessing", false,
    "Whether or not to run the preprocessing pipeline on the input depth "
    "image. Currently, this preprocessing only consists of dilating invalid "
    "regions in the input depth image."};

constexpr Param<int>::Description kDepthPreprocessingNumDilationsParamDesc{
    "depth_preprocessing_num_dilations", 4,
    "Number of times to run the invalid region dilation in the depth "
    "preprocessing pipeline (if do_depth_preprocessing is enabled)."};

// ======= DECAY =======
constexpr Param<bool>::Description kExcludeLastViewFromDecayParamDesc{
    "exclude_last_view_from_decay", false,
    "Whether contributions from the last depth frame should be excluded when "
    "decaying"};

constexpr Param<bool>::Description kAddObservedBlocksInFovParamDesc{
    "add_observed_blocks_in_fov", false,
    "Whether to collect newly observed TSDF blocks in the current field of "
    "view so they can be streamed or visualized immediately."};

constexpr Param<bool>::Description kClearUnobservedBlocksInFovParamDesc{
    "clear_unobserved_blocks_in_fov", false,
    "Whether to clear TSDF blocks in the current field of view that were not "
    "updated by the current depth frame."};

// ======= MESH OPTIMIZATION =======
constexpr Param<bool>::Description kMeshOptimizerEnableParamDesc{
    "mesh_optimizer_enable", false,
    "Enable block-level mesh post processing and simplification."};
constexpr Param<float>::Description kMeshOptimizerMinTriangleAreaFactorParamDesc{
    "mesh_optimizer_min_triangle_area_factor", 0.01f,
    "Minimum triangle area factor relative to voxel_size^2 used for culling."};
constexpr Param<float>::Description kMeshOptimizerMaxEdgeLengthFactorParamDesc{
    "mesh_optimizer_max_edge_length_factor", 4.0f,
    "Maximum edge length factor relative to voxel_size used for filtering."};
constexpr Param<float>::Description kMeshOptimizerMaxAspectRatioParamDesc{
    "mesh_optimizer_max_aspect_ratio", 10.0f,
    "Maximum allowed triangle aspect ratio (larger values are more permissive)."};
constexpr Param<float>::Description kMeshOptimizerSmallComponentAreaFactorParamDesc{
    "mesh_optimizer_small_component_area_factor", 0.0f,
    "Minimum summed triangle area factor (relative to voxel_size^2) for "
    "connected components to keep. Set to 0 to keep all components."};
constexpr Param<float>::Description kMeshOptimizerSimplifyTargetRatioParamDesc{
    "mesh_optimizer_simplify_target_ratio", 0.6f,
    "Target triangle index ratio for mesh simplification (0-1]."};
constexpr Param<float>::Description kMeshOptimizerSimplifyAbsErrorVoxParamDesc{
    "mesh_optimizer_simplify_abs_error_vox", 0.0f,
    "Allowed absolute simplification error expressed in voxel sizes. "
    "Set to 0 to disable the absolute error constraint."};
constexpr Param<float>::Description kMeshOptimizerSimplifyRelativeErrorParamDesc{
    "mesh_optimizer_simplify_relative_error", 0.01f,
    "Relative error bound used when absolute error is disabled."};
constexpr Param<bool>::Description kMeshOptimizerSimplifyUseSloppyParamDesc{
    "mesh_optimizer_simplify_use_sloppy", false,
    "Use the faster sloppy simplifier variant (lower quality)."};
constexpr Param<bool>::Description kMeshOptimizerSimplifyLockBorderParamDesc{
    "mesh_optimizer_simplify_lock_border", true,
    "Lock block borders during simplification to avoid cracks across blocks."};
constexpr Param<bool>::Description kMeshOptimizerOptimizeOverdrawParamDesc{
    "mesh_optimizer_optimize_overdraw", true,
    "Run the overdraw optimization pass after simplification."};
constexpr Param<float>::Description kMeshOptimizerOverdrawThresholdParamDesc{
    "mesh_optimizer_overdraw_threshold", 1.05f,
    "Expected overdraw threshold used by the overdraw optimizer."};

struct MeshOptimizerParams {
  Param<bool> mesh_optimizer_enable{kMeshOptimizerEnableParamDesc};
  Param<float> mesh_optimizer_min_triangle_area_factor{
      kMeshOptimizerMinTriangleAreaFactorParamDesc};
  Param<float> mesh_optimizer_max_edge_length_factor{
      kMeshOptimizerMaxEdgeLengthFactorParamDesc};
  Param<float> mesh_optimizer_max_aspect_ratio{
      kMeshOptimizerMaxAspectRatioParamDesc};
  Param<float> mesh_optimizer_small_component_area_factor{
      kMeshOptimizerSmallComponentAreaFactorParamDesc};
  Param<float> mesh_optimizer_simplify_target_ratio{
      kMeshOptimizerSimplifyTargetRatioParamDesc};
  Param<float> mesh_optimizer_simplify_abs_error_vox{
      kMeshOptimizerSimplifyAbsErrorVoxParamDesc};
  Param<float> mesh_optimizer_simplify_relative_error{
      kMeshOptimizerSimplifyRelativeErrorParamDesc};
  Param<bool> mesh_optimizer_simplify_use_sloppy{
      kMeshOptimizerSimplifyUseSloppyParamDesc};
  Param<bool> mesh_optimizer_simplify_lock_border{
      kMeshOptimizerSimplifyLockBorderParamDesc};
  Param<bool> mesh_optimizer_optimize_overdraw{
      kMeshOptimizerOptimizeOverdrawParamDesc};
  Param<float> mesh_optimizer_overdraw_threshold{
      kMeshOptimizerOverdrawThresholdParamDesc};
};

/// A structure containing the mapper parameters. This object can be used to set
/// all parameters of a mapper.
struct MapperParams {
  Param<bool> do_depth_preprocessing{kDoDepthPrepocessingParamDesc};
  Param<int> depth_preprocessing_num_dilations{
      kDepthPreprocessingNumDilationsParamDesc};
  Param<bool> exclude_last_view_from_decay{kExcludeLastViewFromDecayParamDesc};
  Param<bool> add_observed_blocks_in_fov{
      kAddObservedBlocksInFovParamDesc};
  Param<bool> clear_unobserved_blocks_in_fov{
      kClearUnobservedBlocksInFovParamDesc};

  EsdfIntegratorParams esdf_integrator_params;
  ProjectiveIntegratorParams projective_integrator_params;
  ViewCalculatorParams view_calculator_params;
  OccupancyIntegratorParams occupancy_integrator_params;
  MeshIntegratorParams mesh_integrator_params;
  MeshOptimizerParams mesh_optimizer_params;
  TsdfDecayIntegratorParams tsdf_decay_integrator_params;
  DecayIntegratorBaseParams decay_integrator_base_params;
  OccupancyDecayIntegratorParams occupancy_decay_integrator_params;
  FreespaceIntegratorParams freespace_integrator_params;
};

}  // namespace nvblox
