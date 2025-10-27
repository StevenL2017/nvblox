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

constexpr Param<bool>::Description kClearUnobservedBlocksInFovParamDesc{
    "clear_unobserved_blocks_in_fov", false,
    "Whether to clear TSDF blocks in the current field of view that were not "
    "updated by the current depth frame."};

constexpr Param<bool>::Description kFilterSmallTsdfBlockUpdatesParamDesc{
    "filter_small_tsdf_block_updates", true,
    "Whether to filter TSDF block updates whose aggregate change is below a "
    "threshold so the mesh stays static when geometry is unchanged."};

constexpr Param<float>::Description
    kMinTsdfBlockDistanceChangeThresholdParamDesc{
        "min_tsdf_block_distance_change_threshold", 1e-3f,
        "Minimum change in average TSDF distance (or absolute distance) per "
        "block required before propagating the block as updated."};

constexpr Param<float>::Description
    kMinTsdfBlockWeightChangeThresholdParamDesc{
        "min_tsdf_block_weight_change_threshold", 1e-2f,
        "Minimum change in average or maximum TSDF weight per block required "
        "before propagating the block as updated."};

/// A structure containing the mapper parameters. This object can be used to set
/// all parameters of a mapper.
struct MapperParams {
  Param<bool> do_depth_preprocessing{kDoDepthPrepocessingParamDesc};
  Param<int> depth_preprocessing_num_dilations{
      kDepthPreprocessingNumDilationsParamDesc};
  Param<bool> exclude_last_view_from_decay{kExcludeLastViewFromDecayParamDesc};
  Param<bool> clear_unobserved_blocks_in_fov{
      kClearUnobservedBlocksInFovParamDesc};
  Param<bool> filter_small_tsdf_block_updates{
      kFilterSmallTsdfBlockUpdatesParamDesc};
  Param<float> min_tsdf_block_distance_change_threshold{
      kMinTsdfBlockDistanceChangeThresholdParamDesc};
  Param<float> min_tsdf_block_weight_change_threshold{
      kMinTsdfBlockWeightChangeThresholdParamDesc};

  EsdfIntegratorParams esdf_integrator_params;
  ProjectiveIntegratorParams projective_integrator_params;
  ViewCalculatorParams view_calculator_params;
  OccupancyIntegratorParams occupancy_integrator_params;
  MeshIntegratorParams mesh_integrator_params;
  TsdfDecayIntegratorParams tsdf_decay_integrator_params;
  DecayIntegratorBaseParams decay_integrator_base_params;
  OccupancyDecayIntegratorParams occupancy_decay_integrator_params;
  FreespaceIntegratorParams freespace_integrator_params;
};

}  // namespace nvblox
