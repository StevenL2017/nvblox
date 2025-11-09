/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#pragma once

#include <torch/script.h>

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nvblox/mapper/multi_mapper.h"

#include "nvblox_torch/convert_tensors.h"
#include "nvblox_torch/py_mapper_params.h"
#include "nvblox_torch/py_mesh.h"

namespace pynvblox {

struct MultiMapper : torch::CustomClassHolder {
  MultiMapper(double voxel_size_m, const std::string& mapping_type_str,
              const std::string& esdf_mode_str,
              c10::intrusive_ptr<MapperParams> background_mapper_params,
              c10::optional<c10::intrusive_ptr<MapperParams>>
                  foreground_mapper_params = c10::nullopt);

  ~MultiMapper() = default;

  void integrateDepth(torch::Tensor depth_frame_t, torch::Tensor T_L_C_t,
                      torch::Tensor intrinsics_t,
                      c10::optional<int64_t> update_time_ms);

  void integrateColor(torch::Tensor color_frame_t, torch::Tensor T_L_C_t,
                      torch::Tensor intrinsics_t);

  void updateColorMesh();

  std::vector<c10::intrusive_ptr<pynvblox::PyBlockMesh>> getDeltaBlockMesh();

  c10::intrusive_ptr<pynvblox::PyColorMesh> getColorMesh();

  std::shared_ptr<nvblox::MultiMapper> multi_mapper_;
};

}  // namespace pynvblox
