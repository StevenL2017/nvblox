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

#include "nvblox_torch/py_multi_mapper.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/map/blocks_to_update_tracker.h"
#include "nvblox/serialization/layer_streamer.h"
#include "nvblox_torch/check_utils.h"

namespace pynvblox {

namespace {

nvblox::MappingType mappingTypeFromString(const std::string& mapping_type) {
  if (mapping_type == "kStaticTsdf") {
    return nvblox::MappingType::kStaticTsdf;
  }
  if (mapping_type == "kStaticOccupancy") {
    return nvblox::MappingType::kStaticOccupancy;
  }
  if (mapping_type == "kDynamic") {
    return nvblox::MappingType::kDynamic;
  }
  if (mapping_type == "kHumanWithStaticTsdf") {
    return nvblox::MappingType::kHumanWithStaticTsdf;
  }
  if (mapping_type == "kHumanWithStaticOccupancy") {
    return nvblox::MappingType::kHumanWithStaticOccupancy;
  }
  throw std::runtime_error("Unsupported mapping type string: " + mapping_type);
}

nvblox::EsdfMode esdfModeFromString(const std::string& esdf_mode) {
  if (esdf_mode == "k3D") {
    return nvblox::EsdfMode::k3D;
  }
  if (esdf_mode == "k2D") {
    return nvblox::EsdfMode::k2D;
  }
  if (esdf_mode == "kUnset") {
    return nvblox::EsdfMode::kUnset;
  }
  throw std::runtime_error("Unsupported esdf mode string: " + esdf_mode);
}

}  // namespace

MultiMapper::MultiMapper(
    double voxel_size_m, const std::string& mapping_type_str,
    const std::string& esdf_mode_str,
    c10::intrusive_ptr<MapperParams> background_mapper_params,
    c10::optional<c10::intrusive_ptr<MapperParams>> foreground_mapper_params) {
  CHECK(background_mapper_params);
  const nvblox::MappingType mapping_type =
      mappingTypeFromString(mapping_type_str);
  nvblox::EsdfMode esdf_mode = esdfModeFromString(esdf_mode_str);
  if (esdf_mode == nvblox::EsdfMode::kUnset) {
    esdf_mode = nvblox::EsdfMode::k3D;
  }
  multi_mapper_ = std::make_shared<nvblox::MultiMapper>(
      voxel_size_m, mapping_type, esdf_mode, nvblox::MemoryType::kDevice);

  nvblox::MapperParams background_params =
      *background_mapper_params->params_;
  std::optional<nvblox::MapperParams> foreground_params = std::nullopt;
  if (foreground_mapper_params.has_value() &&
      foreground_mapper_params.value() != nullptr) {
    foreground_params = *foreground_mapper_params.value()->params_;
  }
  multi_mapper_->setMapperParams(background_params, foreground_params);
}
void MultiMapper::integrateDepth(torch::Tensor depth_frame_t,
                                 torch::Tensor T_L_C_t,
                                 torch::Tensor intrinsics_t,
                                 c10::optional<int64_t> update_time_ms) {
  TORCH_CHECK(depth_frame_t.dim() == 2,
              "Depth frame must be HxW float32 tensor.");
  TORCH_CHECK(depth_frame_t.scalar_type() == at::kFloat,
              "Depth frame must be float32.");
  if (!checkSizes(T_L_C_t, {4, 4}) || !checkSizes(intrinsics_t, {3, 3})) {
    LOG(WARNING) << "Pose and intrinsics tensor sizes are not correct";
    return;
  }
  const int64_t height = depth_frame_t.size(0);
  const int64_t width = depth_frame_t.size(1);
  nvblox::DepthImage depth_image = copy_depth_image_from_tensor(depth_frame_t);
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);

  std::optional<nvblox::Time> time_opt = std::nullopt;
  if (update_time_ms.has_value()) {
    time_opt = static_cast<nvblox::Time>(update_time_ms.value());
  }
  multi_mapper_->integrateDepth(depth_image, T_L_C, camera, time_opt);
}

void MultiMapper::integrateColor(torch::Tensor color_frame_t,
                                 torch::Tensor T_L_C_t,
                                 torch::Tensor intrinsics_t) {
  TORCH_CHECK(color_frame_t.dim() == 3,
              "Color frame must be HxWx4 tensor.");
  TORCH_CHECK(color_frame_t.scalar_type() == at::kByte,
              "Color frame must be uint8.");
  if (!checkSizes(T_L_C_t, {4, 4}) || !checkSizes(intrinsics_t, {3, 3})) {
    LOG(WARNING) << "Pose and intrinsics tensor sizes are not correct";
    return;
  }
  const int64_t height = color_frame_t.size(0);
  const int64_t width = color_frame_t.size(1);
  nvblox::ColorImage color_image = copy_color_image_from_tensor(color_frame_t);
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);
  multi_mapper_->integrateColor(color_image, T_L_C, camera);
}

void MultiMapper::updateColorMesh() { multi_mapper_->updateColorMesh(); }

std::vector<c10::intrusive_ptr<pynvblox::PyBlockMesh>>
MultiMapper::getDeltaBlockMesh() {
  std::vector<c10::intrusive_ptr<pynvblox::PyBlockMesh>> block_meshes;
  auto background_mapper = multi_mapper_->background_mapper();
  std::vector<nvblox::Index3D> block_indices =
      background_mapper->getBlocksToUpdate(
          nvblox::BlocksToUpdateType::kLayerStreamer,
          nvblox::UpdateFullLayer::kNo);
  block_meshes.reserve(block_indices.size());
  for (const nvblox::Index3D& block_index : block_indices) {
    auto block =
        background_mapper->color_mesh_layer().getBlockAtIndex(block_index);
    if (!block) {
      continue;
    }
    block_meshes.emplace_back(
        c10::make_intrusive<pynvblox::PyBlockMesh>(block, block_index));
  }
  return block_meshes;
}

c10::intrusive_ptr<pynvblox::PyColorMesh> MultiMapper::getColorMesh() {
  auto background_mapper = multi_mapper_->background_mapper();
  constexpr float kUnlimitedBandwidth = -1.0F;
  std::vector<nvblox::Index3D> all_blocks =
      background_mapper->color_mesh_layer().getAllBlockIndices();
  background_mapper->serializeSelectedLayers(
      nvblox::LayerType::kColorMesh, kUnlimitedBandwidth,
      nvblox::BlockExclusionParams(), all_blocks);
  auto serialized_mesh = background_mapper->serializedColorMeshLayer();
  return c10::make_intrusive<pynvblox::PyColorMesh>(serialized_mesh);
}

}  // namespace pynvblox
