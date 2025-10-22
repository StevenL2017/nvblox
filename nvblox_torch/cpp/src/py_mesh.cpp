/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */
#include "nvblox_torch/py_mesh.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox_torch/cuda_stream.h"

#include <cstdint>
#include <c10/cuda/CUDAStream.h>

namespace pynvblox {

template <typename NativeAppearanceType>
torch::Tensor PyMesh<NativeAppearanceType>::vertices() const {
  if (mesh_->vertices.empty()) {
    return torch::empty({0, 3});
  }
  const int num_vertices = mesh_->vertices.size();
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  return torch::from_blob(mesh_->vertices.data(), {num_vertices, 3}, options);
}

template <typename NativeAppearanceType>
torch::Tensor PyMesh<NativeAppearanceType>::triangles() const {
  if (mesh_->triangle_indices.empty()) {
    return torch::empty({0, 3});
  }

  // Since the triangle indices are block-local, we need to unwrap them.
  // TODO(dtingdahl) Avoid the CPU roundtrip copy by writing a kernel that does
  // this directly on the GPU
  nvblox::host_vector<int> triangles_unwrapped(mesh_->triangle_indices.size());
  int triangle_idx_out = 0;
  for (size_t i_block = 0; i_block < mesh_->block_indices.size(); ++i_block) {
    const int num_triangle_indices_in_block =
        mesh_->getNumTriangleIndicesInBlock(i_block);

    for (int i_tri = 0; i_tri < num_triangle_indices_in_block; ++i_tri) {
      triangles_unwrapped[triangle_idx_out] =
          mesh_->triangle_indices[triangle_idx_out] +
          mesh_->vertex_block_offsets[i_block];

      ++triangle_idx_out;
    }
  }

  const auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);

  const int num_triangles = mesh_->triangle_indices.size() / 3;
  auto triangle_tensor = torch::empty({num_triangles, 3}, options);

  triangles_unwrapped.copyToAsync(
      reinterpret_cast<int*>(triangle_tensor.data_ptr()), getCurrentStream());
  getCurrentStream().synchronize();

  return triangle_tensor;
}

template <typename NativeAppearanceType>
torch::Tensor PyMesh<NativeAppearanceType>::vertex_appearances() const {
  if (mesh_->vertex_appearances.empty()) {
    return torch::empty({0, NativeAppearanceType::size()});
  }
  const int num_vertices = mesh_->vertex_appearances.size();
  if constexpr (std::is_same_v<NativeAppearanceType, nvblox::Color>) {
    const auto options =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
    return torch::from_blob(mesh_->vertex_appearances.data(),
                            {num_vertices, NativeAppearanceType::size()},
                            options);
  } else if constexpr (std::is_same_v<NativeAppearanceType,
                                      nvblox::FeatureArray>) {
    const auto options =
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    return torch::from_blob(mesh_->vertex_appearances.data(),
                            {num_vertices, NativeAppearanceType::size()},
                            options);
  } else {
    // Conversion not implemented for this voxel type
    assert(false);
  }
}

// Specializations
template class PyMesh<nvblox::Color>;
template class PyMesh<nvblox::FeatureArray>;

PyBlockMesh::PyBlockMesh()
    : block_(nullptr), block_index_(nvblox::Index3D::Zero()) {}

PyBlockMesh::PyBlockMesh(NativeBlockConstPtr block,
                         const nvblox::Index3D& block_index)
    : block_(nullptr),
      block_index_(block ? block_index : nvblox::Index3D::Zero()) {
  deepCopyBlock(block);
}

void PyBlockMesh::deepCopyBlock(const NativeBlockConstPtr& block) {
  if (!block) {
    block_.reset();
    return;
  }
  if (!block_) {
    block_ = std::make_shared<NativeBlockType>();
  }
  block_->copyFrom(*block);
}

torch::Tensor PyBlockMesh::vertices() const {
  if (!block_ || block_->vertices.empty()) {
    return torch::empty({0, 3});
  }
  const int num_vertices = static_cast<int>(block_->vertices.size());
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  return torch::from_blob(block_->vertices.data(), {num_vertices, 3}, options);
}

torch::Tensor PyBlockMesh::triangles() const {
  if (!block_ || block_->triangles.empty()) {
    return torch::empty({0, 3});
  }
  const int num_indices = static_cast<int>(block_->triangles.size());
  TORCH_CHECK(num_indices % 3 == 0,
              "Mesh block triangle index array must be divisible by 3.");
  const auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
  return torch::from_blob(block_->triangles.data(), {num_indices / 3, 3},
                          options);
}

torch::Tensor PyBlockMesh::vertex_appearances() const {
  if (!block_ || block_->vertex_appearances.empty()) {
    return torch::empty({0, nvblox::Color::size()});
  }
  const int num_vertices = static_cast<int>(block_->vertex_appearances.size());
  const auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
  return torch::from_blob(block_->vertex_appearances.data(),
                          {num_vertices, nvblox::Color::size()}, options);
}

torch::Tensor PyBlockMesh::block_key() const {
  const auto options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  if (!block_) {
    return torch::empty({0}, options);
  }
  auto key = torch::empty({3}, options);
  auto* key_ptr = key.data_ptr<int32_t>();
  key_ptr[0] = static_cast<int32_t>(block_index_.x());
  key_ptr[1] = static_cast<int32_t>(block_index_.y());
  key_ptr[2] = static_cast<int32_t>(block_index_.z());
  return key;
}

}  // namespace pynvblox
