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

#include "nvblox/mesh/mesh_block_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "meshoptimizer.h"

#include "nvblox/core/color.h"
#include "nvblox/core/feature_array.h"

namespace nvblox {
namespace {

constexpr float kAreaEpsilon = 1e-10f;
constexpr float kLengthEpsilon = 1e-6f;

template <typename AppearanceType>
struct HostVertex {
  Vector3f position;
  Vector3f normal;
  AppearanceType appearance;
};

inline bool isFinite(const Vector3f& v) { return v.allFinite(); }

inline float triangleArea(const Vector3f& a, const Vector3f& b,
                          const Vector3f& c) {
  return 0.5f * ((b - a).cross(c - a)).norm();
}

inline float edgeLength(const Vector3f& a, const Vector3f& b) {
  return (a - b).norm();
}

inline uint64_t makeEdgeKey(uint32_t a, uint32_t b) {
  if (a > b) {
    std::swap(a, b);
  }
  return (static_cast<uint64_t>(a) << 32u) | static_cast<uint64_t>(b);
}

template <typename AppearanceType>
void copyBlockToHost(const MeshBlock<AppearanceType>& block,
                     const CudaStream& cuda_stream,
                     std::vector<HostVertex<AppearanceType>>* vertices,
                     std::vector<uint32_t>* indices) {
  CHECK_NOTNULL(vertices);
  CHECK_NOTNULL(indices);

  const size_t vertex_count = block.vertices.size();
  const size_t triangle_index_count = block.triangles.size();

  vertices->clear();
  indices->clear();

  if (vertex_count == 0 || triangle_index_count < 3) {
    return;
  }

  const auto positions = block.vertices.toVectorAsync(cuda_stream);
  const auto normals = block.vertex_normals.toVectorAsync(cuda_stream);

  std::vector<AppearanceType> appearances;
  const bool has_appearances =
      block.vertex_appearances.size() == vertex_count && vertex_count > 0;
  if (has_appearances) {
    appearances = block.vertex_appearances.toVectorAsync(cuda_stream);
  }

  const auto triangle_indices = block.triangles.toVectorAsync(cuda_stream);
  cuda_stream.synchronize();

  vertices->resize(vertex_count);
  for (size_t i = 0; i < vertex_count; ++i) {
    auto& vertex = vertices->at(i);
    vertex.position = positions[i];
    vertex.normal =
        i < normals.size() ? normals[i] : Vector3f::Zero().eval();
    vertex.appearance =
        has_appearances ? appearances[i] : AppearanceType();
  }

  indices->reserve(triangle_indices.size());
  for (int tri_index : triangle_indices) {
    if (tri_index < 0) {
      indices->push_back(std::numeric_limits<uint32_t>::max());
    } else {
      indices->push_back(static_cast<uint32_t>(tri_index));
    }
  }
}

template <typename AppearanceType>
void copyHostToBlock(const std::vector<HostVertex<AppearanceType>>& vertices,
                     const std::vector<uint32_t>& indices,
                     MeshBlock<AppearanceType>* block,
                     const CudaStream& cuda_stream) {
  CHECK_NOTNULL(block);
  if (vertices.empty() || indices.size() < 3) {
    block->clear();
    return;
  }

  std::vector<Vector3f> positions(vertices.size());
  std::vector<Vector3f> normals(vertices.size());
  std::vector<AppearanceType> appearances(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    positions[i] = vertices[i].position;
    normals[i] = vertices[i].normal;
    appearances[i] = vertices[i].appearance;
  }

  std::vector<int> triangle_indices(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    triangle_indices[i] = static_cast<int>(indices[i]);
  }

  block->vertices.copyFromAsync(positions, cuda_stream);
  block->vertex_normals.copyFromAsync(normals, cuda_stream);
  block->vertex_appearances.copyFromAsync(appearances, cuda_stream);
  block->triangles.copyFromAsync(triangle_indices, cuda_stream);
  cuda_stream.synchronize();
}

template <typename AppearanceType>
void removeInvalidTriangles(
    const std::vector<HostVertex<AppearanceType>>& vertices,
    const MeshBlockOptimizerOptions& options,
    std::vector<uint32_t>* indices, std::vector<float>* triangle_areas) {
  CHECK_NOTNULL(indices);
  CHECK_NOTNULL(triangle_areas);
  triangle_areas->clear();

  if (indices->size() < 3) {
    indices->clear();
    return;
  }

  std::vector<uint32_t> filtered;
  filtered.reserve(indices->size());
  triangle_areas->reserve(indices->size() / 3);

  for (size_t i = 0; i + 2 < indices->size(); i += 3) {
    const uint32_t i0 = indices->at(i);
    const uint32_t i1 = indices->at(i + 1);
    const uint32_t i2 = indices->at(i + 2);

    if (i0 >= vertices.size() || i1 >= vertices.size() ||
        i2 >= vertices.size()) {
      continue;
    }

    if (i0 == i1 || i1 == i2 || i0 == i2) {
      continue;
    }

    const HostVertex<AppearanceType>& v0 = vertices[i0];
    const HostVertex<AppearanceType>& v1 = vertices[i1];
    const HostVertex<AppearanceType>& v2 = vertices[i2];

    if (!isFinite(v0.position) || !isFinite(v1.position) ||
        !isFinite(v2.position) || !isFinite(v0.normal) ||
        !isFinite(v1.normal) || !isFinite(v2.normal)) {
      continue;
    }

    const float area = triangleArea(v0.position, v1.position, v2.position);
    if (!std::isfinite(area) || area <= kAreaEpsilon) {
      continue;
    }
    if (options.min_triangle_area_m2 > 0.0f &&
        area < options.min_triangle_area_m2) {
      continue;
    }

    const float e0 = edgeLength(v0.position, v1.position);
    const float e1 = edgeLength(v1.position, v2.position);
    const float e2 = edgeLength(v2.position, v0.position);
    if (e0 <= kLengthEpsilon || e1 <= kLengthEpsilon ||
        e2 <= kLengthEpsilon) {
      continue;
    }
    if (options.max_edge_length_m > 0.0f) {
      const float max_edge = std::max({e0, e1, e2});
      if (max_edge > options.max_edge_length_m) {
        continue;
      }
    }

    if (options.max_aspect_ratio > 0.0f) {
      const float perimeter = e0 + e1 + e2;
      if (perimeter <= kLengthEpsilon) {
        continue;
      }
      const float inradius = (2.0f * area) / perimeter;
      if (inradius <= kLengthEpsilon) {
        continue;
      }
      const float max_edge = std::max({e0, e1, e2});
      const float aspect_ratio = max_edge / (2.0f * inradius);
      if (aspect_ratio > options.max_aspect_ratio) {
        continue;
      }
    }

    filtered.push_back(i0);
    filtered.push_back(i1);
    filtered.push_back(i2);
    triangle_areas->push_back(area);
  }

  indices->swap(filtered);
}

void removeHighValenceTriangles(const MeshBlockOptimizerOptions& options,
                                std::vector<uint32_t>* indices,
                                std::vector<float>* triangle_areas) {
  CHECK_NOTNULL(indices);
  CHECK_NOTNULL(triangle_areas);
  if (indices->empty()) {
    return;
  }

  const size_t triangle_count = indices->size() / 3;
  std::unordered_map<uint64_t, std::vector<int>> edge_to_triangles;
  edge_to_triangles.reserve(triangle_count * 3);

  for (size_t tri = 0; tri < triangle_count; ++tri) {
    const uint32_t i0 = indices->at(3 * tri + 0);
    const uint32_t i1 = indices->at(3 * tri + 1);
    const uint32_t i2 = indices->at(3 * tri + 2);

    edge_to_triangles[makeEdgeKey(i0, i1)].push_back(static_cast<int>(tri));
    edge_to_triangles[makeEdgeKey(i1, i2)].push_back(static_cast<int>(tri));
    edge_to_triangles[makeEdgeKey(i2, i0)].push_back(static_cast<int>(tri));
  }

  std::vector<bool> remove_triangle(triangle_count, false);
  const float suspicious_area =
      options.min_triangle_area_m2 > 0.0f
          ? std::max(options.min_triangle_area_m2 * 4.0f, kAreaEpsilon)
          : kAreaEpsilon;

  for (const auto& item : edge_to_triangles) {
    const auto& tris = item.second;
    if (tris.size() <= 2) {
      continue;
    }
    for (int tri_index : tris) {
      if (tri_index < 0 || static_cast<size_t>(tri_index) >= triangle_count) {
        continue;
      }
      if (triangle_areas->at(tri_index) <= suspicious_area) {
        remove_triangle[tri_index] = true;
      }
    }
  }

  if (std::none_of(remove_triangle.begin(), remove_triangle.end(),
                   [](bool value) { return value; })) {
    return;
  }

  std::vector<uint32_t> filtered_indices;
  std::vector<float> filtered_areas;
  filtered_indices.reserve(indices->size());
  filtered_areas.reserve(triangle_areas->size());

  for (size_t tri = 0; tri < triangle_count; ++tri) {
    if (remove_triangle[tri]) {
      continue;
    }
    filtered_indices.push_back(indices->at(3 * tri + 0));
    filtered_indices.push_back(indices->at(3 * tri + 1));
    filtered_indices.push_back(indices->at(3 * tri + 2));
    filtered_areas.push_back(triangle_areas->at(tri));
  }

  indices->swap(filtered_indices);
  triangle_areas->swap(filtered_areas);
}

template <typename AppearanceType>
void removeSmallComponents(
    const std::vector<HostVertex<AppearanceType>>& vertices,
    const MeshBlockOptimizerOptions& options, std::vector<uint32_t>* indices,
    std::vector<float>* triangle_areas) {
  (void)vertices;
  CHECK_NOTNULL(indices);
  CHECK_NOTNULL(triangle_areas);
  if (options.small_component_area_m2 <= 0.0f || indices->empty()) {
    return;
  }

  const size_t triangle_count = indices->size() / 3;
  std::unordered_map<uint64_t, std::vector<int>> edge_to_triangles;
  edge_to_triangles.reserve(triangle_count * 3);

  std::vector<std::vector<int>> adjacency(triangle_count);
  for (size_t tri = 0; tri < triangle_count; ++tri) {
    const uint32_t i0 = indices->at(3 * tri + 0);
    const uint32_t i1 = indices->at(3 * tri + 1);
    const uint32_t i2 = indices->at(3 * tri + 2);

    const uint64_t edges[3] = {makeEdgeKey(i0, i1), makeEdgeKey(i1, i2),
                               makeEdgeKey(i2, i0)};
    for (uint64_t edge_key : edges) {
      auto& tris = edge_to_triangles[edge_key];
      for (int other_tri : tris) {
        adjacency[tri].push_back(other_tri);
        adjacency[other_tri].push_back(static_cast<int>(tri));
      }
      tris.push_back(static_cast<int>(tri));
    }
  }

  std::vector<bool> visited(triangle_count, false);
  std::vector<bool> remove_triangle(triangle_count, false);

  for (size_t tri = 0; tri < triangle_count; ++tri) {
    if (visited[tri]) {
      continue;
    }

    std::queue<int> queue;
    std::vector<int> component;
    float component_area = 0.0f;

    queue.push(static_cast<int>(tri));
    visited[tri] = true;

    while (!queue.empty()) {
      const int current = queue.front();
      queue.pop();

      component.push_back(current);
      component_area += triangle_areas->at(current);

      for (int neighbor : adjacency[current]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }

    if (component_area < options.small_component_area_m2) {
      for (int idx : component) {
        remove_triangle[idx] = true;
      }
    }
  }

  if (std::none_of(remove_triangle.begin(), remove_triangle.end(),
                   [](bool value) { return value; })) {
    return;
  }

  std::vector<uint32_t> filtered_indices;
  std::vector<float> filtered_areas;
  filtered_indices.reserve(indices->size());
  filtered_areas.reserve(triangle_areas->size());

  for (size_t tri = 0; tri < triangle_count; ++tri) {
    if (remove_triangle[tri]) {
      continue;
    }
    filtered_indices.push_back(indices->at(3 * tri + 0));
    filtered_indices.push_back(indices->at(3 * tri + 1));
    filtered_indices.push_back(indices->at(3 * tri + 2));
    filtered_areas.push_back(triangle_areas->at(tri));
  }

  indices->swap(filtered_indices);
  triangle_areas->swap(filtered_areas);
}

template <typename AppearanceType>
void remapVertices(std::vector<HostVertex<AppearanceType>>* vertices,
                   std::vector<uint32_t>* indices) {
  CHECK_NOTNULL(vertices);
  CHECK_NOTNULL(indices);
  if (vertices->empty() || indices->empty()) {
    return;
  }

  std::vector<uint32_t> remap(vertices->size());
  const size_t unique_vertices =
      meshopt_generateVertexRemap(remap.data(), indices->data(),
                                  indices->size(), vertices->data(),
                                  vertices->size(), sizeof(HostVertex<AppearanceType>));

  std::vector<HostVertex<AppearanceType>> remapped_vertices(unique_vertices);
  std::vector<uint32_t> remapped_indices(indices->size());

  meshopt_remapVertexBuffer(remapped_vertices.data(), vertices->data(),
                            vertices->size(), sizeof(HostVertex<AppearanceType>),
                            remap.data());
  meshopt_remapIndexBuffer(remapped_indices.data(), indices->data(),
                           indices->size(), remap.data());

  vertices->swap(remapped_vertices);
  indices->swap(remapped_indices);
}

template <typename AppearanceType>
void simplifyMesh(const MeshBlockOptimizerOptions& options,
                  std::vector<HostVertex<AppearanceType>>* vertices,
                  std::vector<uint32_t>* indices) {
  CHECK_NOTNULL(vertices);
  CHECK_NOTNULL(indices);
  if (indices->size() < 3 || vertices->empty()) {
    return;
  }

  const bool simplification_requested =
      options.simplify_target_ratio < 0.999f ||
      options.simplify_abs_error_m > 0.0f;
  if (!simplification_requested) {
    return;
  }

  size_t target_index_count = static_cast<size_t>(
      static_cast<double>(indices->size()) * options.simplify_target_ratio);
  target_index_count = (target_index_count / 3) * 3;
  target_index_count = std::max<size_t>(target_index_count, 3);

  if (target_index_count >= indices->size()) {
    return;
  }

  std::vector<uint32_t> simplified(indices->size());
  float relative_error = std::max(options.simplify_relative_error, 1e-5f);

  if (options.simplify_abs_error_m > 0.0f) {
    const float scale =
        meshopt_simplifyScale(vertices->front().position.data(),
                              vertices->size(), sizeof(HostVertex<AppearanceType>));
    if (scale > 0.0f) {
      relative_error = options.simplify_abs_error_m / scale;
    }
  }

  size_t new_index_count = 0;
  if (options.simplify_use_sloppy && !options.simplify_lock_border) {
    new_index_count = meshopt_simplifySloppy(
        simplified.data(), indices->data(), indices->size(),
        vertices->front().position.data(), vertices->size(),
        sizeof(HostVertex<AppearanceType>), target_index_count, relative_error,
        nullptr);
  } else {
    const unsigned int flags = options.simplify_lock_border
                                   ? static_cast<unsigned int>(
                                         meshopt_SimplifyLockBorder)
                                   : 0u;
    new_index_count = meshopt_simplify(
        simplified.data(), indices->data(), indices->size(),
        vertices->front().position.data(), vertices->size(),
        sizeof(HostVertex<AppearanceType>), target_index_count, relative_error,
        flags, nullptr);
  }

  if (new_index_count < 3 || new_index_count % 3 != 0 ||
      new_index_count >= indices->size()) {
    return;
  }

  indices->assign(simplified.begin(), simplified.begin() + new_index_count);
}

template <typename AppearanceType>
void optimizeForRendering(const MeshBlockOptimizerOptions& options,
                          std::vector<HostVertex<AppearanceType>>* vertices,
                          std::vector<uint32_t>* indices) {
  CHECK_NOTNULL(vertices);
  CHECK_NOTNULL(indices);
  if (indices->empty() || vertices->empty()) {
    return;
  }

  meshopt_optimizeVertexCache(indices->data(), indices->data(),
                              indices->size(), vertices->size());

  if (options.optimize_overdraw) {
    meshopt_optimizeOverdraw(
        indices->data(), indices->data(), indices->size(),
        vertices->front().position.data(), vertices->size(),
        sizeof(HostVertex<AppearanceType>), options.overdraw_threshold);
  }

  std::vector<HostVertex<AppearanceType>> reordered(vertices->size());
  meshopt_optimizeVertexFetch(
      reordered.data(), indices->data(), indices->size(), vertices->data(),
      vertices->size(), sizeof(HostVertex<AppearanceType>));
  vertices->swap(reordered);
}

}  // namespace

template <typename AppearanceType>
void optimizeMeshBlock(MeshBlock<AppearanceType>* block,
                       const MeshBlockOptimizerOptions& options,
                       const CudaStream& cuda_stream) {
  CHECK_NOTNULL(block);
  if (!options.enabled) {
    return;
  }

  std::vector<HostVertex<AppearanceType>> vertices;
  std::vector<uint32_t> indices;
  copyBlockToHost(*block, cuda_stream, &vertices, &indices);

  if (indices.empty() || vertices.empty()) {
    block->clear();
    return;
  }

  std::vector<float> triangle_areas;
  removeInvalidTriangles(vertices, options, &indices, &triangle_areas);
  removeHighValenceTriangles(options, &indices, &triangle_areas);
  removeSmallComponents(vertices, options, &indices, &triangle_areas);

  if (indices.empty()) {
    block->clear();
    return;
  }

  remapVertices(&vertices, &indices);
  simplifyMesh(options, &vertices, &indices);

  if (indices.empty() || vertices.empty()) {
    block->clear();
    return;
  }

  remapVertices(&vertices, &indices);
  optimizeForRendering(options, &vertices, &indices);
  copyHostToBlock(vertices, indices, block, cuda_stream);
}

template void optimizeMeshBlock<Color>(MeshBlock<Color>*,
                                       const MeshBlockOptimizerOptions&,
                                       const CudaStream&);
template void optimizeMeshBlock<FeatureArray>(
    MeshBlock<FeatureArray>*, const MeshBlockOptimizerOptions&,
    const CudaStream&);

}  // namespace nvblox
