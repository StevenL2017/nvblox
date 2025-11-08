/*
Copyright 2024 NVIDIA CORPORATION

Licensed under the the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "nvblox/tracking/depth_to_tsdf_icp.h"

#include <Eigen/Dense>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>

#include "nvblox/core/internal/error_check.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

namespace {

using nvblox::Camera;
using nvblox::DepthImageConstView;
using nvblox::DepthImageView;
using nvblox::ImageView;
using nvblox::Index2D;
using nvblox::MaskedDepthImageConstView;
using nvblox::MaskMode;
using nvblox::Transform;
using nvblox::Vector2f;
using nvblox::Vector3f;

constexpr int kThreadsPerDim = 16;
constexpr int kIcpSystemSize = 6;
constexpr float kNormalEpsilon = 1e-6f;
constexpr int kMinCorrespondenceCount = 100;
constexpr float kDegToRad = static_cast<float>(M_PI) / 180.0f;

__device__ inline bool isValidDepthValue(float depth) {
  return depth > 0.0f && isfinite(depth);
}

__device__ inline bool isValidVertex(const Vector3f& vertex) {
  return isfinite(vertex.x()) && isfinite(vertex.y()) &&
         isfinite(vertex.z()) && vertex.z() > 0.0f;
}

__device__ inline bool isValidNormal(const Vector3f& normal) {
  return isfinite(normal.x()) && isfinite(normal.y()) &&
         isfinite(normal.z()) &&
         normal.squaredNorm() > kNormalEpsilon * kNormalEpsilon;
}

__global__ void applyMaskKernel(DepthImageView depth_view,
                                ImageView<const uint8_t> mask_view,
                                MaskMode mask_mode) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= depth_view.rows() || col >= depth_view.cols()) {
    return;
  }

  const uint8_t mask_value = mask_view(row, col);
  const bool is_active =
      (mask_mode == MaskMode::kNonInverted) ? (mask_value != 0)
                                            : (mask_value == 0);
  if (!is_active) {
    depth_view(row, col) = 0.0f;
  }
}

__global__ void downsampleDepthKernel(const DepthImageConstView depth_in,
                                      DepthImageView depth_out,
                                      const int factor) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= depth_out.rows() || col >= depth_out.cols()) {
    return;
  }

  const int base_row = row * factor;
  const int base_col = col * factor;

  float sum = 0.0f;
  int count = 0;
  for (int r = 0; r < factor; ++r) {
    for (int c = 0; c < factor; ++c) {
      const float depth = depth_in(base_row + r, base_col + c);
      if (isValidDepthValue(depth)) {
        sum += depth;
        ++count;
      }
    }
  }

  depth_out(row, col) = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void depthToVertexMapCameraKernel(
    DepthImageConstView depth_view, Camera camera,
    ImageView<Vector3f> vertex_view) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= depth_view.rows() || col >= depth_view.cols()) {
    return;
  }

  const float depth = depth_view(row, col);
  if (!isValidDepthValue(depth)) {
    vertex_view(row, col) = Vector3f::Zero();
    return;
  }

  const Vector2f pixel(static_cast<float>(col) + 0.5f,
                       static_cast<float>(row) + 0.5f);
  const Vector3f ray = camera.vectorFromImagePlaneCoordinates(pixel);
  vertex_view(row, col) = depth * ray;
}

__global__ void depthToVertexMapWorldKernel(DepthImageConstView depth_view,
                                            Camera camera,
                                            Transform T_L_C,
                                            ImageView<Vector3f> vertex_view) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= depth_view.rows() || col >= depth_view.cols()) {
    return;
  }

  const float depth = depth_view(row, col);
  if (!isValidDepthValue(depth)) {
    vertex_view(row, col) = Vector3f::Zero();
    return;
  }

  const Vector2f pixel(static_cast<float>(col) + 0.5f,
                       static_cast<float>(row) + 0.5f);
  const Vector3f ray = camera.vectorFromImagePlaneCoordinates(pixel);
  const Vector3f point_C = depth * ray;
  vertex_view(row, col) = T_L_C * point_C;
}

__global__ void computeNormalsKernel(ImageView<const Vector3f> vertex_view,
                                     ImageView<Vector3f> normal_view) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= vertex_view.rows() || col >= vertex_view.cols()) {
    return;
  }

  // Skip border pixels.
  if (row == 0 || col == 0 || row == vertex_view.rows() - 1 ||
      col == vertex_view.cols() - 1) {
    normal_view(row, col) = Vector3f::Zero();
    return;
  }

  const Vector3f center = vertex_view(row, col);
  if (!isValidVertex(center)) {
    normal_view(row, col) = Vector3f::Zero();
    return;
  }

  const Vector3f left = vertex_view(row, col - 1);
  const Vector3f right = vertex_view(row, col + 1);
  const Vector3f up = vertex_view(row - 1, col);
  const Vector3f down = vertex_view(row + 1, col);

  if (!isValidVertex(left) || !isValidVertex(right) || !isValidVertex(up) ||
      !isValidVertex(down)) {
    normal_view(row, col) = Vector3f::Zero();
    return;
  }

  const Vector3f d_col = right - left;
  const Vector3f d_row = down - up;
  Vector3f normal = d_row.cross(d_col);
  const float norm = normal.norm();
  if (norm > kNormalEpsilon) {
    normal_view(row, col) = normal / norm;
  } else {
    normal_view(row, col) = Vector3f::Zero();
  }
}

__global__ void accumulateNormalEquationsKernel(
    ImageView<const Vector3f> vertices_obs_C,
    ImageView<const Vector3f> vertices_mod_L,
    ImageView<const Vector3f> normals_mod_L, DepthImageConstView depth_obs,
    DepthImageConstView depth_syn, Transform T_L_C,
    const float depth_threshold_m, const float huber_delta_m, float* JTJ,
    float* JTr, float* residual_sums, int* counts) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= vertices_obs_C.rows() || col >= vertices_obs_C.cols()) {
    return;
  }

  const float depth_obs_val = depth_obs(row, col);
  const float depth_syn_val = depth_syn(row, col);
  if (!isValidDepthValue(depth_obs_val) ||
      !isValidDepthValue(depth_syn_val) ||
      fabsf(depth_obs_val - depth_syn_val) > depth_threshold_m) {
    return;
  }

  const Vector3f p_obs_C = vertices_obs_C(row, col);
  const Vector3f p_mod_L = vertices_mod_L(row, col);
  const Vector3f n_L = normals_mod_L(row, col);
  if (!isValidVertex(p_obs_C) || !isValidVertex(p_mod_L) ||
      !isValidNormal(n_L)) {
    return;
  }

  const Vector3f p_obs_L = T_L_C * p_obs_C;
  const Vector3f residual_vec = p_obs_L - p_mod_L;
  const float residual = n_L.dot(residual_vec);
  if (!isfinite(residual)) {
    return;
  }

  const float abs_residual = fabsf(residual);
  float weight = 1.0f;
  if (abs_residual > huber_delta_m) {
    weight = huber_delta_m / (abs_residual + 1e-6f);
  }

  const Vector3f rot_jac = n_L.cross(p_obs_L);
  const Vector3f trans_jac = n_L;

  float J[kIcpSystemSize];
  J[0] = rot_jac.x();
  J[1] = rot_jac.y();
  J[2] = rot_jac.z();
  J[3] = trans_jac.x();
  J[4] = trans_jac.y();
  J[5] = trans_jac.z();

  const float weighted_residual = weight * residual;

  for (int r = 0; r < kIcpSystemSize; ++r) {
    atomicAdd(&JTr[r], weighted_residual * J[r]);
    for (int c = 0; c < kIcpSystemSize; ++c) {
      atomicAdd(&JTJ[r * kIcpSystemSize + c], weight * J[r] * J[c]);
    }
  }

  atomicAdd(residual_sums, abs_residual);
  atomicAdd(&counts[0], 1);
  if (abs_residual < depth_threshold_m) {
    atomicAdd(&counts[1], 1);
  }
}

__global__ void maskOverlappingPixelsKernel(
    DepthImageView depth_obs, DepthImageConstView depth_syn,
    ImageView<const Vector3f> normals_obs,
    ImageView<const Vector3f> normals_syn, const float depth_threshold_m,
    const float cos_normal_threshold) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= depth_obs.rows() || col >= depth_obs.cols()) {
    return;
  }

  const float depth_obs_val = depth_obs(row, col);
  const float depth_syn_val = depth_syn(row, col);
  if (!isValidDepthValue(depth_obs_val) ||
      !isValidDepthValue(depth_syn_val)) {
    return;
  }

  if (fabsf(depth_obs_val - depth_syn_val) > depth_threshold_m) {
    return;
  }

  const Vector3f normal_obs = normals_obs(row, col);
  const Vector3f normal_syn = normals_syn(row, col);
  if (!isValidNormal(normal_obs) || !isValidNormal(normal_syn)) {
    return;
  }

  const float cos_angle = normal_obs.normalized().dot(normal_syn.normalized());
  if (cos_angle < cos_normal_threshold) {
    return;
  }

  depth_obs(row, col) = 0.0f;
}

Camera scaledCamera(const Camera& camera, int subsampling) {
  if (subsampling <= 1) {
    return camera;
  }
  const float scale = static_cast<float>(subsampling);
  return Camera(camera.fu() / scale, camera.fv() / scale,
                camera.cu() / scale, camera.cv() / scale,
                camera.width() / subsampling, camera.height() / subsampling);
}

Eigen::Matrix3f skewSymmetric(const Eigen::Vector3f& v) {
  Eigen::Matrix3f mat;
  mat << 0.0f, -v.z(), v.y(), v.z(), 0.0f, -v.x(), -v.y(), v.x(), 0.0f;
  return mat;
}

Transform se3Exp(const Eigen::Matrix<float, 6, 1>& xi) {
  const Eigen::Vector3f omega = xi.head<3>();
  const Eigen::Vector3f upsilon = xi.tail<3>();
  const float theta = omega.norm();
  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f V = Eigen::Matrix3f::Identity();

  if (theta < 1e-5f) {
    const Eigen::Matrix3f Omega = skewSymmetric(omega);
    R += Omega;
    V += 0.5f * Omega;
  } else {
    const Eigen::Matrix3f Omega = skewSymmetric(omega);
    const Eigen::Matrix3f Omega2 = Omega * Omega;
    const float sin_theta = std::sin(theta);
    const float cos_theta = std::cos(theta);
    R += (sin_theta / theta) * Omega +
         ((1.0f - cos_theta) / (theta * theta)) * Omega2;
    V += ((1.0f - cos_theta) / (theta * theta)) * Omega +
         ((theta - sin_theta) / (theta * theta * theta)) * Omega2;
  }

  const Eigen::Vector3f t = V * upsilon;

  Transform T = Transform::Identity();
  T.linear() = R;
  T.translation() = t;
  return T;
}

}  // namespace

namespace nvblox {

DepthToTsdfICP::DepthToTsdfICP(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(std::move(cuda_stream)),
      sphere_tracer_(cuda_stream_),
      depth_buffer_(MemoryType::kDevice),
      synthetic_depth_full_res_(MemoryType::kDevice),
      vertices_full_res_obs_L_(MemoryType::kDevice),
      vertices_full_res_syn_L_(MemoryType::kDevice),
      normals_full_res_obs_L_(MemoryType::kDevice),
      normals_full_res_syn_L_(MemoryType::kDevice),
      normal_matrix_buffer_(MemoryType::kUnified),
      normal_vector_buffer_(MemoryType::kUnified),
      residual_buffer_(MemoryType::kUnified),
      count_buffer_(MemoryType::kUnified) {}

void DepthToTsdfICP::setConfig(const Config& config) { config_ = config; }

bool DepthToTsdfICP::refinePoseAndMask(
    const MaskedDepthImageConstView& depth_frame, const Camera& camera,
    const TsdfLayer& tsdf_layer, const float voxel_size_m,
    const float truncation_distance_m, Transform* T_L_C,
    MaskedDepthImageConstView* masked_depth_view,
    DepthToTsdfIcpResult* result) {
  CHECK_NOTNULL(T_L_C);
  CHECK_NOTNULL(masked_depth_view);
  DepthToTsdfIcpResult local_result;
  DepthToTsdfIcpResult* stats =
      (result != nullptr) ? result : &local_result;
  stats->converged = false;
  stats->inlier_ratio = 0.0f;
  stats->mean_abs_residual_m = 0.0f;

  if (depth_frame.rows() == 0 || depth_frame.cols() == 0) {
    std::cout << "[DepthToTsdfICP] Reject: empty depth frame (rows="
              << depth_frame.rows() << ", cols=" << depth_frame.cols() << ")"
              << std::endl;
    return false;
  }

  // Copy the input depth to our working buffer.
  depth_buffer_.copyFromAsync(depth_frame, *cuda_stream_);

  // Apply the incoming mask by zeroing inactive pixels.
  if (depth_frame.hasMask()) {
    DepthImageView depth_view(depth_buffer_);
    ImageView<const uint8_t> mask_view = depth_frame.mask();
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_view.cols() + threads.x - 1) / threads.x,
                      (depth_view.rows() + threads.y - 1) / threads.y, 1);
    applyMaskKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        depth_view, mask_view, depth_frame.mode());
    checkCudaErrors(cudaPeekAtLastError());
  }

  const DepthImageConstView full_res_depth(depth_buffer_);

  // Determine the pyramid of subsampling factors (coarse to fine).
  std::vector<int> requested = config_.pyramid_subsampling_factors;
  requested.erase(std::remove_if(requested.begin(), requested.end(),
                                 [](int f) { return f <= 0; }),
                  requested.end());
  if (requested.empty()) {
    requested.push_back(1);
  }
  std::sort(requested.begin(), requested.end());
  std::reverse(requested.begin(), requested.end());

  std::vector<int> subsampling_factors;
  subsampling_factors.reserve(requested.size());
  for (const int factor : requested) {
    if (depth_frame.rows() % factor == 0 &&
        depth_frame.cols() % factor == 0) {
      subsampling_factors.push_back(factor);
    }
  }
  if (subsampling_factors.empty() ||
      subsampling_factors.back() != 1) {
    subsampling_factors.push_back(1);
  }

  ensureLevelCapacity(subsampling_factors.size());

  // Pre-compute the observed vertex maps (camera frame) for each level.
  for (size_t i = 0; i < subsampling_factors.size(); ++i) {
    LevelBuffers& level = level_buffers_[i];
    level.subsampling = subsampling_factors[i];
    level.camera = scaledCamera(camera, level.subsampling);
    const int level_rows = depth_frame.rows() / level.subsampling;
    const int level_cols = depth_frame.cols() / level.subsampling;

    level.depth.resizeAsync(level_rows, level_cols, *cuda_stream_);
    DepthImageView level_depth_view(level.depth);

    if (level.subsampling == 1) {
      level.depth.copyFromAsync(depth_buffer_, *cuda_stream_);
    } else {
      const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
      const dim3 blocks((level_cols + threads.x - 1) / threads.x,
                        (level_rows + threads.y - 1) / threads.y, 1);
      downsampleDepthKernel<<<blocks, threads, 0, *cuda_stream_>>>(
          full_res_depth, level_depth_view, level.subsampling);
      checkCudaErrors(cudaPeekAtLastError());
    }

    level.vertices_obs_C.resizeAsync(level_rows, level_cols, *cuda_stream_);
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((level_cols + threads.x - 1) / threads.x,
                      (level_rows + threads.y - 1) / threads.y, 1);
    depthToVertexMapCameraKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        DepthImageConstView(level.depth), level.camera,
        ImageView<Vector3f>(level.vertices_obs_C));
    checkCudaErrors(cudaPeekAtLastError());
  }

  Transform current_T = *T_L_C;
  const float correspondence_depth_threshold =
      std::max(config_.overlap_depth_voxel_multiplier * voxel_size_m,
               config_.overlap_depth_min_m);
  const float huber_delta =
      std::max(0.5f * correspondence_depth_threshold, 1e-3f);

  bool any_valid_level = false;

  for (size_t i = 0; i < subsampling_factors.size(); ++i) {
    LevelBuffers& level = level_buffers_[i];

    // Render the synthetic depth for the current pose.
    sphere_tracer_.renderImageOnGPU(
        level.camera, current_T, tsdf_layer, truncation_distance_m,
        &level.synthetic_depth, MemoryType::kDevice, level.subsampling);

    const int level_rows = level.depth.rows();
    const int level_cols = level.depth.cols();

    level.vertices_mod_L.resizeAsync(level_rows, level_cols, *cuda_stream_);
    {
      const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
      const dim3 blocks((level_cols + threads.x - 1) / threads.x,
                        (level_rows + threads.y - 1) / threads.y, 1);
      depthToVertexMapWorldKernel<<<blocks, threads, 0, *cuda_stream_>>>(
          DepthImageConstView(level.synthetic_depth), level.camera,
          current_T, ImageView<Vector3f>(level.vertices_mod_L));
      checkCudaErrors(cudaPeekAtLastError());
    }

    level.normals_mod_L.resizeAsync(level_rows, level_cols, *cuda_stream_);
    {
      const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
      const dim3 blocks((level_cols + threads.x - 1) / threads.x,
                        (level_rows + threads.y - 1) / threads.y, 1);
      computeNormalsKernel<<<blocks, threads, 0, *cuda_stream_>>>(
          ImageView<const Vector3f>(level.vertices_mod_L),
          ImageView<Vector3f>(level.normals_mod_L));
      checkCudaErrors(cudaPeekAtLastError());
    }

    const int iterations = config_.iterations_per_level.empty()
                               ? 4
                               : config_
                                     .iterations_per_level[std::min(
                                         i, config_.iterations_per_level.size() - 1)];

    bool level_has_valid_solution = false;
    for (int iter = 0; iter < iterations; ++iter) {
      normal_matrix_buffer_.resizeAsync(kIcpSystemSize * kIcpSystemSize,
                                        *cuda_stream_);
      normal_vector_buffer_.resizeAsync(kIcpSystemSize, *cuda_stream_);
      residual_buffer_.resizeAsync(1, *cuda_stream_);
      count_buffer_.resizeAsync(2, *cuda_stream_);

      normal_matrix_buffer_.setZeroAsync(*cuda_stream_);
      normal_vector_buffer_.setZeroAsync(*cuda_stream_);
      residual_buffer_.setZeroAsync(*cuda_stream_);
      count_buffer_.setZeroAsync(*cuda_stream_);

      const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
      const dim3 blocks((level_cols + threads.x - 1) / threads.x,
                        (level_rows + threads.y - 1) / threads.y, 1);
      accumulateNormalEquationsKernel<<<blocks, threads, 0, *cuda_stream_>>>(
          ImageView<const Vector3f>(level.vertices_obs_C),
          ImageView<const Vector3f>(level.vertices_mod_L),
          ImageView<const Vector3f>(level.normals_mod_L),
          DepthImageConstView(level.depth),
          DepthImageConstView(level.synthetic_depth), current_T,
          correspondence_depth_threshold, huber_delta,
          normal_matrix_buffer_.data(), normal_vector_buffer_.data(),
          residual_buffer_.data(), count_buffer_.data());
      checkCudaErrors(cudaPeekAtLastError());

      cuda_stream_->synchronize();

      const int total_correspondences = count_buffer_[0];
      const int inliers = count_buffer_[1];
      if (total_correspondences < kMinCorrespondenceCount) {
        std::cout << "[DepthToTsdfICP] Skip level (insufficient correspondences="
                  << total_correspondences << ") at level_subsampling="
                  << level.subsampling << ", iteration=" << iter << std::endl;
        break;
      }
      any_valid_level = true;
      level_has_valid_solution = true;

      const Eigen::Map<Eigen::Matrix<float, kIcpSystemSize, kIcpSystemSize>>
          JTJ_matrix(normal_matrix_buffer_.data());
      const Eigen::Matrix<float, kIcpSystemSize, kIcpSystemSize> JTJ_sym =
          0.5f * (JTJ_matrix + JTJ_matrix.transpose());
      const Eigen::Map<Eigen::Matrix<float, kIcpSystemSize, 1>> JTr(
          normal_vector_buffer_.data());

      Eigen::LDLT<Eigen::Matrix<float, kIcpSystemSize, kIcpSystemSize>> ldlt(
          JTJ_sym);
      if (ldlt.info() != Eigen::Success) {
        std::cout << "[DepthToTsdfICP] Reject: normal equation solve failed at "
                     "level_subsampling="
                  << level.subsampling << ", iteration=" << iter << std::endl;
        return false;
      }
      Eigen::Matrix<float, kIcpSystemSize, 1> delta =
          ldlt.solve(-JTr);

      const float step_norm = delta.norm();
      if (step_norm > config_.max_step_norm &&
          step_norm > 1e-6f) {
        delta *= config_.max_step_norm / step_norm;
      }

      const Transform delta_T = se3Exp(delta);
      current_T = delta_T * current_T;

      const float sum_abs_residuals = residual_buffer_[0];
      stats->mean_abs_residual_m =
          sum_abs_residuals / static_cast<float>(total_correspondences);
      stats->inlier_ratio = static_cast<float>(inliers) /
                            static_cast<float>(total_correspondences);
    }
    if (!level_has_valid_solution) {
      continue;
    }
  }

  if (!any_valid_level) {
    std::cout << "[DepthToTsdfICP] Reject: no pyramid level produced valid "
                 "correspondences"
              << std::endl;
    return false;
  }

  if (stats->inlier_ratio < config_.min_inlier_ratio ||
      stats->mean_abs_residual_m > config_.max_mean_residual_m) {
    std::cout << "[DepthToTsdfICP] Reject: metrics out of bounds (inlier_ratio="
              << stats->inlier_ratio
              << ", min_required=" << config_.min_inlier_ratio
              << ", mean_residual_m=" << stats->mean_abs_residual_m
              << ", max_allowed=" << config_.max_mean_residual_m << ")"
              << std::endl;
    return false;
  }

  *T_L_C = current_T;
  stats->converged = true;

  // Render a full-resolution synthetic depth with the refined pose for masking.
  sphere_tracer_.renderImageOnGPU(camera, current_T, tsdf_layer,
                                  truncation_distance_m,
                                  &synthetic_depth_full_res_,
                                  MemoryType::kDevice, 1);

  // Build vertex and normal maps for observed depth at full resolution.
  vertices_full_res_obs_L_.resizeAsync(depth_frame.rows(), depth_frame.cols(),
                                       *cuda_stream_);
  {
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_frame.cols() + threads.x - 1) / threads.x,
                      (depth_frame.rows() + threads.y - 1) / threads.y, 1);
    depthToVertexMapWorldKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        full_res_depth, camera, current_T,
        ImageView<Vector3f>(vertices_full_res_obs_L_));
    checkCudaErrors(cudaPeekAtLastError());
  }

  normals_full_res_obs_L_.resizeAsync(depth_frame.rows(), depth_frame.cols(),
                                      *cuda_stream_);
  {
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_frame.cols() + threads.x - 1) / threads.x,
                      (depth_frame.rows() + threads.y - 1) / threads.y, 1);
    computeNormalsKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        ImageView<const Vector3f>(vertices_full_res_obs_L_),
        ImageView<Vector3f>(normals_full_res_obs_L_));
    checkCudaErrors(cudaPeekAtLastError());
  }

  // Build vertex and normal maps for synthetic full-resolution depth.
  vertices_full_res_syn_L_.resizeAsync(depth_frame.rows(), depth_frame.cols(),
                                       *cuda_stream_);
  {
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_frame.cols() + threads.x - 1) / threads.x,
                      (depth_frame.rows() + threads.y - 1) / threads.y, 1);
    depthToVertexMapWorldKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        DepthImageConstView(synthetic_depth_full_res_), camera, current_T,
        ImageView<Vector3f>(vertices_full_res_syn_L_));
    checkCudaErrors(cudaPeekAtLastError());
  }

  normals_full_res_syn_L_.resizeAsync(depth_frame.rows(), depth_frame.cols(),
                                      *cuda_stream_);
  {
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_frame.cols() + threads.x - 1) / threads.x,
                      (depth_frame.rows() + threads.y - 1) / threads.y, 1);
    computeNormalsKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        ImageView<const Vector3f>(vertices_full_res_syn_L_),
        ImageView<Vector3f>(normals_full_res_syn_L_));
    checkCudaErrors(cudaPeekAtLastError());
  }

  const float cos_normal_threshold =
      std::cos(config_.overlap_normal_threshold_deg * kDegToRad);
  {
    const dim3 threads(kThreadsPerDim, kThreadsPerDim, 1);
    const dim3 blocks((depth_frame.cols() + threads.x - 1) / threads.x,
                      (depth_frame.rows() + threads.y - 1) / threads.y, 1);
    maskOverlappingPixelsKernel<<<blocks, threads, 0, *cuda_stream_>>>(
        DepthImageView(depth_buffer_),
        DepthImageConstView(synthetic_depth_full_res_),
        ImageView<const Vector3f>(normals_full_res_obs_L_),
        ImageView<const Vector3f>(normals_full_res_syn_L_),
        correspondence_depth_threshold, cos_normal_threshold);
    checkCudaErrors(cudaPeekAtLastError());
  }

  *masked_depth_view = MaskedDepthImageConstView(
      depth_buffer_,
      depth_frame.hasMask()
          ? std::optional<ImageView<const uint8_t>>(depth_frame.mask())
          : std::nullopt,
      depth_frame.mode());

  return true;
}

void DepthToTsdfICP::ensureLevelCapacity(const size_t required_levels) {
  if (required_levels <= level_buffers_.size()) {
    return;
  }
  level_buffers_.resize(required_levels);
}

}  // namespace nvblox
