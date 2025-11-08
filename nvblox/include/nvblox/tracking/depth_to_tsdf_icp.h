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

#include <memory>
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/map/common_names.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// Metrics describing the outcome of the depth-to-TSDF ICP refinement.
struct DepthToTsdfIcpResult {
  /// Whether Gauss-Newton converged and produced a usable pose.
  bool converged = false;
  /// Ratio between correspondences considered inliers and all candidates.
  float inlier_ratio = 0.0f;
  /// Mean absolute point-to-plane residual in meters.
  float mean_abs_residual_m = 0.0f;
};

/// Lightweight tracker that aligns an incoming depth frame against the
/// currently fused TSDF using projective ICP directly on the GPU. The aligned
/// frame is then masked such that overlapping pixels do not get re-integrated.
class DepthToTsdfICP {
 public:
  struct Config {
    /// Pyramid subsampling factors, processed from coarse to fine.
    std::vector<int> pyramid_subsampling_factors = {4, 2, 1};
    /// ICP iteration budget per level (same order as factors).
    std::vector<int> iterations_per_level = {6, 4, 4};
    /// Point-to-plane Huber loss width (meters).
    float point_to_plane_huber_delta_m = 0.02f;
    /// Point-to-point Huber loss width (meters).
    float point_to_point_huber_delta_m = 0.02f;
    /// Use point-to-point residuals for pyramid levels whose subsampling factor
    /// is greater-or-equal to this value (set <= 1 to disable point-to-point).
    int point_to_point_min_subsampling = 1;
    /// Maximum accepted mean absolute residual (meters).
    float max_mean_residual_m = 0.05f;
    /// Minimum fraction of inliers required to accept the result.
    float min_inlier_ratio = 0.15f;
    /// Maximum norm of the 6-DoF update (rad/ meters) per Gauss-Newton step.
    float max_step_norm = 0.15f;
    /// Depth threshold multiplier relative to voxel size used when masking
    /// overlapping pixels as well as for correspondence rejection.
    float overlap_depth_voxel_multiplier = 2.5f;
    /// Minimum absolute overlap threshold (meters).
    float overlap_depth_min_m = 0.01f;
    /// Maximum angle in degrees between observed/model normals for masking.
    float overlap_normal_threshold_deg = 15.0f;
  };

  explicit DepthToTsdfICP(std::shared_ptr<CudaStream> cuda_stream);

  DepthToTsdfICP(const DepthToTsdfICP&) = delete;
  DepthToTsdfICP& operator=(const DepthToTsdfICP&) = delete;

  const Config& config() const { return config_; }
  void setConfig(const Config& config);

  /// Performs pose refinement and overlap masking. On success the supplied
  /// pose is updated in-place, a depth view with invalidated overlaps is
  /// returned and metrics are populated.
  /// @param depth_frame Input depth frame (device memory).
  /// @param camera Camera intrinsics for the frame.
  /// @param tsdf_layer Reference TSDF layer to raycast against.
  /// @param voxel_size_m Mapper voxel size (used for thresholds).
  /// @param[in,out] T_L_C Pose to refine (updated on success).
  /// @param[out] masked_depth_view Depth view referencing internal storage.
  /// @param[out] result Optional refinement metrics.
  /// @return True if ICP converged and masking succeeded.
  bool refinePoseAndMask(const MaskedDepthImageConstView& depth_frame,
                         const Camera& camera, const TsdfLayer& tsdf_layer,
                         float voxel_size_m, float truncation_distance_m,
                         Transform* T_L_C,
                         MaskedDepthImageConstView* masked_depth_view,
                         DepthToTsdfIcpResult* result);

 private:
  struct LevelBuffers {
    LevelBuffers()
        : depth(MemoryType::kDevice),
          synthetic_depth(MemoryType::kDevice),
          vertices_obs_C(MemoryType::kDevice),
          vertices_mod_L(MemoryType::kDevice),
          normals_mod_L(MemoryType::kDevice) {}
    int subsampling = 1;
    Camera camera;
    DepthImage depth;
    DepthImage synthetic_depth;
    Image<Vector3f> vertices_obs_C;
    Image<Vector3f> vertices_mod_L;
    Image<Vector3f> normals_mod_L;
  };

  std::shared_ptr<CudaStream> cuda_stream_;
  SphereTracer sphere_tracer_;
  Config config_;

  DepthImage depth_buffer_;
  DepthImage synthetic_depth_full_res_;

  std::vector<LevelBuffers> level_buffers_;

  Image<Vector3f> vertices_full_res_obs_L_;
  Image<Vector3f> vertices_full_res_syn_L_;
  Image<Vector3f> normals_full_res_obs_L_;
  Image<Vector3f> normals_full_res_syn_L_;

  unified_vector<float> normal_matrix_buffer_;
  unified_vector<float> normal_vector_buffer_;
  unified_vector<float> residual_buffer_;
  unified_vector<int> count_buffer_;

  void ensureLevelCapacity(size_t required_levels);
};

}  // namespace nvblox
