#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

from enum import Enum
from typing import List, Optional

import torch

from nvblox_torch.lib.utils import get_nvblox_torch_class
from nvblox_torch.mapper_params import MapperParams
from nvblox_torch.mesh import BlockMesh, ColorMesh


class MappingType(str, Enum):
    """Mapping configuration used by MultiMapper."""

    STATIC_TSDF = "kStaticTsdf"
    STATIC_OCCUPANCY = "kStaticOccupancy"
    DYNAMIC = "kDynamic"
    HUMAN_WITH_STATIC_TSDF = "kHumanWithStaticTsdf"
    HUMAN_WITH_STATIC_OCCUPANCY = "kHumanWithStaticOccupancy"


class EsdfMode(str, Enum):
    """Target ESDF mode for MultiMapper."""

    MODE_3D = "k3D"
    MODE_2D = "k2D"
    UNSET = "kUnset"


class MultiMapper:
    """Python wrapper for the C++ nvblox MultiMapper."""

    def __init__(
        self,
        voxel_size_m: float,
        mapping_type: MappingType,
        esdf_mode: EsdfMode,
        background_params: Optional[MapperParams] = None,
        foreground_params: Optional[MapperParams] = None,
    ) -> None:
        if background_params is None:
            background_params = MapperParams()
        if foreground_params is not None:
            fg_params = foreground_params._c_params
        else:
            fg_params = None

        self._c_multi_mapper = get_nvblox_torch_class("MultiMapper")(
            float(voxel_size_m),
            mapping_type.value,
            esdf_mode.value,
            background_params._c_params,
            fg_params,
        )

    def add_depth_frame(
        self,
        depth_frame: torch.Tensor,
        t_w_c: torch.Tensor,
        intrinsics: torch.Tensor,
        update_time_ms: Optional[int] = None,
    ) -> None:
        """Integrate a depth frame using the dynamic/static split."""
        self._c_multi_mapper.integrate_depth(
            depth_frame, t_w_c, intrinsics, update_time_ms
        )

    def add_color_frame(
        self,
        color_frame: torch.Tensor,
        t_w_c: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> None:
        """Integrate a color frame into the static background map."""
        self._c_multi_mapper.integrate_color(color_frame, t_w_c, intrinsics)

    def update_color_mesh(self) -> None:
        """Update the mesh representation."""
        self._c_multi_mapper.update_color_mesh()

    def get_delta_block_mesh(self) -> List[BlockMesh]:
        """Fetch per-block mesh deltas from the background mapper."""
        c_block_meshes = self._c_multi_mapper.get_delta_block_mesh()
        return [BlockMesh(c_mesh=block) for block in c_block_meshes]

    def get_color_mesh(self) -> ColorMesh:
        """Fetch the full background mesh."""
        return ColorMesh(c_mesh=self._c_multi_mapper.get_color_mesh())

