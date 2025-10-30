# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# pylint: disable=protected-access
from nvblox_torch.lib.utils import get_nvblox_torch_class
from typing import Optional, Type, Any, no_type_check

# // NOTE(alexmillane, 2024.11.14): The following sub-parameter structs are currently unwrapped.
# // If you need them wrapped, ask alex.
# // Unwrapped sub-parameter classes:
# // - ViewCalculatorParams view_calculator_params;
# // - OccupancyIntegratorParams occupancy_integrator_params;
# // - OccupancyDecayIntegratorParams occupancy_decay_integrator_params;
# // - FreespaceIntegratorParams freespace_integrator_params;


# noqa
class NvbloxParameterClass:
    """NvbloxParameterClass is a base class for Nvblox parameter classes."""

    def __init__(self) -> None:
        """Constructor that does nothing."""
        pass

    def wrap_getter_and_setters(self, parameter_class: Type, c_param_struct: Any) -> None:
        """Wrap the getter and setter methods of the C++ parameter struct."""
        attribute_names = [
            method_name[len('get_'):] for method_name in c_param_struct._method_names()
            if method_name.startswith('get')
        ]
        for name in attribute_names:

            # Create a getter function
            def getter(_: Any, name: str = name) -> object:
                print(f'Getting: {name}')
                getter_name = 'get_' + name
                method = getattr(c_param_struct, getter_name)
                return method()

            # Create a setter function
            def setter(_: Any, value: object, name: str = name) -> None:
                setter_name = 'set_' + name
                method = getattr(c_param_struct, setter_name)
                method(value)

            # Add an attribute to the class
            setattr(parameter_class, name, property(getter, setter))


@no_type_check
class ProjectiveIntegratorParams(NvbloxParameterClass):
    """Parameters governing the projective integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('ProjectiveIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(ProjectiveIntegratorParams, self._c_params)


class MeshIntegratorParams(NvbloxParameterClass):
    """Parameters governing the mesh integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MeshIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(MeshIntegratorParams, self._c_params)


class MeshOptimizerParams(NvbloxParameterClass):
    """Parameters governing mesh post-processing and simplification."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MeshOptimizerParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(MeshOptimizerParams, self._c_params)


class DecayIntegratorBaseParams(NvbloxParameterClass):
    """Base parameters for the decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('DecayIntegratorBaseParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(DecayIntegratorBaseParams, self._c_params)


class TsdfDecayIntegratorParams(NvbloxParameterClass):
    """Parameters governing the TSDF decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('TsdfDecayIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(TsdfDecayIntegratorParams, self._c_params)


class OccupancyDecayIntegratorParams(NvbloxParameterClass):
    """Parameters governing the occupancy decay integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('OccupancyDecayIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(OccupancyDecayIntegratorParams, self._c_params)


class EsdfIntegratorParams(NvbloxParameterClass):
    """Parameters governing the ESDF integrator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('EsdfIntegratorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(EsdfIntegratorParams, self._c_params)


class ViewCalculatorParams(NvbloxParameterClass):
    """Parameters governing the view calculator."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('ViewCalculatorParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(ViewCalculatorParams, self._c_params)


class BlockMemoryPoolParams(NvbloxParameterClass):
    """Parameters governing memory allocation."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('BlockMemoryPoolParams')()
        else:
            self._c_params = c_params
        self.wrap_getter_and_setters(BlockMemoryPoolParams, self._c_params)


class MapperParams:
    """MapperParams is a class that wraps the C++ MapperParams class."""

    def __init__(self, c_params: Optional[object] = None) -> None:
        """Construct from C++ object."""
        if c_params is None:
            self._c_params = get_nvblox_torch_class('MapperParams')()
        else:
            self._c_params = c_params
        # NOTE: We don't call the automatic wrapping function here because we
        # need to convert the subclasses to python objects manually.

    @property
    def do_depth_preprocessing(self) -> bool:
        """Whether to preprocess input depth images."""
        return bool(self._c_params.get_do_depth_preprocessing())

    @do_depth_preprocessing.setter
    def do_depth_preprocessing(self, value: bool) -> None:
        self._c_params.set_do_depth_preprocessing(bool(value))

    @property
    def clear_unobserved_blocks_in_fov(self) -> bool:
        """Whether to clear unobserved TSDF blocks in the current frustum."""
        return bool(self._c_params.get_clear_unobserved_blocks_in_fov())

    @clear_unobserved_blocks_in_fov.setter
    def clear_unobserved_blocks_in_fov(self, value: bool) -> None:
        self._c_params.set_clear_unobserved_blocks_in_fov(bool(value))

    @property
    def filter_small_tsdf_block_updates(self) -> bool:
        """Whether to filter unchanged TSDF blocks when reporting updates."""
        return bool(self._c_params.get_filter_small_tsdf_block_updates())

    @filter_small_tsdf_block_updates.setter
    def filter_small_tsdf_block_updates(self, value: bool) -> None:
        self._c_params.set_filter_small_tsdf_block_updates(bool(value))

    @property
    def tsdf_filter_zc_ratio_epsilon(self) -> float:
        """Zero-crossing relative change threshold for block filtering."""
        return float(self._c_params.get_tsdf_filter_zc_ratio_epsilon())

    @tsdf_filter_zc_ratio_epsilon.setter
    def tsdf_filter_zc_ratio_epsilon(self, value: float) -> None:
        self._c_params.set_tsdf_filter_zc_ratio_epsilon(float(value))

    @property
    def tsdf_filter_iou_tolerance(self) -> float:
        """IoU tolerance for the near-zero mask comparison."""
        return float(self._c_params.get_tsdf_filter_iou_tolerance())

    @tsdf_filter_iou_tolerance.setter
    def tsdf_filter_iou_tolerance(self, value: float) -> None:
        self._c_params.set_tsdf_filter_iou_tolerance(float(value))

    @property
    def tsdf_filter_l1_q75_threshold(self) -> float:
        """Distance change threshold for TSDF block update filtering."""
        return float(self._c_params.get_tsdf_filter_l1_q75_threshold())

    @tsdf_filter_l1_q75_threshold.setter
    def tsdf_filter_l1_q75_threshold(self, value: float) -> None:
        self._c_params.set_tsdf_filter_l1_q75_threshold(float(value))

    @property
    def tsdf_filter_near_zero_band_m(self) -> float:
        """Half-width of the near-zero band used for filtering."""
        return float(self._c_params.get_tsdf_filter_near_zero_band_m())

    @tsdf_filter_near_zero_band_m.setter
    def tsdf_filter_near_zero_band_m(self, value: float) -> None:
        self._c_params.set_tsdf_filter_near_zero_band_m(float(value))

    @property
    def tsdf_filter_min_weight(self) -> float:
        """Weight change threshold for TSDF block update filtering."""
        return float(self._c_params.get_tsdf_filter_min_weight())

    @tsdf_filter_min_weight.setter
    def tsdf_filter_min_weight(self, value: float) -> None:
        self._c_params.set_tsdf_filter_min_weight(float(value))

    def get_projective_integrator_params(self) -> ProjectiveIntegratorParams:
        """Parameter getter."""
        return ProjectiveIntegratorParams(self._c_params.get_projective_integrator_params())

    def set_projective_integrator_params(self, params: ProjectiveIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_projective_integrator_params(params._c_params)

    def get_mesh_integrator_params(self) -> MeshIntegratorParams:
        """Parameter getter."""
        return MeshIntegratorParams(self._c_params.get_mesh_integrator_params())

    def set_mesh_integrator_params(self, params: MeshIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_mesh_integrator_params(params._c_params)

    def get_mesh_optimizer_params(self) -> MeshOptimizerParams:
        """Parameter getter."""
        return MeshOptimizerParams(self._c_params.get_mesh_optimizer_params())

    def set_mesh_optimizer_params(self, params: MeshOptimizerParams) -> None:
        """Parameter setter."""
        return self._c_params.set_mesh_optimizer_params(params._c_params)

    def get_decay_integrator_base_params(self) -> DecayIntegratorBaseParams:
        """Parameter getter."""
        return DecayIntegratorBaseParams(self._c_params.get_decay_integrator_base_params())

    def set_decay_integrator_base_params(self, params: DecayIntegratorBaseParams) -> None:
        """Parameter setter."""
        return self._c_params.set_decay_integrator_base_params(params._c_params)

    def get_tsdf_decay_integrator_params(self) -> TsdfDecayIntegratorParams:
        """Parameter getter."""
        return TsdfDecayIntegratorParams(self._c_params.get_tsdf_decay_integrator_params())

    def set_tsdf_decay_integrator_params(self, params: TsdfDecayIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_tsdf_decay_integrator_params(params._c_params)

    def get_occupancy_decay_integrator_params(self) -> OccupancyDecayIntegratorParams:
        """Parameter getter."""
        return OccupancyDecayIntegratorParams(
            self._c_params.get_occupancy_decay_integrator_params())

    def set_occupancy_decay_integrator_params(self, params: OccupancyDecayIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_occupancy_decay_integrator_params(params._c_params)

    def get_esdf_integrator_params(self) -> EsdfIntegratorParams:
        """Parameter getter."""
        return EsdfIntegratorParams(self._c_params.get_esdf_integrator_params())

    def set_esdf_integrator_params(self, params: EsdfIntegratorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_esdf_integrator_params(params._c_params)

    def get_view_calculator_params(self) -> ViewCalculatorParams:
        """Parameter getter."""
        return ViewCalculatorParams(self._c_params.get_view_calculator_params())

    def set_view_calculator_params(self, params: ViewCalculatorParams) -> None:
        """Parameter setter."""
        return self._c_params.set_view_calculator_params(params._c_params)

    def get_block_memory_pool_params(self) -> BlockMemoryPoolParams:
        """Parameter getter."""
        return BlockMemoryPoolParams(self._c_params.get_block_memory_pool_params())

    def set_block_memory_pool_params(self, params: BlockMemoryPoolParams) -> None:
        """Parameter setter."""
        return self._c_params.set_block_memory_pool_params(params._c_params)
