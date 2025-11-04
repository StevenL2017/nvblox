if(TARGET meshoptimizer)
  return()
endif()

set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(MESHOPT_INSTALL OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  meshoptimizer
  SYSTEM
  URL https://github.com/zeux/meshoptimizer/archive/refs/tags/v0.25.tar.gz
  URL_HASH MD5=376039e0b37dd9b451b463bc323f7206)

FetchContent_MakeAvailable(meshoptimizer)

set_target_properties(meshoptimizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_nvblox_compiler_options_nowarnings(meshoptimizer)
