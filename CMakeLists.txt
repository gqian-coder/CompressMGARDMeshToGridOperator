cmake_minimum_required(VERSION 3.21)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Enable HIP language support if requested
option(ENABLE_HIP "Enable HIP (AMD GPU) support" OFF)
option(ENABLE_CUDA "Enable CUDA (NVIDIA GPU) support" OFF)

if(ENABLE_HIP)
  # For HIP on AMD GPUs
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
  endif()
  
  # Set HIP as a project language
  project(CompressMGARDMeshToGridOperator C CXX HIP)
  
  # Find HIP package
  find_package(hip REQUIRED)
  
  message(STATUS "HIP enabled: ROCm path = ${ROCM_PATH}")
  message(STATUS "HIP compiler: ${CMAKE_HIP_COMPILER}")
  
elseif(ENABLE_CUDA)
  # For CUDA on NVIDIA GPUs
  project(CompressMGARDMeshToGridOperator C CXX CUDA)
  
  find_package(CUDAToolkit REQUIRED)
  
  message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
  
else()
  # CPU-only build
  project(CompressMGARDMeshToGridOperator C CXX)
  message(STATUS "Building CPU-only version (no GPU acceleration)")
endif()

find_package(ADIOS2 REQUIRED)
find_package(MGARD REQUIRED)
set(BUILD_SHARED_LIBS ON)

# Source files
set(SOURCES CompressMGARDMeshToGridOperator.cpp)
set(HEADERS 
    CompressMGARDMeshToGridOperator.h
    CompressMGARDMeshToGridOperator_GPU.hpp
)

if(ENABLE_HIP)
  # For HIP, we need to compile the .cpp as HIP source
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)
  
  add_library(CompressMGARDMeshToGridOperator ${SOURCES} ${HEADERS})
  
  target_link_libraries(CompressMGARDMeshToGridOperator 
    PRIVATE
      hip::device
    PUBLIC
      adios2::cxx11 
      adios2::core 
      mgard::mgard
  )
  
  # Set HIP architecture (MI250X on Frontier uses gfx90a)
  set(GPU_TARGETS "gfx90a" CACHE STRING "GPU architectures to compile for")
  set_target_properties(CompressMGARDMeshToGridOperator PROPERTIES
    HIP_ARCHITECTURES "${GPU_TARGETS}"
  )
  
  target_compile_definitions(CompressMGARDMeshToGridOperator PRIVATE __HIP_PLATFORM_AMD__ ENABLE_HIP)

elseif(ENABLE_CUDA)
  # For CUDA
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
  
  add_library(CompressMGARDMeshToGridOperator ${SOURCES} ${HEADERS})
  
  target_link_libraries(CompressMGARDMeshToGridOperator 
    PRIVATE
      CUDA::cudart
    PUBLIC
      adios2::cxx11 
      adios2::core 
      mgard::mgard
  )
  
  # Set CUDA architecture (adjust as needed for your GPU)
  set(CMAKE_CUDA_ARCHITECTURES "70;80" CACHE STRING "CUDA architectures to compile for")
  set_target_properties(CompressMGARDMeshToGridOperator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  
  target_compile_definitions(CompressMGARDMeshToGridOperator PRIVATE ENABLE_CUDA)

else()
  # CPU-only build
  add_library(CompressMGARDMeshToGridOperator ${SOURCES} ${HEADERS})
  target_link_libraries(CompressMGARDMeshToGridOperator 
    PUBLIC
      adios2::cxx11 
      adios2::core 
      mgard::mgard
  )
endif()

include(GenerateExportHeader)
generate_export_header(CompressMGARDMeshToGridOperator BASE_NAME compress_mgard_mesh_to_grid)
target_include_directories(CompressMGARDMeshToGridOperator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

# Print configuration summary
message(STATUS "=== CompressMGARDMeshToGridOperator Configuration ===")
message(STATUS "  HIP support: ${ENABLE_HIP}")
message(STATUS "  CUDA support: ${ENABLE_CUDA}")
message(STATUS "  ADIOS2: ${ADIOS2_DIR}")
message(STATUS "  MGARD: ${MGARD_DIR}")
message(STATUS "=====================================================")
