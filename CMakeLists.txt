cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(CompressMGARDMeshToGridOperator C CXX)

find_package(ADIOS2 REQUIRED)
find_package(MGARD REQUIRED)
set(BUILD_SHARED_LIBS ON)

add_library(CompressMGARDMeshToGridOperator CompressMGARDMeshToGridOperator.cpp)
target_link_libraries(CompressMGARDMeshToGridOperator adios2::cxx11 adios2::core mgard::mgard)

include(GenerateExportHeader)
generate_export_header(CompressMGARDMeshToGridOperator BASE_NAME compress_mgard_mesh_to_grid)
target_include_directories(CompressMGARDMeshToGridOperator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>)
