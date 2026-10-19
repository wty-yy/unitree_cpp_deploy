cmake_minimum_required(VERSION 3.12)
project(g1_29dof_controller)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The G1 controller requires Linux; detected ${CMAKE_SYSTEM_NAME}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy ${CMAKE_BINARY_DIR}/cnpy)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an extracted ONNX Runtime package")
if(NOT ONNXRUNTIME_ROOT)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ONNXRUNTIME_ARCH)
  if(ONNXRUNTIME_ARCH MATCHES "^(aarch64|arm64)$")
    set(ONNXRUNTIME_PACKAGE "onnxruntime-linux-aarch64-gpu-1.24.0")
  elseif(ONNXRUNTIME_ARCH MATCHES "^(x86_64|amd64)$")
    set(ONNXRUNTIME_PACKAGE "onnxruntime-linux-x64-gpu-1.24.2")
  else()
    message(FATAL_ERROR
      "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. "
      "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
  endif()
  set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/${ONNXRUNTIME_PACKAGE}")
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE
  BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h"
   OR NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
  message(FATAL_ERROR
    "ONNX Runtime headers or library not found in ${ONNXRUNTIME_ROOT}. "
    "Extract the matching package there or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
endif()
message(STATUS "ONNX Runtime (${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}): ${ONNXRUNTIME_ROOT}")

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy
  "${ONNXRUNTIME_INCLUDE_DIR}"
  ${PROJECT_SOURCE_DIR}/../../thirdparty/
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  cnpy
  "${ONNXRUNTIME_LIBRARY}"
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)
