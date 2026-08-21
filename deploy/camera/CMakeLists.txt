cmake_minimum_required(VERSION 3.16)
project(camera_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  DEPLOY_CAMERA_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  CACHE INTERNAL
  "Absolute path to deploy/camera"
)

add_subdirectory(d435i)
add_subdirectory(viewer)
