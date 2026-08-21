cmake_minimum_required(VERSION 3.16)
project(d435i_depth_publisher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")

find_package(Threads REQUIRED)
find_package(realsense2 REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(DEPLOY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if(NOT DEFINED DEPLOY_CAMERA_SOURCE_DIR)
  get_filename_component(
    DEPLOY_CAMERA_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
  )
endif()

add_executable(d435i_depth_publisher
  src/main.cpp
  src/Config.cpp
  src/D435iCamera.cpp
  src/DepthPublisher.cpp
  ${DEPLOY_ROOT}/robots/go2/src/DepthObservation_.cpp
)

target_include_directories(d435i_depth_publisher PRIVATE
  include
  ${DEPLOY_ROOT}/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
)

target_link_libraries(d435i_depth_publisher PRIVATE
  realsense2::realsense2
  unitree_sdk2
  yaml-cpp
  ddsc
  ddscxx
  rt
  Threads::Threads
)

target_compile_options(d435i_depth_publisher PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

target_compile_definitions(d435i_depth_publisher PRIVATE
  D435I_CONFIG_PATH="${DEPLOY_CAMERA_SOURCE_DIR}/d435i/config/config.yaml"
)
