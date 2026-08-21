cmake_minimum_required(VERSION 3.16)
project(depth_camera_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")

find_package(OpenCV REQUIRED COMPONENTS core highgui)
find_package(Threads REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_program(FFMPEG_EXECUTABLE ffmpeg REQUIRED)

set(DEPLOY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
if(NOT DEFINED DEPLOY_CAMERA_SOURCE_DIR)
  get_filename_component(
    DEPLOY_CAMERA_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    ABSOLUTE
  )
endif()

add_executable(depth_camera_viewer
  src/main.cpp
  src/ViewerConfig.cpp
  src/DepthViewer.cpp
  src/RecordSwitchSubscriber.cpp
  src/WirelessControllerSubscriber.cpp
  src/FfmpegRecorder.cpp
  ${DEPLOY_ROOT}/robots/go2/src/DepthObservation_.cpp
)

target_include_directories(depth_camera_viewer PRIVATE
  include
  ${DEPLOY_ROOT}/include
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(depth_camera_viewer PRIVATE
  unitree_sdk2
  yaml-cpp
  ddsc
  ddscxx
  rt
  Threads::Threads
  ${OpenCV_LIBS}
)

target_compile_definitions(depth_camera_viewer PRIVATE
  FFMPEG_EXECUTABLE_PATH="${FFMPEG_EXECUTABLE}"
  DEPLOY_CAMERA_ROOT_PATH="${DEPLOY_CAMERA_SOURCE_DIR}"
  VIEWER_CONFIG_PATH="${DEPLOY_CAMERA_SOURCE_DIR}/viewer/config/config.yaml"
)

target_compile_options(depth_camera_viewer PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
