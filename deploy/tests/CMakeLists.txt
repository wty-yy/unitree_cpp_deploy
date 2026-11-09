cmake_minimum_required(VERSION 3.12)
project(go2_bridge_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
enable_testing()

foreach(test IN ITEMS external_velocity policy_catalog)
  add_executable(${test}_test ${test}_test.cpp)
  target_include_directories(${test}_test PRIVATE ../include ../thirdparty)
  target_compile_options(${test}_test PRIVATE -UNDEBUG)
  target_link_libraries(${test}_test PRIVATE Threads::Threads yaml-cpp)
  add_test(NAME ${test} COMMAND ${test}_test)
endforeach()

add_test(NAME go2_policy_catalog COMMAND policy_catalog_test
  --go2-config ${CMAKE_CURRENT_SOURCE_DIR}/../robots/go2/config/config.yaml)
