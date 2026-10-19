CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)

# Set default install prefix to source directory if not specified CMAKE_INSTALL_PREFIX
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Default install path" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(ENABLE_STATIC "Build static (.a) library" OFF)

find_package(ZLIB REQUIRED)

add_library(cnpy SHARED "cnpy.cpp")
target_include_directories(cnpy PUBLIC ${CMAKE_SOURCE_DIR} ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(ENABLE_STATIC)
    add_library(cnpy-static STATIC "cnpy.cpp")
    set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
    install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
endif(ENABLE_STATIC)

install(FILES "cnpy.h" DESTINATION include)

add_executable(example1 example1.cpp)
target_link_libraries(example1 cnpy)
