cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# Must be C++14
set (CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------------------------
# Project setup
#-------------------------------------------------------------------------------
project(5pt_fundamentals C CXX)
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
include_directories("${CMAKE_SOURCE_DIR}/src")

#-------------------------------------------------------------------------------
# CUDA (only if you want to)
#-------------------------------------------------------------------------------
# find_package(CUDA REQUIRED)

#-------------------------------------------------------------------------------
# OpenCV
#-------------------------------------------------------------------------------
set(OpenCV_BUILD_DIR "${CMAKE_SOURCE_DIR}/deps/opencv/build")
find_package(OpenCV 3.4.0 PATHS ${OpenCV_BUILD_DIR})
find_package(gflags REQUIRED) # SFM requirements

#-------------------------------------------------------------------------------
# Eigen
#-------------------------------------------------------------------------------
set(eigen_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rd_party/eigen")

#-------------------------------------------------------------------------------
# cxxopts
#-------------------------------------------------------------------------------
set(cxxopts_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rd_party/cxxopts/include")

#-------------------------------------------------------------------------------
# Library targets
#-------------------------------------------------------------------------------
add_subdirectory(src/data_handler)
add_subdirectory(src/utilities)
add_subdirectory(src/5pt_algo)
add_subdirectory(src/7pt_algo)

#-------------------------------------------------------------------------------
# Individual driver targets
#-------------------------------------------------------------------------------
add_subdirectory(src/image_viewer)
add_subdirectory(src/test_libs)
add_subdirectory(src/main)
add_subdirectory(src/mono_odom)