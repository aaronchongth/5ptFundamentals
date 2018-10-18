cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# Must be C++14
set (CMAKE_CXX_STANDARD 14)

#-------------------------------------------------------------------------------
# Project setup
#-------------------------------------------------------------------------------
project(wayve_drivers C CXX)
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/bin")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

#-------------------------------------------------------------------------------
# OpenCV
#-------------------------------------------------------------------------------
set(OpenCV_BUILD_DIR "${CMAKE_SOURCE_DIR}/deps/opencv/build")
find_package(OpenCV 3.4.0 REQUIRED HINTS ${OpenCV_BUILD_DIR})

#-------------------------------------------------------------------------------
# Library targets
#-------------------------------------------------------------------------------
add_subdirectory(src/test_libs)

#-------------------------------------------------------------------------------
# Individual driver targets
#-------------------------------------------------------------------------------
# add_subdirectory(src/refactor_test)
