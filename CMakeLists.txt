cmake_minimum_required(VERSION 3.10)
project(soft354_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE soft354_cuda)
set(LIBRARY CannyLib)

find_path(LODEPNG_INCLUDE_DIR lodepng.h)
include_directories(${LODEPNG_INCLUDE_DIR})
find_library(LODEPNG_LIB lodepng REQUIRED)

find_package(GTest MODULE REQUIRED)

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(test)

configure_file(img/tiger.png img/tiger.png COPYONLY)

add_executable(${EXECUTABLE} main.cpp)
target_link_libraries(${EXECUTABLE} ${LIBRARY})
target_link_libraries(${EXECUTABLE} ${LODEPNG_LIB})
