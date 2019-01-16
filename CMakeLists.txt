cmake_minimum_required(VERSION 3.12)
project(soft354_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(test)

add_executable(soft354_cuda main.cpp)
