# soft354-cuda
CUDA half of the SOFT354 assignment

- Make sure the CUDACXX env var is set to the location of nvcc.
  - This can be found by running `which nvcc` if nvcc is already on your PATH.
- Make sure vcpkg is set up, and that cmake is running with `-DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake`

Install the following packages:
- gtest