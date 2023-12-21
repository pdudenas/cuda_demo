cmake_minimum_required(VERSION 3.14)
project(pybind_cuda)
find_package(CUDA REQUIRED)
set(Python_EXECUTABLE /home/pjd/lib/mambaforge/envs/cyrsoxs-build/bin/python)
set(Python_INCLUDE_DIR /home/pjd/lib/mambaforge/envs/cyrsoxs-build/include/python3.11)
set(Python_LIBRARIES /home/pjd/lib/mambaforge/envs/cyrsoxs-build/lib/libpython3.11.so)
find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    )

link_directories(
    /usr/local/cuda/lib64/
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(demo SHARED
  pykernel.cu)

target_link_libraries(demo
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(demo PROPERTIES PREFIX "")
