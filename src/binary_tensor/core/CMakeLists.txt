cmake_minimum_required(VERSION 3.8.0)

file(GLOB BinaryTensor_inc "*.hh")

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
enable_language(CUDA C CXX)

set(CMAKE_CUDA_ARCHITECTURES 52 75 89)
set(CMAKE_CUDA_SEPARABLE_COMPILATION TRUE)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--default-stream per-thread")

file(GLOB BinaryTensor_src "dtype/*.cu" "dtype/*.cc" "*.cc" "*.cu")

# file(MAKE_DIRECTORY "include/tensor_array/core")

add_library(binarytensor_core SHARED ${BinaryTensor_src})
target_link_libraries(binarytensor_core PRIVATE CUDA::cudart)

endif(CUDAToolkit_FOUND)

add_library(BinaryTensor::Core ALIAS binarytensor_core)

install(
    TARGETS binarytensor_core
    EXPORT BinaryTensorTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT libraries)
