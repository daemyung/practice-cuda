cmake_minimum_required(VERSION 3.24)
project(practice-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
)
FetchContent_MakeAvailable(doctest)

add_library(test-kit
    test/test-kit.cuh
    test/test-kit.cu
)

target_link_libraries(test-kit
PUBLIC
    cublas
)

add_executable(cublas-test
    test/cublas-test.cu
)

target_link_libraries(cublas-test
PRIVATE
    doctest
    test-kit
)