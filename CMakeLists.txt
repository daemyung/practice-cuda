cmake_minimum_required(VERSION 3.24)
project(practice-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
)
FetchContent_MakeAvailable(doctest)

add_executable(cublas-test
    test/cublas-test.cu
)

target_link_libraries(cublas-test
PRIVATE
    cublas
    doctest
)