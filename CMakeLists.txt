cmake_minimum_required(VERSION 3.24)
project(practice-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
)
FetchContent_MakeAvailable(doctest)