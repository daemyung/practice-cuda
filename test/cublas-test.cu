// MIT License
//
// Copyright (c) 2023 daemyung jang
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("[cublas] vector") {
    std::vector x{1.0f, 2.0f, 3.0f, 1.0f};
    float* y{nullptr};
    auto n = static_cast<int32_t>(x.size());

    cudaError err;
    cublasStatus_t stat;

    err = cudaMalloc((void**)&y, n * sizeof(decltype(x)::value_type));
    CHECK_EQ(err, cudaSuccess);

    stat = cublasSetVector(n, sizeof(decltype(x)::value_type), x.data(), 1, y, 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector z{0.0f, 0.0f, 0.0f, 0.0f};
    stat = cublasGetVector(n, sizeof(decltype(x)::value_type), y, 1, z.data(), 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(x, z);

    err = cudaFree(y);
    CHECK_EQ(err, cudaSuccess);
}

TEST_CASE("[cublas] matrix") {
    constexpr int32_t rows = 4;
    constexpr int32_t cols = 3;
    std::vector x{1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f};
    float* y{nullptr};

    cudaError err;
    cublasStatus_t stat;

    err = cudaMalloc((void**)&y, x.size() * sizeof(decltype(x)::value_type));
    CHECK_EQ(err, cudaSuccess);

    stat = cublasSetMatrix(rows, cols, sizeof(decltype(x)::value_type), x.data(), rows, y, rows);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector z{0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f};
    stat = cublasGetMatrix(rows, cols, sizeof(decltype(x)::value_type), y, rows, z.data(), rows);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(x, z);

    err = cudaFree(y);
    CHECK_EQ(err, cudaSuccess);
}

TEST_CASE("[cublas] life cycle") {
    cublasStatus_t stat;
    cublasHandle_t ctx{nullptr};

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}
