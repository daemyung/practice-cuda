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
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "test-kit.cuh"

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
    constexpr int32_t r{4};
    constexpr int32_t c{3};
    std::vector x{1.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 1.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 1.0f, 0.0f};
    float* y{nullptr};

    cudaError err;
    cublasStatus_t stat;

    err = cudaMalloc((void**)&y, x.size() * sizeof(decltype(x)::value_type));
    CHECK_EQ(err, cudaSuccess);

    stat = cublasSetMatrix(r, c, sizeof(decltype(x)::value_type), x.data(), r, y, r);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector z{0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f,
                  0.0f, 0.0f, 0.0f, 0.0f};
    stat = cublasGetMatrix(r, c, sizeof(decltype(x)::value_type), y, r, z.data(), r);
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

TEST_CASE("[cublas] life cycle") {
    cublasStatus_t stat;
    cublasHandle_t ctx{nullptr};

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] amax") {
    const std::vector a{3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const auto n = static_cast<int32_t>(a.size());
    auto x = make_vector(a);

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    int32_t res{0};
    stat = cublasIsamax(ctx, n, x.get(), 1, &res);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(res, std::max_element(std::begin(a), std::end(a)) - std::begin(a) + 1);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] amin") {
    const std::vector a{3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const auto n = static_cast<int32_t>(a.size());
    auto x = make_vector(a);

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    int32_t res{0};
    stat = cublasIsamin(ctx, n, x.get(), 1, &res);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(res, std::min_element(std::begin(a), std::end(a)) - std::begin(a) + 1);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] asum") {
    std::vector a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x = make_vector(a);

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    float sum;
    stat = cublasSasum(ctx, a.size(), x.get(), 1, &sum);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(sum, std::accumulate(std::begin(a), std::end(a), 0.0f));

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] axpy") {
    const float a{2.0f};
    const float x{3.0f};
    const float y{1.0f};

    cublasStatus_t stat;
    cublasHandle_t ctx;

    auto i = make_vector(std::vector{x});
    auto j = make_vector(std::vector{y});

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    stat = cublasSaxpy(ctx, 1, &a, i.get(), 1, j.get(), 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    float k;
    cudaMemcpy(&k, j.get(), sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(a * x + y, k);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] copy") {
    const std::vector a{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f};
    const auto n = static_cast<int32_t>(a.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    auto x = make_vector(a);
    auto y = make_vector(n);

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    stat = cublasScopy(ctx, n, x.get(), 1, y.get(), 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector b(n, 0.0f);
    cudaMemcpy(b.data(), y.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(a, b);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] dot") {
    const std::vector a{1.0f, 2.0f, 3.0f};
    const std::vector b{4.0f, 5.0f, 6.0f};
    const auto n = static_cast<int32_t>(a.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    auto x = make_vector(a);
    auto y = make_vector(b);

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    float z;
    stat = cublasSdot(ctx, n, x.get(), 1, y.get(), 1, &z);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(z, std::inner_product(std::begin(a), std::end(a), std::begin(b), 0.0f));

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] nrm2") {
    const std::vector a{1.0f, 2.0f, 3.0f, 4.0f};
    const auto n = static_cast<int32_t>(a.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    auto x = make_vector(a);

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    float y;
    stat = cublasSnrm2(ctx, n, x.get(), 1, &y);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(y, std::sqrt(std::inner_product(std::begin(a), std::end(a), std::begin(a), 0.0f)));

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] rot") {
    const std::vector x{1.0f, 0.0f};
    const std::vector y{0.0f, 1.0f};
    const auto n = static_cast<int32_t>(x.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    auto a = make_vector(x);
    auto b = make_vector(y);
    const float c{cosf(M_PI_2)};
    const float s{sinf(M_PI_2)};

    stat = cublasSrot(ctx, n, a.get(), 1, b.get(), 1, &c, &s);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector i{0.0f, 1.0f};
    cudaMemcpy(i.data(), a.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    std::vector j{0.0f, 0.0f};
    cudaMemcpy(j.data(), b.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] rotg") {
    float x{4.0f};
    float y{3.0f};
    float c{cosf(M_PI_2)};
    float s{sinf(M_PI_2)};

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    float r{x};
    float z{y};
    stat = cublasSrotg(ctx, &r, &z, &c, &s);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(r, sqrtf(x * x + y * y));

    if (abs(z) < 1) {
        CHECK_EQ(c, sqrtf(1 - z * z));
        CHECK_EQ(s, z);
    } else if (abs(z) == 1) {
        CHECK_EQ(c, 0.0f);
        CHECK_EQ(s, 1.0f);
    } else {
        CHECK_EQ(c, 1.0f / z);
        CHECK_EQ(s, sqrtf(1 - z * z));
    }

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] rotm") {
    const std::vector x{2.0f, 8.0f};
    const std::vector y{5.0f, 3.0f};
    const auto n = static_cast<int32_t>(x.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    auto a = make_vector(x);
    auto b = make_vector(y);
    std::vector i(n, 0.0f);
    std::vector j(n, 0.0f);

    // This params make a below matrix.
    // [[1.0, 2.0]
    //  [3.0, 1.0]]
    std::array params{-3.0f, 1.0f, 3.0f, 2.0f, 1.0f};

    params[0] = -1.0f;
    stat = cublasSrotm(ctx, n, a.get(), 1, b.get(), 1, params.data());
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    cudaMemcpy(i.data(), a.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(i[0], params[1] * x[0] + params[3] * y[0]);
    CHECK_EQ(i[1], params[1] * x[1] + params[3] * y[1]);

    cudaMemcpy(j.data(), b.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(j[0], params[2] * x[0] + params[4] * y[0]);
    CHECK_EQ(j[1], params[2] * x[1] + params[4] * y[1]);

    a = make_vector(x);
    b = make_vector(y);

    params[0] = 0.0f;
    stat = cublasSrotm(ctx, n, a.get(), 1, b.get(), 1, params.data());
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    cudaMemcpy(i.data(), a.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(i[0], 1.0 * x[0] + params[3] * y[0]);
    CHECK_EQ(i[1], 1.0 * x[1] + params[3] * y[1]);

    cudaMemcpy(j.data(), b.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(j[0], params[2] * x[0] + 1.0 * y[0]);
    CHECK_EQ(j[1], params[2] * x[1] + 1.0 * y[1]);

    a = make_vector(x);
    b = make_vector(y);

    params[0] = 1.0f;
    stat = cublasSrotm(ctx, n, a.get(), 1, b.get(), 1, params.data());
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    cudaMemcpy(i.data(), a.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(i[0], params[1] * x[0] + 1.0f * y[0]);
    CHECK_EQ(i[1], params[1] * x[1] + 1.0f * y[1]);

    cudaMemcpy(j.data(), b.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(j[0], -1.0f * x[0] + params[4] * y[0]);
    CHECK_EQ(j[1], -1.0f * x[1] + params[4] * y[1]);

    a = make_vector(x);
    b = make_vector(y);

    params[0] = -2.0f;
    stat = cublasSrotm(ctx, n, a.get(), 1, b.get(), 1, params.data());
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    cudaMemcpy(i.data(), a.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(i[0], 1.0 * x[0] + 0.0f * y[0]);
    CHECK_EQ(i[1], 1.0 * x[1] + 0.0f * y[1]);

    cudaMemcpy(j.data(), b.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(j[0], 0.0f * x[0] + 1.0 * y[0]);
    CHECK_EQ(j[1], 0.0f * x[1] + 1.0 * y[1]);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] scal") {
    const float a = 2.0f;
    std::vector i{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    const auto n = static_cast<int32_t>(i.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    auto x = make_vector(i);

    stat = cublasSscal(ctx, n, &a, x.get(), 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector y(n, 0.0f);
    cudaMemcpy(y.data(), x.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    std::transform(std::begin(i), std::end(i), std::begin(i), [a](auto e) { return a * e; });
    CHECK_EQ(y, i);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("[cublas] swap") {
    std::vector a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector b{5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    const auto n = static_cast<int32_t>(a.size());

    cublasStatus_t stat;
    cublasHandle_t ctx;

    stat = cublasCreate(&ctx);
    CHECK_NE(ctx, nullptr);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    auto x = make_vector(a);
    auto y = make_vector(b);

    stat = cublasSswap(ctx, n, x.get(), 1, y.get(), 1);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);

    std::vector i(n, 0.0f);
    cudaMemcpy(i.data(), x.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(i, b);

    std::vector j(n, 0.0f);
    cudaMemcpy(j.data(), y.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    CHECK_EQ(j, a);

    stat = cublasDestroy(ctx);
    CHECK_EQ(stat, CUBLAS_STATUS_SUCCESS);
}
