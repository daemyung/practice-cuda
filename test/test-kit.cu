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

#include "test-kit.cuh"

#include <stdexcept>

vector_ptr make_vector(uint32_t n) {
    float* x;

    if (auto err = cudaMalloc(&x, n * sizeof(float));
        err != cudaSuccess || !x) {
        throw std::runtime_error("error|cuda: fail to malloc.");
    }

    return {x, [](float* x) { cudaFree(x); }};
}

vector_ptr make_vector(std::vector<float> const& x) {
    auto y = make_vector(x.size());

    if (auto stat = cublasSetVector(x.size(), sizeof(float), x.data(), 1, y.get(), 1);
        stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("error|cublas: fail to set vector.");
    }

    return y;
}

matrix_ptr make_matrix(int32_t r, int32_t c) {
    float* x;

    if (auto err = cudaMalloc(&x, r * c * sizeof(float));
        err != cudaSuccess || !x) {
        throw std::runtime_error("error|cuda: fail to malloc.");
    }

    return {x, [](float* x) { cudaFree(x); }};
}

matrix_ptr make_matrix(std::vector<std::vector<float>> const& x) {
    std::vector<float> z;

    for (auto& v : x) {
        z.insert(std::end(z), std::begin(v), std::end(v));
    }

    const auto r = static_cast<int32_t>(x[0].size());
    const auto c = static_cast<int32_t>(x.size());
    auto y = make_matrix(r, c);

    if (auto stat = cublasSetMatrix(r, c, sizeof(float), z.data(), r, y.get(), r);
        stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("error|cublas: fail to set matrix.");
    }

    return y;
}
