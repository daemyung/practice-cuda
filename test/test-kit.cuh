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

#ifndef PRACTICE_CUDA_TEST_KIT_CUH
#define PRACTICE_CUDA_TEST_KIT_CUH


#include <vector>
#include <memory>
#include <functional>
#include <cublas_v2.h>

using vector_ptr = std::unique_ptr<float, std::function<void(float*)>>;
using matrix_ptr = vector_ptr;

extern vector_ptr make_vector(uint32_t n);
extern vector_ptr make_vector(std::vector<float> const& x);
extern matrix_ptr make_matrix(int32_t r, int32_t c);
extern matrix_ptr make_matrix(std::vector<std::vector<float>> const& x);


#endif //PRACTICE_CUDA_TEST_KIT_CUH
