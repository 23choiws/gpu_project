// quantize_extension/bindings.cpp

#include <torch/extension.h>
#include <vector>
#include "matmul.h"
#include "quantize.h"

// matmul_int8 함수 선언
//torch::Tensor matmul_int8_cuda(torch::Tensor a, torch::Tensor b);

// quantize_symmetric_cuda_interface 함수 선언
//std::pair<torch::Tensor, float> quantize_symmetric_cuda(const torch::Tensor& weight, int num_bits = 8);

// PYBIND11 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_int8_cuda", &matmul_int8_cuda, "Int8 Matrix Multiplication (CUDA)");
    m.def("quantize_symmetric_cuda", &quantize_symmetric_cuda, "Quantize symmetric (CUDA)", 
          py::arg("weight"), py::arg("num_bits")=8);
}
