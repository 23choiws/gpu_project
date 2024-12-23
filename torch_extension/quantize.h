// quantize_extension/quantize_symmetric.h

#ifndef QUANTIZE_SYMMETRIC_H
#define QUANTIZE_SYMMETRIC_H

#include <torch/extension.h>

// quantize_symmetric_cuda_interface 함수 선언
std::pair<torch::Tensor, torch::Tensor> quantize_symmetric_cuda(const torch::Tensor& weight, int num_bits = 8);

#endif // QUANTIZE_SYMMETRIC_H
