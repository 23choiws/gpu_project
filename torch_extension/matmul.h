// quantize_extension/matmul_int8.h

#ifndef MATMUL_INT8_H
#define MATMUL_INT8_H

#include <torch/extension.h>

// matmul_int8 함수 선언
torch::Tensor matmul_int8_cuda(torch::Tensor a, torch::Tensor b);

#endif // MATMUL_INT8_H
