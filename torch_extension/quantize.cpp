#include <torch/extension.h>
#include <vector>
#include <cmath>

// 선언된 CUDA 함수
float find_max_abs(const torch::Tensor& input);
torch::Tensor quantize(const torch::Tensor& input, float scale);

// 파이썬에서 호출 가능한 함수
std::pair<torch::Tensor, torch::Tensor> quantize_symmetric_cuda(const torch::Tensor& weight, int num_bits=8) {
    // 최대 절대값 찾기
    float max_val = find_max_abs(weight);

    // 스케일 계산
    float qmax = (1 << (num_bits - 1)) - 1;
    float scale = max_val / qmax;
    // if (max_val == 0) scale = 1.0;
    if (max_val == 0.0f) scale = 1.0f;

    
    // 양자화 수행
    torch::Tensor weight_int8 = quantize(weight, scale);
    //torch::Tensor scale_tensor = torch::tensor(scale, weight.options()).view({1, 1});
    torch::Tensor scale_tensor = torch::tensor({{scale}});

    // return std::make_pair(weight_int8, torch::tensor(scale, weight.options()));
    return std::make_pair(weight_int8, scale_tensor);

}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("quantize_symmetric_cuda", &quantize_symmetric_cuda, "Quantize symmetric (CUDA)");
// }

// PyBind11 모듈 정의
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("quantize_symmetric_cuda", &quantize_symmetric_cuda, "Quantize symmetric (CUDA)", 
//           py::arg("weight"), py::arg("num_bits")=8);
// }