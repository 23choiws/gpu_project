#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// CUDA 커널: float에 대한 atomicMax 구현
// __device__ float atomicMaxFloat(float* address, float value) {
//     int old = __float_as_int(*address);
//     int assumed;
//     do {
//         assumed = old;
//         float old_f = __int_as_float(assumed);
//         if (old_f >= value)
//             break;
//         old = atomicCAS((int*)address, assumed, __float_as_int(value));
//     } while (assumed != old);
//     return __int_as_float(old);
// }

__device__ float atomicMaxFloat(float* address, float value) {
    int* int_addr = reinterpret_cast<int*>(address);
    int old = *int_addr, assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(value, old_val);
        int new_int = __float_as_int(new_val);
        old = atomicCAS(int_addr, assumed, new_int);
    } while (assumed != old);
    return __int_as_float(old); 
}
// atomicMax(reinterpret_cast<int*>(max_val), __float_as_int(sdata[0]));
//atomicMax(max_val, sdata[0]);


// CUDA 커널: 절대값의 최대값 찾기
__global__ void max_abs_kernel(const float* __restrict__ input, float* __restrict__ max_val, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (index < size) ? fabsf(input[index]) : -INFINITY;
    sdata[tid] = val;
    __syncthreads();

    // 블록 내에서 최대값 찾기
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 블록의 첫 번째 스레드가 블록의 최대값을 저장
    if (tid == 0) {
        atomicMaxFloat(max_val, sdata[0]);
    }
}

// CUDA 커널: 양자화 연산
__global__ void quantize_kernel(const float* __restrict__ input, int8_t* __restrict__ output, float scale, int size) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float scaled = roundf(input[index] / scale);
        scaled = fmaxf(-128.0f, fminf(127.0f, scaled));
        output[index] = static_cast<int8_t>(scaled);
    }
}

// C++ 함수: 최대 절대값 찾기
float find_max_abs(const torch::Tensor& input) {
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    //torch::Tensor max_tensor = torch::zeros(1, input.options().dtype(torch::kFloat32));
    // 명시적으로 디바이스 지정
    torch::Tensor max_tensor = torch::full({1}, -INFINITY, input.options().dtype(torch::kFloat32));

    max_abs_kernel<<<blocks, threads, threads * sizeof(float)>>>(input.data_ptr<float>(), max_tensor.data_ptr<float>(), input.numel());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 에러 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in max_abs_kernel: %s\n", cudaGetErrorString(err));
    }

    // 커널 동기화 및 오류 체크
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA Device Synchronize Error: %s\n", cudaGetErrorString(err));
    // }

    // 호스트로 최대값 복사
    float max_val = 0.0;
    //cudaMemcpy(&max_val, max_tensor.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA(cudaMemcpy(&max_val, max_tensor.data_ptr<float>(), sizeof(float), cudaMemcpyDeviceToHost));
    
    return max_val;
}

// C++ 함수: 양자화 수행
torch::Tensor quantize(const torch::Tensor& input, float scale) {
    auto output = torch::empty_like(input, torch::dtype(torch::kInt8));
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<int8_t>(), scale, input.numel());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 에러 체크
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in quantize_kernel: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("find_max_abs", &find_max_abs, "Find max absolute value (CUDA)");
//     m.def("quantize", &quantize, "Quantize tensor (CUDA)");
// }
