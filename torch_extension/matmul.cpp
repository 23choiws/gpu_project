// quantize_extension/matmul_int8_cuda.cpp

#include <torch/extension.h>
#include <vector>

// matmul_int8_cuda 함수 선언
void matmul_int8(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int M,
    int N,
    int K);

// C++ 인터페이스
torch::Tensor matmul_int8_cuda(torch::Tensor a, torch::Tensor b)
{
    // 입력 텐서가 CUDA에 있고 int8 타입인지 확인
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kInt8, "Input tensor a must be int8");
    TORCH_CHECK(b.dtype() == torch::kInt8, "Input tensor b must be int8");

    // 행렬 크기 가져오기
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    // 출력 텐서 생성
    auto options = torch::TensorOptions().device(a.device()).dtype(torch::kInt32);
    torch::Tensor c = torch::zeros({M, N}, options);
    // torch::Tensor c = torch::zeros({N, M}, options);


    //a = a.contiguous();
    //b = b.contiguous();
    // torch::Tensor b_t = b.t().contiguous();

    // TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    // TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    // TORCH_CHECK(b_t.is_contiguous(), "Input tensor b (transposed) must be contiguous");

    // CUDA 커널 실행
    matmul_int8(a, b, c, M, N, K);
  
    return c;
}


// #include <cuda_runtime.h>
// #include <cublasLt.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <iostream>

// #define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INT8(x) AT_ASSERTM(x.dtype() == at::kChar, #x " must be int8 (char) tensor")
// #define CHECK_DEVICE(x, d) AT_ASSERTM(x.device().type() == at::kCUDA, #x " must be on CUDA")

// #define CHECK_CUBLASLT(expr) {                                 \
//     cublasStatus_t status = (expr);                            \
//     if (status != CUBLAS_STATUS_SUCCESS) {                     \
//         std::cerr << "CUBLASLt Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
//         std::exit(EXIT_FAILURE);                               \
//     }                                                          \
// }

// at::Tensor matmul_int8_cuda(at::Tensor a, at::Tensor b) {
//     CHECK_CUDA(a);
//     CHECK_CUDA(b);
//     CHECK_CONTIGUOUS(a);
//     CHECK_CONTIGUOUS(b);
//     CHECK_INT8(a);
//     CHECK_INT8(b);

//     // shape check
//     TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimension mismatch");
//     int64_t M = a.size(0);
//     int64_t K = a.size(1);
//     int64_t N = b.size(1);

//     // 결과 텐서 생성: int32, same device
//     auto c = at::empty({M, N}, at::device(a.device()).dtype(at::kInt));

//     // get raw pointers
//     int8_t* dA = (int8_t*)a.data_ptr<int8_t>();
//     int8_t* dB = (int8_t*)b.data_ptr<int8_t>();
//     int32_t* dC = (int32_t*)c.data_ptr<int32_t>();

//     // cublasLt handle
//     cublasLtHandle_t ltHandle;
//     CHECK_CUBLASLT(cublasLtCreate(&ltHandle));

//     // Operation descriptor
//     cublasLtMatmulDesc_t operationDesc;
//     CHECK_CUBLASLT(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

//     // set transpose attributes (no transpose)
//     cublasOperation_t opA = CUBLAS_OP_N;
//     cublasOperation_t opB = CUBLAS_OP_N;
//     CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operationDesc,
//         CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
//     CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operationDesc,
//         CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

//     // Matrix layouts
//     cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
//     CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, M, K, K));
//     CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, N, N));
//     CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, M, N, N));

//     // set row-major order
//     cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
//     CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER,
//         &order, sizeof(order)));
//     CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER,
//         &order, sizeof(order)));
//     CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER,
//         &order, sizeof(order)));

//     // alpha, beta
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Preference
//     cublasLtMatmulPreference_t preference;
//     CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));

//     // 워크스페이스 할당 (간단히)
//     size_t workspaceSize = 4 * 1024 * 1024; // 4MB
//     void* dWorkspace = nullptr;
//     cudaMalloc(&dWorkspace, workspaceSize);
//     CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
//         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

//     // 히유리스틱 조회
//     cublasLtMatmulHeuristicResult_t heuristicResult;
//     int returnedResults = 0;
//     CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
//         ltHandle,
//         operationDesc,
//         layoutA,
//         layoutB,
//         layoutC,
//         layoutC,
//         preference,
//         1,
//         &heuristicResult,
//         &returnedResults
//     ));

//     if (returnedResults == 0) {
//         std::cerr << "No suitable cublasLt algorithm found for int8 matmul!" << std::endl;
//         std::exit(EXIT_FAILURE);
//     }

//     // CUDA 스트림 가져오기 (PyTorch 현재 스트림)
//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//     // Matmul 실행
//     CHECK_CUBLASLT(cublasLtMatmul(
//         ltHandle,
//         operationDesc,
//         &alpha,
//         dA, layoutA,
//         dB, layoutB,
//         &beta,
//         dC, layoutC,
//         dC, layoutC,
//         &heuristicResult.algo,
//         dWorkspace,
//         workspaceSize,
//         stream
//     ));

//     // 동기화 (필요하다면)
//     // cudaStreamSynchronize(stream);

//     // 리소스 정리
//     cublasLtMatmulPreferenceDestroy(preference);
//     cublasLtMatrixLayoutDestroy(layoutA);
//     cublasLtMatrixLayoutDestroy(layoutB);
//     cublasLtMatrixLayoutDestroy(layoutC);
//     cublasLtMatmulDescDestroy(operationDesc);
//     cublasLtDestroy(ltHandle);

//     cudaFree(dWorkspace);

//     return c;
// }

