// quantize_extension/matmul_int8_cuda_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <vector>
#include <cublasLt.h>

using namespace nvcuda;

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// 오류 체크 매크로 정의 (cuBLAS)
#define CHECK_CUBLAS(call)                                                \
  do {                                                                    \
    cublasStatus_t status_ = call;                                       \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                              \
      fprintf(stderr, "cuBLAS error (%s:%d): %s\n", __FILE__, __LINE__, \
              cublasGetErrorString(status_));                            \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

  // cuBLAS 상태를 문자열로 변환하는 함수
const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        // 필요에 따라 추가적인 상태를 처리할 수 있습니다.
        default:
            return "Unknown cuBLAS status";
    }
}

// Tensor Cores를 사용한 int8 행렬 곱셈 커널
__global__ void matmul_int8_cuda_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int32_t* __restrict__ c,
    int M,
    int N,
    int K)
{
    // WMMA는 M과 N이 16의 배수, K도 16의 배수여야 함

    __shared__ int8_t as[256];
    __shared__ int8_t bs[8][256];

    if (blockDim.x != 256) return;  // force 256 threads per block

    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize; // warp rank in grid

    int cx = warp % (N / 16);  // (x,y) location of active tile
    int cy = warp / (N / 16);  // for current warp in C matrix

    int Atile_pos = cy * 16 * K; // start x (row) for first A tile
    int Btile_pos = cx * 16;    // start y (col) for first B tile

    int wb = threadIdx.x / 32;  // warp rank in block  in [0,255]
    int trw = threadIdx.x % 32;  // thread rank in warp 
    int txw = trw % 16;          // thread x in warp    in [0,15]
    int tyw = trw / 16;          // thread y in warp    in [0, 1]

    int idx = threadIdx.x % 16;  // assign 256 threads to cover
    int idy = threadIdx.x / 16;  // 16 x 16 x-y values in tile

    // WMMA 프래그먼트 정의
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, int8_t, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, int32_t> c_frag;

    // 결과 초기화
    wmma::fill_fragment(c_frag, 0);

    // 입력 로드
    // wmma::load_matrix_sync(a_frag, a, K);
    // wmma::load_matrix_sync(b_frag, b, N);
    // //wmma::load_matrix_sync(b_frag, b, K);

    // // 행렬 곱 수행
    // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // for (int i = 0; i < K / 16; i++) { // accumulate su, of row*column for C tile
    //     wmma::load_matrix_sync(a_frag, &a[Atile_pos], K);  // load A as 16x16 tile
    //     wmma::load_matrix_sync(b_frag, &b[Btile_pos], N);  // load B as 16x16 tile		
    //     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);      // C = A*B + C
    //     Atile_pos += 16;   // step along row of A
    //     Btile_pos += 16 * N;  // step down column of B
    // }
    
    for (int i = 0; i < K / 16; i++) {
        as[idy * 16 + idx] = a[Atile_pos + idy * K + idx];  // 256 threads used here
        __syncthreads();   // 32 threads fill tile in 8 passes
        for (int p = 0; p < 8; p++)  
            bs[wb][p * 32 + tyw * 16 + txw] = b[p * 2 * N + Btile_pos + tyw * N + txw];
        __syncwarp();
        wmma::load_matrix_sync(a_frag, &as[0], 16);      // load A as 16x16 tile
        wmma::load_matrix_sync(b_frag, &bs[wb][0], 16);  // load B as 16x16 tile	
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);   // C = A*B + C
        Atile_pos += 16;     // move along A row
        Btile_pos += 16 * N;  // move down B cols
    }

    wmma::store_matrix_sync(&c[(cy * N + cx) * 16], c_frag, N, wmma::mem_row_major);
   
    // 출력 저장
    // wmma::store_matrix_sync(c, c_frag, N, wmma::mem_row_major);
}

// // 래퍼 함수
// void matmul_int8(
//     at::Tensor a,
//     at::Tensor b,
//     at::Tensor c,
//     int M,
//     int N,
//     int K)
// {
//     // 적절한 그리드 및 블록 차원으로 커널 실행
//     // 단순화를 위해 하나의 블록을 사용. 성능 향상을 위해 여러 블록으로 작업 분할 필요
//     //dim3 grid(1, 1);
//     //dim3 block(1, 1);

//     // dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
//     // dim3 block(1, 1);
//     int threadsT, blocksT;
//     threadsT = 256; // fixed
//     blocksT = M * N / (8 * threadsT);
//     if(blocksT == 0) blocksT++;

//     matmul_int8_cuda_kernel<<<blocksT, threadsT>>>(a.data_ptr<int8_t>(),
//                                       b.data_ptr<int8_t>(),
//                                       c.data_ptr<int32_t>(),
//                                       M, N, K);
//     CHECK_CUDA(cudaDeviceSynchronize());

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA Error in matmul_kernel: %s\n", cudaGetErrorString(err));
//     }
// }

// #define CHECK_CUBLAS(stat) \
//     if (stat != CUBLAS_STATUS_SUCCESS) { \
//         fprintf(stderr, "CUBLAS error at %s:%d\n", __FILE__, __LINE__); \
//         exit(EXIT_FAILURE); \
//     }


// 래퍼 함수
void matmul_int8(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int M,
    int N,
    int K)
{
    // 입력 텐서가 모두 contiguous()이고 row-major 형식임을 가정
    // cuBLAS는 column-major을 사용하므로, 전치를 활용하여 연산 수행

    // cuBLAS 핸들 초기화 (한 번만 수행하는 것이 좋음)
    // cublasHandle_t handle = [](){
    //     cublasHandle_t h;
    //     cublasStatus_t status = cublasCreate(&h);
    //     if (status != CUBLAS_STATUS_SUCCESS) {
    //         fprintf(stderr, "cuBLAS initialization failed: %s\n", cublasGetErrorString(status));
    //         exit(EXIT_FAILURE);
    //     }
    //     // 필요에 따라 스트림 설정 가능
    //     // cublasSetStream(h, stream);
    //     return h;
    // }();
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    // 스케일 팩터 설정 (필요에 따라 조정)
    // float alpha = 1.0f; // 스케일 팩터 float?
    // float beta = 0.0f;
    const int alpha = 1;
    const int beta = 0;

    // // 행렬 곱 수행
    // CHECK_CUBLAS(
    //     cublasGemmEx(
    //     handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_N,
    //     N, M, K,
    //     &alpha,
    //     b.data_ptr<int8_t>(), CUDA_R_8I, N,     
    //     a.data_ptr<int8_t>(), CUDA_R_8I, K,     
    //     &beta,
    //     c.data_ptr<int32_t>(), CUDA_R_32I, N,     
    //     CUDA_R_32I,
    //     CUBLAS_GEMM_DEFAULT_TENSOR_OP
    //     )
    // );

    // 행렬 곱 수행
    CHECK_CUBLAS(
        cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        b.data_ptr<int8_t>(), CUDA_R_8I, K,     
        a.data_ptr<int8_t>(), CUDA_R_8I, K,     
        &beta,
        c.data_ptr<int32_t>(), CUDA_R_32I, N,     
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
        )
    );

    // 행렬 곱 수행
    // CHECK_CUBLAS(
    //     cublasGemmEx(
    //     handle,
    //     CUBLAS_OP_N, // Note: B와 A를 뒤바꾼 상태에서 transa, transb를 설정
    //     CUBLAS_OP_N,
    //     N,     // C^T 행 수 (원래 C의 열 수 N)
    //     M,     // C^T 열 수 (원래 C의 행 수 M)
    //     K,     // 공통 차원
    //     &alpha,
    //     b.data_ptr<int8_t>(),  // B^T 행렬
    //     CUDA_R_8I,
    //     K,     // lda = B^T의 leading dim = N
    //     a.data_ptr<int8_t>(),  // A^T 행렬
    //     CUDA_R_8I,
    //     M,     // ldb = A^T의 leading dim = K
    //     &beta,
    //     c.data_ptr<int32_t>(),  // C^T 행렬
    //     CUDA_R_32I,
    //     N,     // ldc = C^T의 leading dim = N
    //     CUBLAS_COMPUTE_32I,
    //     CUBLAS_GEMM_DEFAULT_TENSOR_OP
    //     )
    // );
    
    // 동기화 및 오류 검사
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasDestroy(handle);
}


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