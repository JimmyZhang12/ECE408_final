//#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define TILE_WIDTH 32

__global__ void x_unroll(float *x, __half *x_unroll, const int B, const int M, const int C, const int H, const int W, const int K){
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //B X (C*K*K) X (H_out*W_out)
    #define x_unroll_4d(i2, i1, i0) x_unroll[(i2) * (C*K*K*H_out*W_out) + (i1) * (H_out*W_out) + i0]

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int h_x = bx / W_out + tx / K;  
    int w_x = bx % W_out + tx % K;

    for(int b=0; b<B; b++){
        int row = tx + by*K*K;
        x_unroll_4d(b, row, bx) = __float2half(x4d(b, by, h_x, w_x));
    }

}
__global__ void k_unroll(float *k, __half *k_unroll, const int B, const int M, const int C, const int H, const int W, const int K){
    //const int H_out = H - K + 1;
    //const int W_out = W - K + 1;
    int tx = threadIdx.x;
    int m = blockIdx.y;
    int c = blockIdx.x;

    //M X C X K X K
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    //K^2 * C X M
    #define k_unroll_4d(i1, i0) k_unroll[(i1) * (C*K*K) + i0]


    int h = tx / K; 
    int w = tx % K; 
    
    k_unroll_4d(m, c*K*K + tx) = __float2half(k4d(m, c, h, w));
}

//C = A*B
__global__ void matrixMultiplyShared(__half *x_unroll, __half *k_unroll, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int x_rows = K*K*C; 
    const int x_cols = H_out*W_out;
    const int k_rows = M; 
    const int k_cols = K*K*C;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define k_unroll_4d(i1, i0) k_unroll[(i1) * (C*K*K) + i0]
    #define x_unroll_4d(i2, i1, i0) x_unroll[(i2) * (C*K*K*H_out*W_out) + (i1) * (H_out*W_out) + i0]

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; 
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ __half rowTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half colTile[TILE_WIDTH][TILE_WIDTH];
    

    for(int b=0; b<B; b++){
        __half accum = 0;

        for(int p=0; p<ceil(k_cols / (float)TILE_WIDTH); ++p){
            if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols) 
                rowTile[threadIdx.y][threadIdx.x] = k_unroll_4d(row, threadIdx.x + (p*TILE_WIDTH) );
            else
                rowTile[threadIdx.y][threadIdx.x] = 0.0;
            
            if (col < x_cols && (threadIdx.y+TILE_WIDTH*p) < x_rows)
                colTile[threadIdx.y][threadIdx.x] = x_unroll_4d(b, (threadIdx.y + p*TILE_WIDTH), col);
            else
                colTile[threadIdx.y][threadIdx.x] = 0.0;

            __syncthreads();

            for(int i=0; i<TILE_WIDTH; ++i){
                accum += rowTile[threadIdx.y][i] * colTile[i][threadIdx.x];
            }
            __syncthreads();

        }
        if (row < M && col < H_out*W_out) 
            y4d(b, row, col/W_out, col%W_out) = __half2float(accum);
    }
  
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    float *device_k;
    __half *device_x_unroll;
    __half *device_k_unroll;

    // std::cout << "K: " << K << " M: " << M << " C: " << C << std::endl;
    // std::cout << "B: " << B << " H: " << H << " W: " << W << std::endl;
    // std::cout << "H_out: " << H_out << " W_out: " << W_out << std::endl;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (B*C*H*W) * sizeof(float));
    cudaMalloc((void **) &device_y, (B*M*H_out*W_out) * sizeof(float));
    cudaMalloc((void **) &device_k, (C*M*K*K) * sizeof(float));
    //B X M X C
    cudaMalloc((void **) &device_x_unroll, B * (C*K*K) * (H_out*W_out) * sizeof(__half));
    cudaMalloc((void **) &device_k_unroll, K*K * C * M * sizeof(__half));

    bool val;
    val = cudaMemcpy(device_x, host_x, (B*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    val = cudaMemcpy(device_y, host_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    val = cudaMemcpy(device_k, host_k, (C*M*K*K) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    dim3 gridDim_x(H_out*W_out, C, 1);
    dim3 blockDim_x(K*K, 1, 1);
    x_unroll<<< gridDim_x, blockDim_x>>>(device_x, device_x_unroll, 
                B, M, C, H, W, K);

    dim3 gridDim_k(C, M, 1);
    dim3 blockDim_k(K*K, 1, 1);
    k_unroll<<< gridDim_k, blockDim_k>>>(device_k, device_k_unroll, 
                B, M, C, H, W, K);

    dim3 gridDim(ceil(H_out*W_out/(1.0 * TILE_WIDTH)), ceil(M/(1.0 * TILE_WIDTH)), 1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyShared<<< gridDim, blockDim>>>(device_x_unroll, device_k_unroll, device_y,
                B, M, C, H, W, K);

    val = cudaMemcpy(host_y, device_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost);

    // if (W == 86){
    //     #define y4d(i3, i2, i1, i0) host_y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    //     for(int i=0; i<H_out; i++){
    //         for(int j=0; j<W_out; j++){
    //             float temp = y4d(0,0,i,j);
    //             std::cout << temp << " ";
    //         }
    //         std::cout << std::endl << std::endl;
    //     }

    // }

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);
    cudaFree(device_x_unroll);
    cudaFree(device_k_unroll);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
