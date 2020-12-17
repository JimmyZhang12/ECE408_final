
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#include <mma.h>
#define TILE_WIDTH 16 //fixed for tensors
#define KERNEL_SIZE 7

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__constant__ float K_ONE[4*KERNEL_SIZE*KERNEL_SIZE];
__constant__ float K_TWO[16*4*KERNEL_SIZE*KERNEL_SIZE];

//C = A*B
using namespace nvcuda;
__global__ void matrixMultiplyShared(float *x, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int x_rows = K*K*C; 
    const int x_cols = H_out*W_out;
    const int k_rows = M; 
    const int k_cols = K*K*C;
    const int b = blockIdx.z;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define k4d_2(i3, i2, i1, i0) K_TWO[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; 
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ __half rowTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half colTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float outTile[TILE_WIDTH][TILE_WIDTH];

    const half *row_ptr = &rowTile[0][0];
    const half *col_ptr = &colTile[0][0];
    float *out_ptr = &outTile[0][0];

#pragma unroll
    for(int p=0; p<ceil(k_cols / (float)TILE_WIDTH); ++p){

        int c = (threadIdx.x + p*TILE_WIDTH)/(K*K);
        int k_remain =  (threadIdx.x + p*TILE_WIDTH) % (K*K);

        if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols) 
            rowTile[threadIdx.y][threadIdx.x] = __float2half(k4d_2(row, c, k_remain/K, k_remain%K));
        else
            rowTile[threadIdx.y][threadIdx.x] = 0.0; 
        
        c = (threadIdx.y + p*TILE_WIDTH)/(K*K);
        int x_remain =  (threadIdx.y + p*TILE_WIDTH) % (K*K);
        int y_in = col/W_out;
        int x_in = col%W_out;

        if (col < x_cols && (threadIdx.y+TILE_WIDTH*p) < x_rows)
            colTile[threadIdx.y][threadIdx.x] = __float2half(x4d(b,c, y_in + x_remain/K, x_in + x_remain%K));
        else
            colTile[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        wmma::load_matrix_sync(a_frag, row_ptr, 16);
        wmma::load_matrix_sync(b_frag, col_ptr, 16);  
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }

    wmma::store_matrix_sync(out_ptr, c_frag, 16, wmma::mem_row_major);
    if (row < M && col < H_out*W_out)
        y4d(b, row, col/W_out, col%W_out) = outTile[threadIdx.y][threadIdx.x];
    
}

__global__ void matrixMultiplyShared_l1(float *x, float *y,
                                     const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int x_rows = K*K*C; 
    const int x_cols = H_out*W_out;
    const int k_rows = M; 
    const int k_cols = K*K*C;
    const int b = blockIdx.z;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define k4d_1(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int row = threadIdx.y; 
    int col = blockIdx.x * 32 + threadIdx.x;

    __shared__ __half rowTile[8][16];
    __shared__ __half colTile[16][32];
    __shared__ float outTile[8][32];

    const half *row_ptr = &rowTile[0][0];
    const half *col_ptr = &colTile[0][0];
    float *out_ptr = &outTile[0][0];

    //__half accum = 0;

#pragma unroll
    for(int p=0; p<ceil(k_cols / (float)TILE_WIDTH); ++p){
        if(threadIdx.x < 16 && threadIdx.y < 8){

            int c = (threadIdx.x + p*TILE_WIDTH)/(K*K);
            int k_remain =  (threadIdx.x + p*TILE_WIDTH) % (K*K);

            if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols) 
                rowTile[threadIdx.y][threadIdx.x] = __float2half(k4d_1(row, c, k_remain/K, k_remain%K));
            else
                rowTile[threadIdx.y][threadIdx.x] = 0.0;

            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && p == 1){
            //     //if(threadIdx.x == 0 && threadIdx.y == 0){
            //         printf("rowtile[%d][%d] = k[%d][%d][%d][%d]\n", threadIdx.y, threadIdx.x, row, c, k_remain/K, k_remain%K);
            //     //}
            // }

        }
        
        int c = (threadIdx.y + p*TILE_WIDTH)/(K*K);
        int x_remain =  (threadIdx.y + p*TILE_WIDTH) % (K*K);
        int y_in = col/W_out;
        int x_in = col%W_out;

        if (col < x_cols && (threadIdx.y+TILE_WIDTH*p) < x_rows)
            colTile[threadIdx.y][threadIdx.x] = __float2half(x4d(b,c, y_in + x_remain/K, x_in + x_remain%K));
        else
            colTile[threadIdx.y][threadIdx.x] = 0.0;

        // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && p==0 && col == 0){
        //     //if(threadIdx.x == 0 && threadIdx.y == 0){
        //         printf("coltile[%d][%d] = x[%d][%d][%d][%d], col = %d\n", threadIdx.x, threadIdx.y, b,c, y_in + x_remain/K, x_in + x_remain%K,col);
        //     //}
        // }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, row_ptr, 16);
        wmma::load_matrix_sync(b_frag, col_ptr, 32);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        //  for(int i=0; i<16; ++i){
        //         accum += rowTile[threadIdx.y][i] * colTile[i][threadIdx.x];
        // }
        __syncthreads();
    }

    wmma::store_matrix_sync(out_ptr, c_frag, 32, wmma::mem_row_major);

        if (row < M && col < H_out*W_out)
            //y4d(b, row, col/W_out, col%W_out) = outTile[threadIdx.y][threadIdx.x];
            y4d(b, row, col/W_out, col%W_out) = outTile[threadIdx.y][threadIdx.x];
            
            // if (blockIdx.x == 100 && blockIdx.y == 0 && blockIdx.z == 4){
            //     printf("tid[%d][%d] = outtile: %f, accum: %f\n", threadIdx.y, threadIdx.x, outTile[threadIdx.y][threadIdx.x], __half2float(accum));
            // }
    
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    //float *device_k;

    //  std::cout << "K: " << K << " M: " << M << " C: " << C << std::endl;
    //  std::cout << "B: " << B << " H: " << H << " W: " << W << std::endl;
    //  std::cout << "H_out: " << H_out << " W_out: " << W_out << std::endl;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (B*C*H*W) * sizeof(float));
    //cudaMalloc((void **) &device_k, (C*M*K*K) * sizeof(float));
    cudaMalloc((void **) &device_y, (B*M*H_out*W_out) * sizeof(float));

    bool val;

    //val = cudaMemcpy(device_k, host_k, (C*M*K*K) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';
    val = cudaMemcpy(device_y, host_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';
    val = cudaMemcpy(device_x, host_x, (B*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);

    if (M == 4){
        cudaMemcpyToSymbol(K_ONE, host_k, M*C*K*K*sizeof(float));
        dim3 gridDim(ceil(H_out*W_out/(32.0)), 1, B);
        dim3 blockDim(32, 16, 1);
        matrixMultiplyShared_l1<<< gridDim, blockDim>>>(device_x, device_y,
                    B, M, C, H, W, K);
    }

    if (M == 16) {
        cudaMemcpyToSymbol(K_TWO, host_k, M*C*K*K*sizeof(float));
        dim3 gridDim(ceil(H_out*W_out/(1.0 * TILE_WIDTH)), ceil(M/(1.0 * TILE_WIDTH)), B);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<< gridDim, blockDim>>>(device_x, device_y,
                    B, M, C, H, W, K);
    }
    val = cudaMemcpy(host_y, device_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost);
        

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    //cudaFree(device_k);

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
