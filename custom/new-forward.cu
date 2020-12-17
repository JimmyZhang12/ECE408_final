
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#include <mma.h>
#define TILE_WIDTH 16 //fixed for tensors
#define KERNEL_SIZE 7
#define K2 49

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__constant__ float K_ONE[4*KERNEL_SIZE*KERNEL_SIZE];
__constant__ __half K_TWO[16*4*KERNEL_SIZE*KERNEL_SIZE];

//C = A*B
using namespace nvcuda;

// A little birdie on the Nvidia forums said that integer division is the devil's handiwork.
// So consider this float approximation an excorcism.

__device__ int int_mod(int a, int b) {
    return a - ((int) (__fdividef(a, b)))*b;// Float division via https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html
                                            // Can also use __fdiv instead, but the approximate solution is good enough for these indexes.
}


__global__ void matrixMultiplyShared(float * __restrict__ x, float * __restrict__ y,
                                     const int B, const int M, const int C, const int H, const int W, const int K) {

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int x_rows = K2*C; 
    const int x_cols = H_out*W_out;
    const int k_rows = M; 
    const int k_cols = K2*C;
    const int b = blockIdx.z;

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define k4d_2(i3, i2, i1, i0) K_TWO[(i3) * (C * K2) + (i2) * (K2) + (i1) * (K) + i0]
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
    for(int p=0; p<13; ++p){ // 13 = ceil(k_cols / (float)TILE_WIDTH)

        int c = ((int) (__fdividef((threadIdx.x + p*TILE_WIDTH),(K2))));
        int k_remain =  int_mod((threadIdx.x + p*TILE_WIDTH), (K2));
        __syncthreads();
        if (row < k_rows && (threadIdx.x + p*TILE_WIDTH) < k_cols) 
            rowTile[threadIdx.y][threadIdx.x] = k4d_2(row, c, ((int) (__fdividef(k_remain,K))), int_mod(k_remain,K));
        else
            rowTile[threadIdx.y][threadIdx.x] = 0.0; 
        
        c = (threadIdx.y + p*TILE_WIDTH)/(K2);
        int x_remain =  int_mod((threadIdx.y + p*TILE_WIDTH), (K2));
        int y_in = ((int) (__fdividef(col,W_out)));
        int x_in = int_mod(col,W_out);

        if (col < x_cols && (threadIdx.y+TILE_WIDTH*p) < x_rows)
            colTile[threadIdx.y][threadIdx.x] = __float2half(x4d(b,c, y_in + ((int) (__fdividef(x_remain,K))), x_in + int_mod(x_remain,K)));
        else
            colTile[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        wmma::load_matrix_sync(a_frag, row_ptr, 16);
        wmma::load_matrix_sync(b_frag, col_ptr, 16);  
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }

    wmma::store_matrix_sync(out_ptr, c_frag, 16, wmma::mem_row_major);
    if (row < M && col < H_out*W_out)
        y4d(b, row, ((int) (__fdividef(col,W_out))), int_mod(col,W_out)) = outTile[threadIdx.y][threadIdx.x];
    
}

__global__ void conv_forward_kernel(float * __restrict__ y, const float * __restrict__ x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k1_4d(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int x_idx = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y_idx = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int b = blockIdx.z;
    
    // compute for each kernel
    if (x_idx < W_out && y_idx < H_out) {
        #pragma unroll
        for (int m = 0; m < 4; m++) { // This kernel is only used on the first layer. So fix the inputs here to unroll.
            float val = 0;
            // iterate over y
            #pragma unroll
            for (int p = 0; p < 7; p++) {
                // iterate over x
                #pragma unroll
                for (int q = 0; q < 7; q++) {
                    val += x4d(b,0,y_idx+p,x_idx+q) * k1_4d(m,0,p,q);
                }
            }
            y4d(b, m, y_idx, x_idx) = val;
        }
    }

#undef y4d
#undef x4d
#undef k1_4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    __half *host_k_half;
    //__half *device_k_half;

    //  std::cout << "K: " << K << " M: " << M << " C: " << C << std::endl;
    //  std::cout << "B: " << B << " H: " << H << " W: " << W << std::endl;
    //  std::cout << "H_out: " << H_out << " W_out: " << W_out << std::endl;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) &device_x, (B*C*H*W) * sizeof(float));
    //cudaMalloc((void **) &device_k, (C*M*K*K) * sizeof(__half));
    cudaMalloc((void **) &device_y, (B*M*H_out*W_out) * sizeof(float));
    if (M == 16) host_k_half = (__half*) malloc(M*C*K*K*sizeof(__half));


    bool val;

    //val = cudaMemcpy(device_k, host_k, (C*M*K*K) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';
    val = cudaMemcpy(device_y, host_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';
    val = cudaMemcpy(device_x, host_x, (B*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);

    if (M == 4){
        cudaMemcpyToSymbol(K_ONE, host_k, M*C*K*K*sizeof(float));
        dim3 dimGridN(ceil(1.0*W_out/(TILE_WIDTH)), ceil(1.0*H_out/(TILE_WIDTH)), B);
        dim3 dimBlockN(TILE_WIDTH, TILE_WIDTH, 1);
        conv_forward_kernel<<<dimGridN, dimBlockN>>>(device_y, device_x, B, M, C, H, W, K);
    }

    if (M == 16) {
        for (int c = 0; c < M*C*K*K; c++)
            host_k_half[c] = __float2half(host_k[c]);

        cudaMemcpyToSymbol(K_TWO, host_k_half, M*C*K*K*sizeof(__half));
        dim3 gridDim(ceil(H_out*W_out/(1.0 * TILE_WIDTH)), ceil(M/(1.0 * TILE_WIDTH)), B);
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<< gridDim, blockDim>>>(device_x, device_y,
                    B, M, C, H, W, K);
    }
    cudaDeviceSynchronize();
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
    if (M == 16) free(host_k_half);
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
