#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 4
#define KERNEL_SIZE 7

__constant__ float K_ONE[4*KERNEL_SIZE*KERNEL_SIZE];
__constant__ float K_TWO[16*4*KERNEL_SIZE*KERNEL_SIZE];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
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

    const int H_out= H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = ceil(W_out/(1.0 * TILE_WIDTH)); // number of horizontal tiles per output map
    //const int H_grid = ceil(H_out/(1.0 * TILE_WIDTH)); // number of vertical tiles per output ma

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define k1_4d(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define k2_4d(i3, i2, i1, i0) K_TWO[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // Insert your GPU convolution kernel code here
    // int h = blockIdx.y / W_grid + threadIdx.y;
    // int w = blockIdx.y % W_grid + threadIdx.x;
    // float acc = 0.;
    // for (int c = 0; c < C; c++) { // sum over all input channels
    //     for (int p = 0; p < K; p++) // loop over KxK filter
    //     for (int q = 0; q < K; q++)
    //     acc += X[c, h + p, w + q] * W[m, c, p, q];
    // }
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    if(h<H_out && w<W_out){
        for(int b=0; b < B; b++) {
            y4d(b, m, h, w) = 0.0;
            float acc = 0.;
            for (int c = 0; c < C; c++){ // sum over all input channels
                for (int p = 0; p < K; p++) // loop over KxK filter
                    for (int q = 0; q < K; q++){
						if (M == 4)
                            acc += x4d(b,c, h + p, w + q) * k1_4d(m, c, p, q);
						else
                            acc += x4d(b,c, h + p, w + q) * k2_4d(m, c, p, q);
                    }
            }
                    
        y4d(b, m, h, w) = acc;
        }   
    }
 

    #undef y4d
    #undef x4d
    #undef k4d
	#undef k1_4d
	#undef k2_4d
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float *device_y;
    float *device_x;
    float *device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    cudaMalloc((void **) &device_x, (B*C*H*W) * sizeof(float));
    cudaMalloc((void **) &device_y, (B*M*H_out*W_out) * sizeof(float));
    cudaMalloc((void **) &device_k, (C*M*K*K) * sizeof(float));
    bool val;
    val = cudaMemcpy(device_x, host_x, (B*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    val = cudaMemcpy(device_y, host_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    val = cudaMemcpy(device_k, host_k, (C*M*K*K) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

	if (M == 4)	cudaMemcpyToSymbol(K_ONE, host_k, M*C*K*K*sizeof(float));
	if (M == 16) cudaMemcpyToSymbol(K_TWO, host_k, M*C*K*K*sizeof(float));

    // Set the kernel dimensions and call the kernel
    int W_grid = ceil((float)(1.0*W_out)/(1.0 * TILE_WIDTH)); // number of horizontal tiles per output map
    int H_grid = ceil((float)(1.0*H_out)/(1.0 * TILE_WIDTH)); // number of vertical tiles per output map
    int Y = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, 1);

    conv_forward_kernel<<< gridDim, blockDim>>>(
        device_y, device_x, device_k, 
        B,M, C, H, W, K
    );
    // Copy the output back to host
    val = cudaMemcpy(host_y, device_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost);
    //std::cout << std::boolalpha << val << '\n';
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);


    //Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

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
