#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 32
#define KERNEL_SIZE 7
#define RADIUS 3
#define O_TILE_WIDTH TILE_WIDTH-6
#define ELEMENTS_PER_THREAD 8
#define INIT_BLOCK_SIZE 256

__constant__ float K_ONE[4*KERNEL_SIZE*KERNEL_SIZE];
__constant__ float K_TWO[16*4*KERNEL_SIZE*KERNEL_SIZE];

// TODO: Initialize y to zero (will be made redundant after input channel tree reduction)
// Investigate coallescing in shared memory?

__global__ void initialize_array(float *y, const int num) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int dx = blockDim.x;
    int stride = dx;
    int x = bx * dx * ELEMENTS_PER_THREAD + tx;
    for (int c = 0; c < ELEMENTS_PER_THREAD; c++) {
        if (x < num)
            y[x] = 0;
        x += stride;
    }
}

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
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k1_4d(i3, i2, i1, i0) K_ONE[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k2_4d(i3, i2, i1, i0) K_TWO[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    __shared__ float D[TILE_WIDTH][TILE_WIDTH];
    int x_idx = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
    int y_idx = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
    int c = blockIdx.z % C;
    int b = blockIdx.z / C;
    
    // Load into shared memory
    D[threadIdx.y][threadIdx.x] = (x_idx<0||y_idx<0||x_idx>=W||y_idx>=H) ? 0 : x4d(b,c,y_idx,x_idx);
    __syncthreads();
    // compute for each kernel
    if (threadIdx.x >= RADIUS && threadIdx.y >= RADIUS && threadIdx.x < TILE_WIDTH-RADIUS && threadIdx.y < TILE_WIDTH-RADIUS)
    for (int m = 0; m < M; m++) {
        float acc = 0;
        // iterate over y
        for (int p = -RADIUS; p < RADIUS+1; p++) {
            // iterate over x
            for (int q = -RADIUS; q < RADIUS+1; q++) {
                if (M == 4)
                    acc += D[threadIdx.y+p][threadIdx.x+q] * k1_4d(m,c,p+RADIUS,q+RADIUS);
                else 
                    acc += D[threadIdx.y+p][threadIdx.x+q] * k2_4d(m,c,p+RADIUS,q+RADIUS);
            }
        }
        if (x_idx>=RADIUS&&y_idx>=RADIUS&&x_idx<W_out+RADIUS&&y_idx<H_out+RADIUS)
            atomicAdd(&y4d(b, m, y_idx-RADIUS, x_idx-RADIUS), acc);
    }

#undef y4d
#undef x4d
#undef k4d
#undef k1_4d
#undef k2_4d

}

/*__global__ void tree_reduce(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define y5d(i4, i3, i2, i1, i0) y[(i4) * (B * M * H_out * W_out) + (i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    __shared__ float D[O_TILE_WIDTH][O_TILE_WIDTH][16]; // Change to 6
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z / M;
    int m = blockIdx.z % M;
    int c = threadIdx.z;
    
    // Load into shared memory
    D[threadIdx.x][threadIdx.y][threadIdx.z] = (x_idx<0||y_idx<0||x_idx>=W_out||y_idx>=H_out) ? 0 : x4d(c);
    __syncthreads();
    // compute for each kernel
    float val = 0;
    if (c == 0)
        for (unsigned int c_idx=0; c_idx < C; c_idx++) {
            val += D[threadIdx.x][threadIdx.y][c_idx];
        }
    if (c==0&&x_idx>=0&&y_idx>=0&&x_idx<W_out&&y_idx<H_out)
        y5d(c, b, m, y_idx, x_idx) = val;

#undef y4d
#undef y5d
#undef x4d
#undef k4d
}*/


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
    //cudaMalloc((void **) &device_k, (C*M*K*K) * sizeof(float));
    bool val;
    val = cudaMemcpy(device_x, host_x, (B*C*H*W) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    val = cudaMemcpy(device_y, host_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

    //val = cudaMemcpy(device_k, host_k, (C*M*K*K) * sizeof(float), cudaMemcpyHostToDevice);
    //std::cout << std::boolalpha << val << '\n';

	if (M == 4)	cudaMemcpyToSymbol(K_ONE, host_k, M*C*K*K*sizeof(float));
	if (M == 16) cudaMemcpyToSymbol(K_TWO, host_k, M*C*K*K*sizeof(float));

    // Set the kernel dimensions and call the kernel
    //int W_grid = ceil((float)(1.0*W_out)/(1.0 * O_TILE_WIDTH)); // number of horizontal tiles per output map
    //int H_grid = ceil((float)(1.0*H_out)/(1.0 * O_TILE_WIDTH)); // number of vertical tiles per output map
    //int Y = H_grid * W_grid;
    //dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //dim3 gridDim(M, Y, C*B);
    dim3 dimGridInit(ceil(1.0*B*M*H_out*W_out/(INIT_BLOCK_SIZE*ELEMENTS_PER_THREAD)), 1, 1);
    dim3 dimBlockInit(INIT_BLOCK_SIZE, 1, 1);
    //initialize_array<<<dimGridInit, dimBlockInit>>>(device_y, B*M*H_out*W_out);

    dim3 dimGrid(ceil(1.0*W_out/(O_TILE_WIDTH)), ceil(1.0*H_out/(O_TILE_WIDTH)), C*B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel<<< dimGrid, dimBlock>>>(
        device_y, device_x, device_k, 
        B,M, C, H, W, K
    );
    // Copy the output back to host
    val = cudaMemcpy(host_y, device_y, (B*M*H_out*W_out) * sizeof(float), cudaMemcpyDeviceToHost);
    //std::cout << std::boolalpha << val << '\n';
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    //cudaFree(device_k);


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
