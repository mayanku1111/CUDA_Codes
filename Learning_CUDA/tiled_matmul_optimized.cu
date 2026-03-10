#include "common.h"
#include "timer.h"

#define TILE_DIM 32
#define COARSE_FACTOR 4

// As per the tiled_matmul_optimized.png , we can see that we can face issue when let's say tile(1,1) is loaded onto
// Shared memory of SM1 and let's say tile(1,2) is loaded onto another shared memory of another SM like SM2

__global__ void mm_tiled_kernel(float *A, float *B, float *C, unsigned int N)
{

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    float sum[COARSE_FACTOR];
    for (unsigned int c = 0; c < coarse_factor; c++)
    {
        sum[c] = 0.0f
    }

    for (unsigned int tile = 0; tile < N / TILE_DIM; ++tile)
    {
        A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_DIM + threadIdx.x];

        for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
        {
            unsigned col = colStart + c * TILE_DIM;
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * N + col];
            __syncthreads();
            for (unsigned int i = 0; i < TILE_DIM; ++i)
            {
                sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c)
    {
        unsigned int col = colStart + c * TILE_DIM;
        C[row * N + col] = sum[c];
    }
}

void mm_gpu(float *A, float *B, float *C, unsigned int N)
{

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, N * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * N * sizeof(float));
    cudaMalloc((void **)&C_d, N * N * sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);

    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);

    dim3 numThreadsPerBlock(TILE_DIM, TILE_DIM);

    dim3 numBlocks(
        (N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x / COARSE_FACTOR,
        (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    mm_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);

    cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}