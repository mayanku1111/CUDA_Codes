#include "common.h"
#include "timer.h"

#define OUTPUT_TILE_DIM 32
__constant__float mask_c[MASK_DIM][MASK_DIM];
__global__ void convolution_Kernel(float *input, float *output, unsigned int width, unsigned int height)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
}

void convolution_gpu(float mask[][MASK_DIM], float *input, float *output, unsigned int width, unsigned int height)
{

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *input_d, *output_d;
    cudaMalloc((void **)&input_d, width * height * sizeof(float));
    cudaMalloc((void **)&output_d, width * height * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Copy mask to constant memory
    startTime(&timer);

    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU constant memory time");

    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(input_d);
    cudaFree(output_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}