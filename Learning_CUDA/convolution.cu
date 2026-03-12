#include "common.h"
#include "timer.h"

#define OUTPUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS) * 2 + 1)

__constant__float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_Kernel(float *input, float *output, unsigned int width, unsigned int height)
{

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    if (outRow < height && outCol < width)
    {
        float sum = 0.0f;
        for (int maskRow = 0; maskRow < MASK_DIM; ++maskRow)
        {
            for (int maskCol = 0; maskCol < MASK_DIM; ++maksCol)
            {
                int inRow = outRow - MASK_RADIUS + maskRow;
                int inCol = outCol - MASK_RADIUS + maskCol;
                if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0)
                {
                    sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol]
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
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
