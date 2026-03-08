#include "timer.h"
#define BLUR_SIZE 1

__global__ void blur_kernel(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width)
    {
        unsigned int average = 0;
        for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow)
        {
            for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol)
            {
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                {
                    average += image[inRow * width + inCol];
                }
            }
        }

        blurred[outRow * width + outCol] = (unsigned char)(average / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1)));
    }
}

void blur_gpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void **)&image_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&blurred_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(image_d, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(blurred, blurred_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(image_d);
    cudaFree(blurred_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}

void blur_cpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    for (int outRow = 0; outRow < height; ++outRow)
    {
        for (int outCol = 0; outCol < width; ++outCol)
        {
            unsigned int average = 0;
            for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow)
            {
                for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol)
                {
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                    {
                        average += image[inRow * width + inCol];
                    }
                }
            }

            blurred[outRow * width + outCol] = (unsigned char)(average / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1)));
        }
    }
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int width = 32000;
    unsigned int height = 32000;
    unsigned char *image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blurred_cpu = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blurred_gpu = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    unsigned int i = 0;
    for (unsigned int row = 0; row < height; ++row)
    {
        for (unsigned int col = 0; col < width; ++col)
        {
            i = row * height + col;
            image[i] = (unsigned char)(rand() % 256);
        }
    }

    // Compute on CPU
    startTime(&timer);
    blur_cpu(image, blurred_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    blur_gpu(image, blurred_gpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Free memory
    free(image);
    free(blurred_cpu);
    free(blurred_gpu);

    return 0;
}
