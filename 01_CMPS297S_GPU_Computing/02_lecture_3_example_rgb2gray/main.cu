#include "timer.h"

void rgb2gray_cpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    unsigned int i = 0;
    for (unsigned int row = 0; row < height; ++row)
    {
        for (unsigned int col = 0; col < width; ++col)
        {
            i = row * height + col;
            gray[i] = red[i] * 3 / 10 + green[i] * 6 / 10 + blue[i] * 1 / 10;
        }
    }
}

__global__ void rgb2gray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert the pixel
    if (row < height && col < width)
    {
        unsigned int i = row * width + col;
        gray[i] = red[i] * 3 / 10 + green[i] * 6 / 10 + blue[i] * 1 / 10;
    }
}

void rgb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void **)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Free GPU memory
    startTime(&timer);
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // Allocate memory and initialize data
    Timer timer;
    unsigned int width = 32000;
    unsigned int height = 32000;
    unsigned char *red = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *green = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blue = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *gray_cpu = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *gray_gpu = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    unsigned int i = 0;
    for (unsigned int row = 0; row < height; ++row)
    {
        for (unsigned int col = 0; col < width; ++col)
        {
            i = row * height + col;
            red[i] = (unsigned char)(rand() % 256);
            green[i] = (unsigned char)(rand() % 256);
            blue[i] = (unsigned char)(rand() % 256);
        }
    }

    // Compute on CPU
    startTime(&timer);
    rgb2gray_cpu(red, green, blue, gray_cpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "CPU time", CYAN);

    // Compute on GPU
    startTime(&timer);
    rgb2gray_gpu(red, green, blue, gray_gpu, width, height);
    stopTime(&timer);
    printElapsedTime(timer, "GPU time", DGREEN);

    // Free memory
    free(red);
    free(green);
    free(blue);
    free(gray_cpu);
    free(gray_gpu);

    return 0;
}
