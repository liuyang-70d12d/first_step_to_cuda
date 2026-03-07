#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__host__ __device__ float f(float a, float b) {
    return a + b;
}

// CPU版本向量加法
void vecadd_cpu(float *x, float *y, float *z, int N)
{
    for (unsigned int i = 0; i < N; ++i)
    {
        z[i] = f(x[i], y[i]);
    }
}

// GPU核函数
__global__ void vecadd_kernel(float *x, float *y, float *z, int N)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = f(x[i], y[i]);
    }
}

// GPU版本向量加法（包含计时）
void vecadd_gpu(float *x, float *y, float *z, int N, float &gpu_time)
{
    // 记录GPU开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1. 分配GPU内存
    float *x_d, *y_d, *z_d;
    cudaMalloc((void **)&x_d, N * sizeof(float));
    cudaMalloc((void **)&y_d, N * sizeof(float));
    cudaMalloc((void **)&z_d, N * sizeof(float));

    // 2. 主机到设备数据拷贝
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 启动核函数（开始计时）
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    cudaEventRecord(start);
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaEventRecord(stop);

    // 4. 设备到主机数据拷贝
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 计算GPU耗时（毫秒）
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    // 5. 释放GPU内存和事件
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 结果验证函数：比较CPU和GPU结果是否一致
bool verify_results(float *cpu_z, float *gpu_z, int N, float tolerance = 1e-5f)
{
    for (unsigned int i = 0; i < N; ++i)
    {
        if (fabsf(cpu_z[i] - gpu_z[i]) > tolerance)
        {
            printf("结果验证失败！第 %d 个元素不匹配: CPU=%.4f, GPU=%.4f\n",
                   i, cpu_z[i], gpu_z[i]);
            return false;
        }
    }
    printf("✅ 所有 %d 个元素结果验证通过！\n", N);
    return true;
}

int main(int argc, char **argv)
{
    // 设置随机数种子（确保每次测试数据一致）
    srand(12345);

    // 1. 初始化参数（默认2^25个元素，可通过命令行参数修改）
    unsigned int N = (argc > 1) ? (atoi(argv[1])) : (1 << 25);
    printf("向量长度 N = %u (%.2f MB 每向量)\n", N, N * sizeof(float) / 1024.0f / 1024.0f);

    // 2. 分配主机内存并初始化数据
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *cpu_z = (float *)malloc(N * sizeof(float)); // CPU计算结果
    float *gpu_z = (float *)malloc(N * sizeof(float)); // GPU计算结果

    if (!x || !y || !cpu_z || !gpu_z)
    {
        printf("❌ 主机内存分配失败！\n");
        return -1;
    }

    // 初始化输入向量
    for (unsigned int i = 0; i < N; ++i)
    {
        x[i] = (float)rand() / RAND_MAX * 100.0f; // 0~100的随机数
        y[i] = (float)rand() / RAND_MAX * 100.0f;
    }

    // 3. CPU版本计算（计时）
    clock_t cpu_start = clock();
    vecadd_cpu(x, y, cpu_z, N);
    clock_t cpu_end = clock();
    float cpu_time = (float)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0f; // 转换为毫秒

    // 4. GPU版本计算（计时）
    float gpu_time = 0.0f;
    vecadd_gpu(x, y, gpu_z, N, gpu_time);

    // 5. 验证结果
    verify_results(cpu_z, gpu_z, N);

    // 6. 输出性能数据
    printf("\n===== 性能对比 =====\n");
    printf("CPU 执行时间: %.2f 毫秒\n", cpu_time);
    printf("GPU 执行时间: %.2f 毫秒 (仅核函数耗时)\n", gpu_time);
    printf("GPU 加速比:   %.2f 倍\n", cpu_time / gpu_time);

    // 7. 释放主机内存
    free(x);
    free(y);
    free(cpu_z);
    free(gpu_z);

    return 0;
}