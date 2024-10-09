#pragma once
#include <cstdio>

#define CHECK_CUDA(call)                                             \
    do {                                                            \
        cudaError_t status_ = call;                                \
        if (status_ != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error (%s:%d): %s: %s\n",       \
                    __FILE__, __LINE__,                            \
                    cudaGetErrorName(status_),                    \
                    cudaGetErrorString(status_));                 \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

extern float *A_gpu, *B_gpu, *C_gpu;
extern float *C_cpu;

/*------ mem init ------*/
void matmul_init(int M, int N, int K);
void matmul_memcpy_toDevice(float* A, float* B, int M, int N, int K);
void matmul_memcpy_toHost(int M, int N);
void matmul_cleanup();


/*----- matmul versions -----*/
void mul33(float* A, float* B, int M, int N, int K, int alpha, int beta);
void mul44(float* A, float* B, int M, int N, int K, int alpha, int beta);
void mul55(float* A, float* B, int M, int N, int K, int alpha, int beta);

/*----- matmul interface -----*/