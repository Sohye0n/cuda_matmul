#pragma once
#include "matmul.h"
#include <cassert>

#define CEIL_DIV(M, bkSize) ((M+bkSize-1)/bkSize)

//shared memory coalescing
template<unsigned int bkSize>
__global__ void mul3(float* A, float* B, float* C, int M, int N, int K, int alpha, int beta){
    
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + bkSize * bkSize;

    int threadCol = threadIdx.x;
    int threadRow = threadIdx.y;
    float tmp = 0.0;
    
    //0. 시작 포인터 이동
    A += blockIdx.y * bkSize * K;
    B += blockIdx.x * bkSize;
    C += ( blockIdx.y * N + blockIdx.x ) * bkSize;

    for(int i=0; i<K; i+=bkSize){
        
        //1. shared memory에 로드해오기
        As[threadRow * bkSize + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * bkSize + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        A += bkSize;
        B += bkSize * N;

        //2. 곱하기
        for(int j=0; j<bkSize; j++){
            tmp += As[threadRow * bkSize + j] * Bs[j * bkSize + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];    
}

float *A_gpu, *B_gpu, *C_gpu;
float *C_cpu;

void mul33(float* A, float* B, int M, int N, int K, int alpha, int beta){
    const int blockSize = 32;
    dim3 gridDim = dim3(CEIL_DIV(N,blockSize), CEIL_DIV(M,blockSize));
    dim3 blockDim = dim3(32,blockSize);
    
    //Device memory로 값 복사
    matmul_memcpy_toDevice(A, B, M, N, K);

    //연산 수행
    mul3<blockSize><<<gridDim, blockDim, 2*sizeof(float)*blockSize*blockSize>>>(A_gpu, B_gpu, C_gpu, M, N, K, alpha, beta);

    //Host memory로 정답 복사
    matmul_memcpy_toHost(M, N);

}