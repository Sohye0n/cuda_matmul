#pragma once
#include "matmul.h"
#include <cassert>

#define CEIL_DIV(M, bkSize) ((M+bkSize-1)/bkSize)

// 1D tiling
// bnSize = bmSize
template<unsigned int bmSize, unsigned int bnSize, unsigned int bkSize, unsigned int TM>
__global__ void mul4(float* A, float* B, float* C, int M, int N, int K, int alpha, int beta){
    
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + bmSize * bkSize;
    float threadResults[bmSize * bkSize];
    for(int i=0; i<bkSize; i++){
        for(int j=0; j<bmSize; j++){
            threadResults[i*bmSize+j] = 0.0;
        }
    }

    int threadCol = threadIdx.x % bnSize;
    int threadRow = threadIdx.x / bnSize;

    int innerColA = threadIdx.x % bkSize;
    int innerRowA = threadIdx.x / bkSize;
    int innerColB = threadIdx.x % bnSize;
    int innerRowB = threadIdx.x / bnSize;

    //0. 포인터를 시작점으로 이동
    A += blockIdx.y * K * bmSize;
    B += blockIdx.x * bnSize;
    C += blockIdx.y * N * bmSize + blockIdx.x * bnSize;

    for(int i=0; i<K; i+=bkSize){

        //1. shared memory에 로드하기
        //일종의 타일링...
        As[innerRowA * bkSize + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * bnSize + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A+=bkSize;
        B+=bkSize*N;

        //2. 곱하기. 여기가 병렬화 + 타일링. TM = BM / BK. (BM*BK개 스레드로 BM*BM개의 효과를 내야 해서)
        float cur;
        //공통 축에 대해 반복
        for(int dotIdx=0; dotIdx<bkSize; ++dotIdx){
            cur = Bs[dotIdx * bnSize + threadCol];
            //B의 한 원소를 읽었을 때 TM 개씩 처리함.
            for(int resIdx=0; resIdx<TM; ++resIdx){
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * bkSize + dotIdx] * cur;
            }
        }
        __syncthreads();
    }

    //3. local 배열에 저장된 값을 global 메모리로 옮긴다.
    for(int resIdx=0; resIdx<TM; ++resIdx){
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx)*N + threadCol];
    }
}


void mul44(float* A, float* B, int M, int N, int K, int alpha, int beta){
    //32 배수로 설정 가능
    const int blockSize = 64;
    const int TM = 8;
    assert(blockSize % TM ==0);

    const int bkSize = blockSize / TM;
    dim3 gridDim = dim3(CEIL_DIV(N,blockSize), CEIL_DIV(M,blockSize));
    dim3 blockDim = dim3(64 * bkSize);

    //Device memory로 값 복사
    matmul_memcpy_toDevice(A, B, M, N, K);

    //연산 수행
    mul4<blockSize, blockSize, bkSize, TM><<<gridDim, blockDim, 2*sizeof(float)*blockSize*bkSize>>>(A_gpu, B_gpu, C_gpu, M, N, K, alpha, beta);

    //Host memory로 정답 복사
    matmul_memcpy_toHost(M, N);
}