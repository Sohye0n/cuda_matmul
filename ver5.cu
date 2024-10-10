#pragma once
#include "matmul.h"
#include <cassert>
#define CEIL_DIV(M, bkSize) ((M+bkSize-1)/bkSize)

//2D tiling
//block을 tile로 쪼개어 순회한다.
template<unsigned int bmSize, unsigned int bnSize, unsigned int bkSize, unsigned int TM, unsigned int TN>
__global__ void mul5(float* A, float* B, float* C, int M, int N, int K, int alpha, int beta){
    
    extern __shared__ float sharedMem[];
    float* As = sharedMem;
    float* Bs = sharedMem + 32 * bkSize;
    
    float AsReg[TM * bkSize];
    float BsReg[bkSize * TN];
    float threadResults[TM * TN];
    for(int i=0; i<TM; i++){
        for(int j=0; j<TN; j++){
            threadResults[i*TM+j] = 0.0;
        }
    }

    int threadCol = threadIdx.y; // (bnSize / TN);
    int threadRow = threadIdx.x; // (bnSize / TN);

    //이 블록에서 계산해야 하는 C의 원소 개수
    int numResultsPerBlock = 32 * 32;
    //한 스레드에서 계산하길 원하는 C 원소 개수 (2D tiling)
    int numResultsPerThread = TM * TN;
    //블록당 스레드 개수
    int numThreadsPerBlock = numResultsPerBlock / numResultsPerThread;
    
    //한 블록은 A 행렬에서 width = bkSize, height = strideA 만큼의 영역을 한 번에 읽어온다.
    int strideA = numThreadsPerBlock / bkSize;
    //한 블록은 B 행렬에서 width = bnSize, height = strideB 만큼의 영역을 한 번에 읽어온다.
    int strideB = numThreadsPerBlock / bnSize;

    int innerRowA = threadIdx.x; // / bkSize;
    int innerColA = threadIdx.y; // % bkSize;
    int innerRowB = threadIdx.x; // / bnSize;
    int innerColB = threadIdx.y; // % bnSize;

    //0. 포인터를 시작점으로
    A += blockIdx.y * K * 32;
    B += blockIdx.x * 32;
    C += blockIdx.y * N * 32 + blockIdx.x * 32;

    for(int i=0; i<K; i+=bkSize){
        for(int j=0; j<32; j+=strideA){
           As[(innerRowA + j) * bkSize + innerColA] = A[(innerRowA + j) * K + innerColA]; 
        }
        for(int j=0; j<32; j+=strideB){
            Bs[(innerRowB)*32 + innerColB + j] = B[(innerRowB)*N + innerColB + j];
        }
        __syncthreads();

        // if(blockIdx.x==0 && blockIdx.y ==0 && threadCol==2 && threadRow ==1){
        //     for(int i=0; i<bkSize; i++){
        //         for(int j=0; j<32; j++){
        //             printf("%.1f ",Bs[i*32+j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("-------\n");
        // }
 
        A += bkSize;
        B += bkSize * N;
        float cur;

        for(int dotIdx=0; dotIdx<bkSize; ++dotIdx){
            
            //As row들 고정, Bs의 column을 바꿔가며 연산
            for(int i=0; i<TM; ++i){
                AsReg[i] = As[threadRow * TM * bkSize + i*bkSize + dotIdx];
            }
            for(int j=0; j<TN; ++j){
                BsReg[j] = Bs[threadCol * TN + dotIdx * 32 + j];
                if(blockIdx.x==0 && blockIdx.y ==0 && threadCol==2 && threadRow ==1){
                    //printf("BsReg[%d] = Bs[%d] = %.1f\n",j, threadCol*TN + dotIdx * 32 + j,Bs[threadCol * TN + dotIdx * 32 + j]);
                }
            }

            // if(blockIdx.x==0 && blockIdx.y ==0 && threadCol==2 && threadRow ==1){
            //     printf("dot Idx : %d\n",dotIdx);
            //     for(int i=0; i<TM; i++) printf("%.1f ",AsReg[i]);
            //     printf("\n");
            //     for(int j=0; j<TN; j++) printf("%.1f",BsReg[j]);
            //     printf("\n");
            // }

            for(int r=0; r<TM; r++){
                cur = AsReg[r];
                for(int c=0; c<TN; c++){
                    threadResults[r * TN + c] += cur*BsReg[c];
                }
            }
        }
        __syncthreads();
    }

    //global 메모리에 옮겨적기
    for(int r=0; r<TM; r++){
        for(int c=0; c<TN; c++){
            C[(threadRow * TM + r) * N + threadCol * TN + c] = alpha * threadResults[r*TN + c] + beta * C[(threadRow * TM + r) * N + threadCol * TN + c];
            //if(threadCol==2 && threadRow==1) printf("C[%d] : %.1f\n",(threadRow * TM + r) * N + threadCol * TN + c, C[(threadRow * TM + r) * N + threadCol * TN + c]);
        }
    }
}

void mul55(float*A, float* B, int M, int N, int K, int alpha, int beta){
    const int TM = 8;
    const int TN = 8;
    const int blockSize = 32;

    assert(blockSize % TM == 0);
    assert(blockSize % TN ==0);

    const int bkSize = blockSize / TM;
    const int bnSize = blockSize / TN;
    const int bmSize = blockSize / TN;
    dim3 gridDim = dim3(CEIL_DIV(N, blockSize), CEIL_DIV(M,blockSize));
    dim3 blockDim = dim3(bnSize, bkSize);

    //Device memory로 값 복사
    matmul_memcpy_toDevice(A, B, M, N, K);

    printf("mul55\n");
    printf("bm : %d, bn : %d, bk : %d, TM : %d, TN : %d\n",bmSize, bnSize, bkSize, TM, TN);
    //연산 수행
    mul5<bmSize, bnSize, bkSize, TM, TN><<<gridDim, blockDim, 2*sizeof(float)*32*bkSize>>>(A_gpu, B_gpu, C_gpu, M, N, K, alpha, beta);

    //Host memory로 정답 복사
    matmul_memcpy_toHost(M, N);
}