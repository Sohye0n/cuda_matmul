#include "matmul.h"

void matmul_init(int M, int N, int K){
    //device memory 할당
    CHECK_CUDA( cudaMalloc((void**)&A_gpu, sizeof(float)*M*K) );
    CHECK_CUDA( cudaMalloc((void**)&B_gpu, sizeof(float)*K*N) );
    CHECK_CUDA( cudaMalloc((void**)&C_gpu, sizeof(float)*M*N) );
    C_cpu = (float*)malloc(sizeof(float)*M*N);

    CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_memcpy_toDevice(float* A, float* B, int M, int N, int K){
    CHECK_CUDA( cudaMemcpy(A_gpu, A, sizeof(float)*M*K, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(B_gpu, B, sizeof(float)*K*N, cudaMemcpyHostToDevice) );
}

void matmul_memcpy_toHost(int M, int N){
    CHECK_CUDA( cudaMemcpy(C_cpu, C_gpu, sizeof(float)*M*N, cudaMemcpyDeviceToHost) );
}

void matmul_cleanup(){
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_cpu);
    CHECK_CUDA( cudaDeviceSynchronize() );
}