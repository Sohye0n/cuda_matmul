#include "util.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double get_current_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

float* fill_matrix(int M, int K) {

    float* A = (float*) malloc(sizeof(float)*M*K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = (float)(i * K + j + 1);
        }
    }

    return A;
}

void clean_matrix(float* A){
    free(A);
    return;
}

void print_result(float* C, int M, int N){
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            printf("%3f",C[i*M+j]);
        }
        printf("\n");
    }
}

void compare_result(float* A, float* B, float* C, int M, int N, int K){

    float ans;

    //C[2][4]    
    ans = 0.0f;
    for(int i=0; i<K; i++){
        ans+=A[2*K+i]*B[i*N+4];
    }
    printf("ans : %f, matmul result : %f\n",ans, C[2*N+4]);


    //C[0][2]
    ans = 0.0f;
    for(int i=0; i<K; i++){
        ans+=A[0*K+i]*B[i*N+2];
    }
    printf("ans : %f, matmul result : %f\n",ans, C[0*N+2]);


    //C[95][63]
    ans = 0.0f;
    for(int i=0; i<K; i++){
        ans+=A[95*K+i]*B[i*N+63];
    }
    printf("ans : %f, matmul result : %f\n",ans, C[95*N+63]);
}