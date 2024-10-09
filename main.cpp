#include <getopt.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "matmul.h"
#include "util.h"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-h] [-v version] [-m M] [-k K] [-n N]\n", prog_name);
  printf("Options:\n");
  printf("  -h : print this page.\n");
  printf("  -v  : version of matmul (default: 3)\n");
  printf("  -m  : height of mat A (default: 8 * 32)\n");
  printf("  -k  : width of mat A & height of mat B (default: 2 * 32)\n");
  printf("  -n  : height of mat B (default: 4 * 32)\n");
}

static int ver = 3;
static int M = 1 * 32 ;
static int K = 2 * 32;
static int N = 1 * 32;
static int alpha = 1;
static int beta = 0;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "hn:v:m:k:n")) != -1) {
    switch (c) {
      case 'v': ver = atoi(optarg); break;
      case 'm': M = atoi(optarg); break;
      case 'k': K = atoi(optarg); break;
      case 'n': N = atoi(optarg); break;
      default: print_help(argv[0]); exit(0);
    }
  }
  printf("Options:\n");
  printf("  Version of multiplication: %d\n", ver);
  printf("  M : %d\n", M);
  printf("  K : %d\n", K);
  printf("  N : %d\n", N);
  printf("\n");
}

void matmul(int ver, float* A, float* B, int M, int N, int K, int alpha, int beta){
    
    switch(ver){
        case 3:
            mul33(A, B, M, N, K, alpha, beta);
            break;

        case 4:
            mul44(A, B, M, N, K, alpha, beta);
            break;

        case 5:
            mul55(A, B, M, N, K, alpha, beta);
            break;
    }
}

int main(int argc, char **argv) {
  parse_opt(argc, argv);

  printf("Initializing... ");
  fflush(stdout);
  matmul_init(M, N, K);
  printf("done!\n");
  fflush(stdout);

  /* matmul 계산 */
  printf("Calculating... ");
  fflush(stdout);

  /*----- A, B 생성 -----*/
  float* A = fill_matrix(M, K);
  float* B = fill_matrix(K, N);

  double start_time = get_current_time();
  matmul(ver, A, B, M, N, K, alpha, beta);
  double elapsed_time = get_current_time() - start_time;
  printf("done!\n");

  /* Print results */
  printf("C result : \n");
  compare_result(A, B, C_cpu, M, N, K);
  printf("Elapsed time: %.3f sec\n", elapsed_time);

  clean_matrix(A);
  clean_matrix(B);
  clean_matrix(C_cpu);

  return 0;
}