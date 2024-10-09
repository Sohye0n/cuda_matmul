#pragma once

double get_current_time();
float* fill_matrix(int M, int K);
void clean_matrix(float* A);
void print_result(float* C, int M, int N);
void compare_result(float* A, float* B, float* C, int M, int N, int K);