#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//基础矩阵乘法函数
void matrix_multiply(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
//循环展开优化的矩阵乘法函数
void matrix_multiply_unrolled(int n, double *A, double *B, double *C) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (k = 0; k <= n - 4; k += 4) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
                C[i * n + j] += A[i * n + (k + 1)] * B[(k + 1) * n + j];
                C[i * n + j] += A[i * n + (k + 2)] * B[(k + 2) * n + j];
                C[i * n + j] += A[i * n + (k + 3)] * B[(k + 3) * n + j];
            }
            for (; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main() {
    int n = 1000; // 矩阵大小
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = i + j + 1;
            B[i * n + j] = (i + j + 1) * 2;
        }
    }
    clock_t start = clock();
    matrix_multiply(n, A, B, C);
    clock_t end = clock();
    double time_no_unroll = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken without loop unrolling: %f seconds\n", time_no_unroll);


    start = clock();
    matrix_multiply_unrolled(n, A, B, C);
    end = clock();
    double time_unroll = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken with loop unrolling: %f seconds\n", time_unroll);
    double performance_gain = ((time_no_unroll - time_unroll) / time_no_unroll) * 100;
    printf("Performance gain with loop unrolling: %f%%\n", performance_gain);

    // 释放动态分配的内存
    free(A);
    free(B);
    free(C);

    return 0;
}
