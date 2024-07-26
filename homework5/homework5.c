#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void print_matrix(double** T, int rows, int cols);

int main(int argc, char* argv[])
{
    int n = 1000;
    double* a0 = (double*)malloc(n * n * sizeof(double));
    double* c0 = (double*)malloc(n * n * sizeof(double));
    double** A = (double**)malloc(n * sizeof(double*));
    double** C = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        A[i] = a0 + i * n;
        C[i] = c0 + i * n;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i][j] = (double)rand() / RAND_MAX;
    double start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            int k;
            for (k = 0; k < n - 4; k += 4) {
                sum += A[i][k] * A[j][k];
                sum += A[i][k+1] * A[j][k+1];
                sum += A[i][k+2] * A[j][k+2];
                sum += A[i][k+3] * A[j][k+3];
            }
            for (; k < n; k++) {
                sum += A[i][k] * A[j][k];
            }
            C[i][j] = sum;
        }
    }

    double end_time = omp_get_wtime();
    printf("Matrix multiplication time: %f seconds\n", end_time - start_time);

    free(a0);
    free(c0);
    free(A);
    free(C);

    return 0;
}

void print_matrix(double** T, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f   ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
