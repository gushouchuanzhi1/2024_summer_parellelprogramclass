#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void print_matrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void special_gaussian_elimination(double* matrix, int n) {
    for (int i = 0; i < n - 1; i++) {
        // 主对角线元素为零的情况下
        if (matrix[i * n + i] == 0) {
            printf("Matrix is singular!\n");
            exit(1);
        }
        for (int k = i + 1; k < n; k++) {
            double factor = matrix[k * n + i] / matrix[i * n + i];
            for (int j = i; j < n; j++) {
                matrix[k * n + j] -= factor * matrix[i * n + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 7;
    double* matrix = NULL;

    if (rank == 0) {
        matrix = (double*)malloc(n * n * sizeof(double));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j == i + 1) {
                    matrix[i * n + j] = (double)rand() / RAND_MAX;
                } else {
                    matrix[i * n + j] = 0.0;
                }
            }
        }
    }
    if (rank == 0) {
        MPI_Bcast(matrix, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        matrix = (double*)malloc(n * n * sizeof(double));
        MPI_Bcast(matrix, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    special_gaussian_elimination(matrix, n);
    if (rank == 0) {
        print_matrix(matrix, n);
        free(matrix);
    } else {
        free(matrix);
    }

    MPI_Finalize();
    return 0;
}
