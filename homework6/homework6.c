//使用 MPI（消息传递接口）库来实现进程间的通信和协作

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// 定义矩阵的大小
#define N 1000

// 矩阵初始化函数
void initialize_matrix(double* matrix, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i * size + j] = (double)rand() / RAND_MAX;
}

// 矩阵乘法的局部计算
void local_matrix_multiply(double* local_A, double* local_C, int local_n, int n, int rank, int size) {
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * local_A[j * n + k];
            }
            local_C[i * n + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int local_n = N / num_procs; // 每个进程处理的矩阵块的大小

    // 分配内存
    double* A = NULL;
    double* local_A = (double*)malloc(local_n * N * sizeof(double));
    double* local_C = (double*)malloc(local_n * N * sizeof(double));

    // 主进程初始化矩阵并分发
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        initialize_matrix(A, N);

        // 将矩阵 A 分发到所有进程
        for (int i = 0; i < num_procs; i++) {
            MPI_Send(A + i * local_n * N, local_n * N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_A, local_n * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 执行局部矩阵乘法
    local_matrix_multiply(local_A, local_C, local_n, N, rank, num_procs);

    // 汇总结果
    double* C = NULL;
    if (rank == 0) {
        C = (double*)malloc(N * N * sizeof(double));
    }

    MPI_Gather(local_C, local_n * N, MPI_DOUBLE, C, local_n * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 输出结果
    if (rank == 0) {
        printf("Matrix multiplication complete.\n");
        // 这里可以选择打印矩阵或进行其他操作
        free(A);
        free(C);
    }

    // 释放内存
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
