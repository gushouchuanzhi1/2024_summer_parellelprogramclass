//我们需要实现顺序扫描的函数，并且需要比较顺序执行和并行执行的效率。
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// 顺序扫描操作
void sequential_scan(int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// 使用OpenMP实现并行扫描操作
void parallel_scan(int *input, int *output, int n) {
    int num_threads, step;

    // 第一步：上扫（归约）阶段
    for (step = 1; step < n; step *= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 2 * step) {
            if (i + step < n) {
                input[i + 2 * step - 1] += input[i + step - 1];
            }
        }
    }

    // 第二步：下扫（分发）阶段
    input[n - 1] = 0;
    for (step = n / 2; step > 0; step /= 2) {
        #pragma omp parallel for
        for (int i = 0; i < n; i += 2 * step) {
            if (i + step < n) {
                int temp = input[i + step - 1];
                input[i + step - 1] = input[i + 2 * step - 1];
                input[i + 2 * step - 1] += temp;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        output[i] = input[i];
    }
}

int main() {
    int n = 1000000;
    int *input = (int *)malloc(n * sizeof(int));
    int *output_seq = (int *)malloc(n * sizeof(int));
    int *output_par = (int *)malloc(n * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        input[i] = rand() % 10;
    }

    clock_t start_seq = clock();
    sequential_scan(input, output_seq, n);
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;
    printf("顺序扫描耗时: %f 秒\n", time_seq);

    double start_par = omp_get_wtime();
    parallel_scan(input, output_par, n);
    double end_par = omp_get_wtime();
    double time_par = end_par - start_par;
    printf("并行扫描耗时: %f 秒\n", time_par);

    int correct = 1; 
    for (int i = 0; i < n; i++) {
        if (output_seq[i] != output_par[i]) {
            correct = 0;
            printf("结果不匹配：索引 %d：顺序 = %d, 并行 = %d\n", i, output_seq[i], output_par[i]);
            break;
        }
    }
    if (correct) {
        printf("顺序扫描和并行扫描的结果一致。\n");
    }

    double performance_gain = ((time_seq - time_par) / time_seq) * 100;
    printf("性能提升：%f%%\n", performance_gain);

    free(input);
    free(output_seq);
    free(output_par);

    return 0;
}
