#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void print_matrix(double** T, int rows, int cols);

int main(int argc, char* argv[])
{
    double* a0; // Auxiliary 1D array for 2D matrix a
    double** a; // 2D matrix for computation
    
    int n; // Input size
    int i, j, k;
    int indk;
    double c, amax;

    clock_t start_time, end_time;
    double elapsed;

    // Set default matrix size
    n = 1000;

    if (argc == 2)
    {
        n = atoi(argv[1]); // Convert argument to integer
        if (n <= 0) {
            printf("Invalid matrix size. Using default size: %d\n", n);
            n = 1000;
        }
    }

    printf("The matrix size:  %d * %d \n", n, n);

    printf("Creating and initializing matrices...\n\n"); 
    /*** Allocate contiguous memory for 2D matrices ***/
    a0 = (double*)malloc(n * n * sizeof(double));
    a = (double**)malloc(n * sizeof(double*));
    for (i = 0; i < n; i++)
    {
        a[i] = a0 + i * n;
    }

    srand(time(0));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i][j] = (double)rand() / RAND_MAX;

    //    printf("matrix a: \n");
    //    print_matrix(a, n, n);

    printf("Starting parallel computation...\n\n"); 
    /**** Parallel Gaussian elimination *****/
    start_time = clock();

    #pragma omp parallel private(i, j, k, indk, amax, c) shared(a, n)
    {
        for (i = 0; i < n - 1; i++)
        {
            // Find and record k where |a(k,i)| = max |a(j,i)|
            amax = a[i][i];
            indk = i;
            #pragma omp for
            for (k = i + 1; k < n; k++)
            {
                if (fabs(a[k][i]) > fabs(amax))
                {
                    #pragma omp critical
                    {
                        if (fabs(a[k][i]) > fabs(amax)) {
                            amax = a[k][i];
                            indk = k;
                        }
                    }
                }
            }

            // Exit with a warning that a is singular
            if (amax == 0)
            {
                printf("Matrix is singular!\n");
                exit(1);
            }  
            else if (indk != i) // Swap row i and row k
            {
                for (j = 0; j < n; j++)
                {
                    c = a[i][j];
                    a[i][j] = a[indk][j];
                    a[indk][j] = c;
                }
            } 

            // Store multiplier in place of A(k,i)
            #pragma omp for
            for (k = i + 1; k < n; k++)
            {
                a[k][i] = a[k][i] / a[i][i];
            }

            // Subtract multiple of row a(i,:) to zero out a(j,i)
            #pragma omp for private(j, c)
            for (k = i + 1; k < n; k++)
            { 
                c = a[k][i]; 
                for (j = i + 1; j < n; j++)
                {
                    a[k][j] -= c * a[i][j];
                }
            }
        }
    }

    end_time = clock();
    elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Parallel calculation time: %f seconds\n\n", elapsed); 

    // Free allocated memory
    free(a0);
    free(a);

    return 0;
}

void print_matrix(double** T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f   ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
}
