#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000

void matrix_multiply(int **A, int **B, int **C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    
    int **matrixA = (int **)malloc(N * sizeof(int *));
    int **matrixB = (int **)malloc(N * sizeof(int *));
    int **result = (int **)malloc(N * sizeof(int *));

    for (int i = 0; i < N; i++) {
        matrixA[i] = (int *)malloc(N * sizeof(int));
        matrixB[i] = (int *)malloc(N * sizeof(int));
        result[i] = (int *)malloc(N * sizeof(int));
    }
    
    // Initialize matrices A and B with values (for example, 1 for A and 2 for B)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixA[i][j] = 1;
            matrixB[i][j] = 2;
	    result[i][j] = 0;
        }
    }
    start = clock();
    matrix_multiply(matrixA, matrixB, result);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    printf("Last Element of resultant matrix: %d", result[N-1][N-1]);

    // Printing matrices is omitted due to their large size

    // Free dynamically allocated memory
    for (int i = 0; i < N; i++) {
        free(matrixA[i]);
        free(matrixB[i]);
        free(result[i]);
    }
    free(matrixA);
    free(matrixB);
    free(result);

    return 0;
}

