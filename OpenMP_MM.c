#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000 // Matrix dimension
#define BLOCK_SIZE 64 // Block size for matrix multiplication

void matrix_multiply_parallel(int **A, int **B, int **C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Multiply blocks
                for (int ii = i; ii < i + BLOCK_SIZE; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++) {
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                        // Print intermediate results for a small section
                        if (ii < 5 && jj < 5) {
                            // printf("C[%d][%d] = %d\n", ii, jj, C[ii][jj]);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Allocate and initialize matrices
    int **matrixA = (int **)malloc(N * sizeof(int *));
    int **matrixB = (int **)malloc(N * sizeof(int *));
    int **result = (int **)malloc(N * sizeof(int *));
    for (int i = 0; i < N; i++) {
        matrixA[i] = (int *)malloc(N * sizeof(int));
        matrixB[i] = (int *)malloc(N * sizeof(int));
        result[i] = (int *)calloc(N, sizeof(int));
        for (int j = 0; j < N; j++) {
            matrixA[i][j] = 1;
            matrixB[i][j] = 2;
        }
    }
    printf("%s\n", "CCDS");

    matrix_multiply_parallel(matrixA, matrixB, result);

    printf("Resulting Matrix (Partial):\n");
    fflush(stdout);
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         printf("%d ", result[i][j]);
    //     }
    //     printf("\n");
    // }

    // Free memory
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
