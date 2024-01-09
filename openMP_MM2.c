#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

void matrixMultiplication(double **A, double **B, double **C) {
    int i, j, k;

    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    double **A, **B, **C;
    int i, j;

    // Allocate memory for matrices
    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    for (i = 0; i < N; i++) {
        A[i] = (double *)malloc(N * sizeof(double));
        B[i] = (double *)malloc(N * sizeof(double));
        C[i] = (double *)malloc(N * sizeof(double));
    }

    // Initialize matrices A and B
    // (You can modify this section as needed)
    #pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0; // Initialize with 1 for example
            B[i][j] = 2.0; // Initialize with 2 for example
        }
    }

    // Perform matrix multiplication
    matrixMultiplication(A, B, C);

    // Display the result or perform further operations with matrix C

    // Free allocated memory
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

