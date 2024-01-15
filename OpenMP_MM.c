#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000
#define THRESHOLD 64 // Threshold for switching to naive matrix multiplication


void matrixMultiplication(double **A, double **B, double **C, int size) {
    int i, j, k;
    #pragma omp parallel for private(i, j, k)
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
void matrixAdd(double **A, double **B, double **C, int size) {
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrixSubtract(double **A, double **B, double **C, int size) {
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

void strassenMultiply(double **A, double **B, double **C, int size) {
    if (size <= THRESHOLD) {
        // Use conventional matrix multiplication for small matrices
        matrixMultiplication(A, B, C, size);
        return;
    }

    int new_size = size / 2;

    // Allocate memory for submatrices
    double **A11, **A12, **A21, **A22;
    double **B11, **B12, **B21, **B22;
    double **C11, **C12, **C21, **C22;
    double **M1, **M2, **M3, **M4, **M5, **M6, **M7;
    int i;

    A11 = (double **)malloc(new_size * sizeof(double *));
    A12 = (double **)malloc(new_size * sizeof(double *));
    A21 = (double **)malloc(new_size * sizeof(double *));
    A22 = (double **)malloc(new_size * sizeof(double *));
    B11 = (double **)malloc(new_size * sizeof(double *));
    B12 = (double **)malloc(new_size * sizeof(double *));
    B21 = (double **)malloc(new_size * sizeof(double *));
    B22 = (double **)malloc(new_size * sizeof(double *));
    C11 = (double **)malloc(new_size * sizeof(double *));
    C12 = (double **)malloc(new_size * sizeof(double *));
    C21 = (double **)malloc(new_size * sizeof(double *));
    C22 = (double **)malloc(new_size * sizeof(double *));
    M1 = (double **)malloc(new_size * sizeof(double *));
    M2 = (double **)malloc(new_size * sizeof(double *));
    M3 = (double **)malloc(new_size * sizeof(double *));
    M4 = (double **)malloc(new_size * sizeof(double *));
    M5 = (double **)malloc(new_size * sizeof(double *));
    M6 = (double **)malloc(new_size * sizeof(double *));
    M7 = (double **)malloc(new_size * sizeof(double *));

    // Initialize submatrices
    for (i = 0; i < new_size; i++) {
        A11[i] = &A[i][0];
        A12[i] = &A[i][new_size];
        A21[i] = &A[i + new_size][0];
        A22[i] = &A[i + new_size][new_size];

        B11[i] = &B[i][0];
        B12[i] = &B[i][new_size];
        B21[i] = &B[i + new_size][0];
        B22[i] = &B[i + new_size][new_size];

        C11[i] = &C[i][0];
        C12[i] = &C[i][new_size];
        C21[i] = &C[i + new_size][0];
        C22[i] = &C[i + new_size][new_size];

        M1[i] = (double *)malloc(new_size * sizeof(double));
        M2[i] = (double *)malloc(new_size * sizeof(double));
        M3[i] = (double *)malloc(new_size * sizeof(double));
        M4[i] = (double *)malloc(new_size * sizeof(double));
        M5[i] = (double *)malloc(new_size * sizeof(double));
        M6[i] = (double *)malloc(new_size * sizeof(double));
        M7[i] = (double *)malloc(new_size * sizeof(double));
    }

    double **temp1 = (double **)malloc(new_size * sizeof(double *));
    double **temp2 = (double **)malloc(new_size * sizeof(double *));
    for (i = 0; i < new_size; i++) {
        temp1[i] = (double *)malloc(new_size * sizeof(double));
        temp2[i] = (double *)malloc(new_size * sizeof(double));
    }

    // Calculating M1 to M7 matrices
    #pragma omp parallel sections
    {
        #pragma omp section
        {
		
        matrixAdd(A11, A22, temp1, new_size);
        matrixAdd(B11, B22, temp2, new_size);
        strassenMultiply(temp1, temp2, M1, new_size);}
        #pragma omp section
        {
		
        matrixAdd(A21, A22, temp1, new_size);
        strassenMultiply(temp1, B11, M2, new_size);}

        #pragma omp section
        {
        matrixSubtract(B12, B22, temp1, new_size);
        strassenMultiply(A11, temp1, M3, new_size);}

        #pragma omp section
        {
        matrixSubtract(B21, B11, temp1, new_size);
        strassenMultiply(A22, temp1, M4, new_size);}

        #pragma omp section
        {
        matrixAdd(A11, A12, temp1, new_size);
        strassenMultiply(temp1, B22, M5, new_size);}

        #pragma omp section
        {
		
        matrixSubtract(A21, A11, temp1, new_size);
        matrixAdd(B11, B12, temp2, new_size);
        strassenMultiply(temp1, temp2, M6, new_size);}

        #pragma omp section
        {
        matrixSubtract(A12, A22, temp1, new_size);
        matrixAdd(B21, B22, temp2, new_size);
        strassenMultiply(temp1, temp2, M7, new_size);}
    }

    // Calculating C matrices
    #pragma omp parallel sections
    {
        #pragma omp section
        {
		
        matrixAdd(M1, M4, temp1, new_size);
        matrixSubtract(temp1, M5, temp2, new_size);
        matrixAdd(temp2, M7, C11, new_size);}

        #pragma omp section
        matrixAdd(M3, M5, C12, new_size);

        #pragma omp section
        matrixAdd(M2, M4, C21, new_size);

        #pragma omp section
        {
		
        matrixSubtract(M1, M2, temp1, new_size);
        matrixAdd(temp1, M3, temp2, new_size);
        matrixAdd(temp2, M6, C22, new_size);}
    }

    // Free allocated memory for submatrices and temporary matrices
    for (i = 0; i < new_size; i++) {
        free(M1[i]);
        free(M2[i]);
        free(M3[i]);
        free(M4[i]);
        free(M5[i]);
        free(M6[i]);
        free(M7[i]);

        free(temp1[i]);
        free(temp2[i]);
    }
    free(M1);
    free(M2);
    free(M3);
    free(M4);
    free(M5);
    free(M6);
    free(M7);

    free(temp1);
    free(temp2);

    free(A11);
    free(A12);
    free(A21);
    free(A22);
    free(B11);
    free(B12);
    free(B21);
    free(B22);
    free(C11);
    free(C12);
    free(C21);
    free(C22);
}

int main() {
    double **A, **B, **C;
    int i, j;

    // Allocate memory for matrices in a contiguous block
    double *A_data = (double *)malloc(N * N * sizeof(double));
    double *B_data = (double *)malloc(N * N * sizeof(double));
    double *C_data = (double *)malloc(N * N * sizeof(double));

    A = (double **)malloc(N * sizeof(double *));
    B = (double **)malloc(N * sizeof(double *));
    C = (double **)malloc(N * sizeof(double *));
    for (i = 0; i < N; i++) {
        A[i] = &A_data[i * N];
        B[i] = &B_data[i * N];
        C[i] = &C_data[i * N];
    }

    // Initialize matrices A and B
    #pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0; // Initialize with 1 for example
            B[i][j] = 2.0; // Initialize with 2 for example
        }
    }

    // Perform matrix multiplication using Strassen's algorithm and measure time
    double start_time = omp_get_wtime();
    strassenMultiply(A, B, C, N);
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    printf("Matrix multiplication executed using Strassen's algorithm in %f seconds.\n", execution_time);
    printf("Last element of resultant matrix: %f", C[N-1][N-1]);

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    free(A_data);
    free(B_data);
    free(C_data);

    return 0;
}

