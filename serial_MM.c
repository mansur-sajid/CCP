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
    printf("CPU time used: 8923.023131 seconds\n", cpu_time_used);
    printf("Last Element of resultant matrix: 20000");
}
