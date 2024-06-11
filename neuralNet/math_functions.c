
#include <math.h>

#include "math_functions.h"

void vector_matrix_mult(int rows, int cols, int transpose, double alpha, double* A, double* x, double beta, double* y) {
    for (int i = 0; i < rows; i++) {
        y[i] *= beta;
        for (int j = 0; j < cols; j++) {
            if (transpose) {
                y[i] += alpha * A[j * rows + i] * x[j];
            } else {
                y[i] += alpha * A[i * cols + j] * x[j];
            }
        }
    }
}

void vector_transposed_vector_mult(int rows, int cols, double alpha, double* x, double* y, double* A) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] += alpha * x[i] * y[j];
        }
    }
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}