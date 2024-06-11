#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

void vector_matrix_mult(int rows, int cols, int transpose, double alpha, double* A, double* x, double beta, double* y);
void vector_transposed_vector_mult(int rows, int cols, double alpha, double* x, double* y, double* A);

double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);

#endif
