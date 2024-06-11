#ifndef MNIST_PARSER_H
#define MNIST_PARSER_H

double** load_mnist_images(char *filename, int* image_count, int* image_size);
double** load_mnist_labels(char *filename, int* label_count);

#endif