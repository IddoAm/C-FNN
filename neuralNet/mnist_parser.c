#include <stdio.h>
#include <stdlib.h>
#include "mnist_parser.h"

#define IMAGES_MAGIC_NUMBER 2051
#define LABELS_MAGIC_NUMBER 2049 

#define FORMATTED_LABEL_SIZE 10

/*
    Function to read an int of a binary filestream
    input: binary filesteram
    output: the int value
*/
int read_int(FILE *fp) {
    int result;
    unsigned char bytes[4];
    fread(bytes, sizeof(bytes), 1, fp);
    result = (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    return result;
}


/*
    Function to extract data from idx3-ubyte format
    input: filename (expext idx3-ubyte file), pointer to image count and image size to be set
    output: normilised data as an array of 784 of double values between 0 and 1.
*/
double** load_mnist_images(char *filename, int* image_count, int* image_size){

    // idx3-ubyte format:
    // 'https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html'

    double** images = NULL;
    unsigned char *buffer = NULL;
    int rows = 0, cols = 0;
    int i = 0;

    FILE* fp = fopen(filename, "rb");

    // Read Headers
    int magic_number = read_int(fp);
    if(magic_number != IMAGES_MAGIC_NUMBER){
        printf("Wrong File! %s, expected magic number: %d but got %d\n", filename, IMAGES_MAGIC_NUMBER, magic_number);
        exit(1);
    }

    *image_count = read_int(fp); // x - image count
    rows = read_int(fp); // y - rows per image
    cols = read_int(fp); // z - cols per image
    *image_size = rows * cols;

    // Read Data
    images = malloc(*image_count * sizeof(double*));
    buffer = malloc(*image_size * sizeof(unsigned char));

    for (i = 0; i < *image_count; i++) {
        images[i] = malloc(*image_size * sizeof(double));
        fread(buffer, *image_size, 1, fp);
        for (int j = 0; j < *image_size; j++) {
            images[i][j] = buffer[j] / 255.0;
        }
    }

    free(buffer);
    fclose(fp);

    return images;
}

/*
    Function to extract data from idx1-ubyte file
    input: filepath to idx1-ubyte file and a pointer to label count to be changed
    output: unsigned char array of length label_count containing the label of each image.
*/
double** load_mnist_labels(char *filename, int* label_count){

    int i = 0;

    double** formatted_labels = NULL;
    unsigned char* labels = NULL;
    FILE* fp = fopen(filename, "rb");

    // Read headers
    int magic_number = read_int(fp);
    if(magic_number != LABELS_MAGIC_NUMBER){
        printf("Wrong File! %s, expected magic number: %d but got %d\n", filename, LABELS_MAGIC_NUMBER, magic_number);
        exit(1);
    }

    *label_count = read_int(fp);

    // Read data
    labels = (unsigned char*)malloc(*label_count * sizeof(unsigned char));
    fread(labels, *label_count, sizeof(unsigned char), fp);

    // Format the labels, to end up with something like this:
    // 0,0,0,0,1,0,0,0,0,0
    // meaning number 4.

    formatted_labels = (double**)malloc(*label_count * sizeof(double*));
    for(i = 0; i < *label_count; i++){
        // Calloc is like malloc but all variables are set to 0.
        formatted_labels[i] = (double*)calloc(FORMATTED_LABEL_SIZE, sizeof(double));
        formatted_labels[i][labels[i]] = 1.0;
    }

    free(labels);
    fclose(fp);

    return formatted_labels;
}