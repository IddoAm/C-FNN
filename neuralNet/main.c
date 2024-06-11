#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <cblas.h>

#include "mnist_parser.h"


typedef struct Layer
{
    // Scalars
    unsigned int count;
    unsigned int prev_count;
    // Vectors - size of 'count'
    double* activations;
    double* pre_activations;
    double* biases;
    double* error;
    // Matrecies
    double* weights;            // this layers "count" * prev layer "count"
    // Methods
    double (*activationFunction)(double);
    double (*activationFunctionDerivative)(double);
} Layer;

// Helpers
double drange(double min, double max);
double sigmoid(double x);
double sigmoid_derivative(double x);
double relu(double x);
double he_initialization(int prev_layer_count);
double relu_derivative(double x);
void apply_activation(const double* x, double* result, int len, double (*activationFunction)(double));
int maxidx(double* vec, int count);

void print_figure(double* pixels, int label);



// Layer Functions
Layer* init_layer(unsigned int count, unsigned int prev_count, double (*activationFunction)(double), double (*activationFunctionDerivative)(double));
void free_layer(Layer* layer);

void layer_forward_step(Layer* layer, double* input);

// Network Functions
void forward_prop(Layer** layers, double* input, int layers_count);
void backward_prop(Layer** layers, int layers_count, double* target, double* input, double learning_rate);
void train_network(Layer** layers, int layers_count, double** images, double** labels, int data_count, double learning_rate);
double validate_network(Layer** layers, int layers_count, double** images, double** labels, int test_count);



/*
    MAIN
*/
int main(int argc, char* argv[]) {

    if(argc < 4){
        printf("Not enough args\n");
        return 1;
    }

    srand(time(NULL));

    const unsigned int INPUT_COUNT = 784;
    const unsigned int OUTPUT_COUNT = 10;

    int i = 0;

    // Load data
    int train_image_count = 0, test_image_count = 0;
    int image_size = 0;
    

    int train_label_count = 0, test_label_count = 0;

    double** mnist_train_images = load_mnist_images("mnist/train-images.idx3-ubyte", &train_image_count, &image_size);
    double** mnist_train_labels = load_mnist_labels("mnist/train-labels.idx1-ubyte", &train_label_count);

    double** mnist_validate_images = load_mnist_images("mnist/t10k-images.idx3-ubyte", &test_image_count, &image_size);
    double** mnist_validate_labels = load_mnist_labels("mnist/t10k-labels.idx1-ubyte", &test_label_count);

    printf("training:\nImage count: %d, Label count: %d\n", train_image_count, train_label_count);
    printf("testing:\nImage count: %d, Label count: %d\n", test_image_count, test_label_count);

    // Set up network
    const double LEARNING_RATE = atof(argv[1]);
    const int LAYER_COUNT = 1 + argc - 2;
    // +1 for output layer, -2 the program name and learning rate

    int count = 0;
    int prev_count = image_size;
    
    Layer** layers = (Layer**)malloc(LAYER_COUNT * sizeof(Layer*));
    for(i = 0; i < LAYER_COUNT-1; i++){
        count = atoi(argv[i+2]);
        printf("Layer: %d, count: %d, prev count: %d\n", i, count, prev_count);
        layers[i] = init_layer(count, prev_count, relu, relu_derivative);
        prev_count = count;
    }
    layers[LAYER_COUNT-1] = init_layer(OUTPUT_COUNT, prev_count, sigmoid, sigmoid_derivative);

    // Training Loop

    printf("Starting Training!\n");

    double validate_acc = 0;

    for(i = 0; i < 15; i++){
        train_network(layers, LAYER_COUNT, mnist_train_images, mnist_train_labels, train_image_count, LEARNING_RATE);
        validate_acc = validate_network(layers, LAYER_COUNT, mnist_validate_images, mnist_validate_labels, test_image_count);
        printf("Epoch %d finished, acc: %f\n", i+1, validate_acc);
    }


    // Free Data

    for(i = 0; i < LAYER_COUNT; i++){
        free_layer(layers[i]);
    }
    free(layers);


    for(i = 0; i < train_image_count; i++){
        free(mnist_train_images[i]);
        free(mnist_train_labels[i]);
    }
    
    free(mnist_train_images);
    free(mnist_train_labels);

    for(i = 0; i < test_image_count; i++){
        free(mnist_validate_images[i]);
        free(mnist_validate_labels[i]);
    }
    
    free(mnist_validate_images);
    free(mnist_validate_labels);

    return 0;
}



double drange(double min, double max) {
    // Generate a random double between 0 and 1
    double rand_double = (double)rand() / RAND_MAX;

    // Scale and shift the double to fit the range [min, max]
    rand_double = rand_double * (max - min) + min;

    return rand_double;
}

// Suitable for relu.
double he_initialization(int prev_layer_count) {
    double stddev = sqrt(2.0 / prev_layer_count);
    return drange(-stddev, stddev);
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

void print_figure(double* pixels, int label){
    int i = 0, j = 0;

    printf("Label: %d\n", label);

    for(i = 0; i < 28; i++){
        for(j = 0; j < 28; j++){
            if(pixels[i * 28 + j] == 0){
                printf(" ");
            }else if(pixels[i * 28 + j] < 0.66){
                printf("*");
            }else{
                printf("#");
            }
        }
        printf("\n");
    }
}

void apply_activation(const double* x, double* result, int len, double (*activationFunction)(double)) {
    for (int i = 0; i < len; i++) {
        result[i] = activationFunction(x[i]);
    }
}

int maxidx(double* vec, int count){
    double max = vec[0];
    int max_idx = 0;
    int i = 0;

    for(i = 1; i < count; i++){
        if(vec[i] > max){
            max = vec[i];
            max_idx = i;
        }
    }

    return max_idx;
}


Layer* init_layer(unsigned int count, unsigned int prev_count, double (*activationFunction)(double), double (*activationFunctionDerivative)(double)){
    int i = 0, j = 0;

    Layer* layer = (Layer*)malloc(sizeof(Layer));

    layer->count = count;
    layer->prev_count = prev_count;
    
    // Allocate some memory
    layer->activations = (double*)calloc(count, sizeof(double));
    layer->pre_activations = (double*)calloc(count, sizeof(double));
    layer->biases = (double*)calloc(count, sizeof(double));
    layer->error = (double*)calloc(count,sizeof(double));
    layer->weights = (double*)malloc(count*prev_count*sizeof(double));
    
    // Initilise the vectors and matrix
    for(i = 0; i < count; i++){
        for(j = 0; j < prev_count; j++){
            layer->weights[i * prev_count + j] = he_initialization(prev_count);
        }
    }

    layer->activationFunction = activationFunction;
    layer->activationFunctionDerivative = activationFunctionDerivative;

    return layer;
}

void free_layer(Layer* layer){
    free(layer->activations);
    free(layer->pre_activations);
    free(layer->biases);
    free(layer->error);
    free(layer->weights);
    
    free(layer);
}

void layer_forward_step(Layer* layer, double* input) {
    // matrix vector multi
    cblas_dgemv(CblasRowMajor, CblasNoTrans, layer->count, layer->prev_count, 1.0, layer->weights, layer->prev_count, input, 1, 0.0, layer->pre_activations, 1);

    for (unsigned int i = 0; i < layer->count; i++) {
        layer->pre_activations[i] += layer->biases[i];
    }

    apply_activation(layer->pre_activations, layer->activations, layer->count, layer->activationFunction);
}

void forward_prop(Layer** layers, double* input, int layers_count){
    double* current_input = input;
    int i = 0;  
    
    for(i = 0; i < layers_count; i++){
        layer_forward_step(layers[i], current_input);
        current_input = layers[i]->activations;
    }
}

void backward_prop(Layer** layers, int layers_count, double* target, double* input, double learning_rate){
    int i = 0, j = 0;


    // Mean squared derivitve
    for(i = 0; i < layers[layers_count-1]->count; i++){
        layers[layers_count-1]->error[i] = 2*(layers[layers_count-1]->activations[i] - target[i]) * layers[layers_count-1]->activationFunctionDerivative(layers[layers_count-1]->pre_activations[i]);   
    }
    // Propgate error
    for(i = layers_count - 2; i > 0; i--){
        cblas_dgemv(CblasRowMajor, CblasTrans, layers[i]->count, layers[i+1]->count, 1.0, layers[i+1]->weights, layers[i+1]->count, layers[i+1]->error, 1, 0.0, layers[i]->error, 1);
        
        for(j = 0; j < layers[i]->count; j++){
            layers[i]->error[j] *= layers[i]->activationFunctionDerivative(layers[i]->pre_activations[j]);
        }     
    }

    // Update weights and biases
    for(i = 0; i < layers_count; i++){
        
        double* prev_layer_activations = i > 0 ? layers[i-1]->activations : input;
        // Outer product mult for weights update
        cblas_dger(CblasRowMajor, layers[i]->count, layers[i-1]->count, -learning_rate, layers[i]->error, 1, prev_layer_activations, 1, layers[i]->weights, layers[i]->prev_count);


        // -learning rate * error
        for(j = 0; j < layers[i]->count; j++){
            layers[i]->biases[j] -= learning_rate * layers[i]->error[j];
        }
    }
}

void train_network(Layer** layers, int layers_count, double** images, double** labels, int data_count, double learning_rate){
    int i = 0;
    clock_t start_time = clock();
    // Train network
    for(i = 0; i < data_count; i++){
        forward_prop(layers, images[i], layers_count);
        backward_prop(layers, layers_count, labels[i], images[i], learning_rate);
    }

    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Processed %d images in %f seconds. rate: %f images per sec\n", data_count, time_taken, data_count / time_taken);
}


double validate_network(Layer** layers, int layers_count, double** images, double** labels, int test_count){
    int i = 0;
    int correct = 0;

    for(i = 0; i < test_count; i++){
        forward_prop(layers, images[i], layers_count);

        if (maxidx(layers[layers_count-1]->activations, layers[layers_count-1]->count) == maxidx(labels[i], layers[layers_count-1]->count)) {
            correct++;
        }
    }

    return (double)correct / (double)test_count;
}