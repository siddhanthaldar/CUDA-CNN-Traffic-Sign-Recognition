#include "headers.h"
#include <cuda_runtime.h>
/**
 * Host main routine
 */

int main(void)
{
    cudaError_t err = cudaSuccess;
    printf("Convolution operation\n");
    int input_size = 32;
    int numElements = input_size*input_size*3;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);

    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        // h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<numElements; i++){
        printf("%f   ", h_A[i]); 
        if ((i+1)%dimension == 0){
            printf("\n");
        }
    }
    Conv2d h(3,10,3);
    h.forward();

}