#include "header.h"
/**
 * Host main routine
 */

int main(void)
{
    // cudaError_t err = cudaSuccess;
    printf("Convolution operation\n");
    int input_size = 5;
    int numElements = input_size*input_size*3;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1; //rand()/(float)RAND_MAX;
    }
    

    Conv2d h(3,3,3);
    float* out = h.forward(h_A, 5,5);

    cout<<"THe output is \n";
    for (int i = 0; i<5*5*3;i++)
        cout<<out[i]<<" ";
    
}