#include "header.h"
/**
 * Host main routine
 */

int main(void)
{
    // cudaError_t err = cudaSuccess;
    printf("Convolution operation\n");
    int input_size = 5;
    int numElements = input_size*input_size*2;
    size_t size = numElements * sizeof(float);

    float *h_in = (float *)malloc(size);
    float *d_out =(float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_in[i] = 1; //rand()/(float)RAND_MAX;
        d_out[i] = 1;
    }
    
    // d_out[4] = 1;

    Conv2d h(2,2,3);
    // float* out = h.forward(h_A, 5,5);
    float* out = h.backward(d_out, h_in, input_size, input_size);

    cout<<"\n THe output is: ";
    for (int i = 0; i<18*2;i++)
        cout<<out[i]<<" ";
    cout<<endl;
    
}