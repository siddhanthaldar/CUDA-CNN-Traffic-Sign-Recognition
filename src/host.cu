#include "../include/header.h"
/**
 * Host main routine
 */

int main(void)
{
    // cudaError_t err = cudaSuccess;
    printf("Softmax and loss compute\n");
    int input_size = 5;
    int numElements = input_size;
    size_t size = numElements * sizeof(float);
    int label = 2;//starts from 0
    float *h_in = (float *)malloc(size);
    float *d_out =(float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_in[i] = i; //rand()/(float)RAND_MAX;
        d_out[i] = 1;
    }
    h_in[0] = 100;
    // d_out[4] = 1;

    softmax_and_loss h;
    // float* out = h.forward(h_A, 5,5);
    float* out = h.forward(h_in, label, input_size);
    float* del_out = h.backward(out, label, input_size);

    cout<<"\n THe output is: ";
    for (int i = 0; i<5;i++)
    {
        cout<<out[i]<<" ";
    }
    cout<<endl;
    cout<<"Loss : "<<h.loss<<endl;
}