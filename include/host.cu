#include "headers.h"
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

    // Allocate the device input vector B
    // float *d_B = NULL;
    // err = cudaMalloc((void **)&d_B, size);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // Allocate the device output vector C
    // float *d_C = NULL;
    // err = cudaMalloc((void **)&d_C, size);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Launch the Vector Add CUDA Kernel

    dim3 X1(1,1,1);
    dim3 Y1(32,32,1);
    printf("Cuda kernel launched\n");
    matrix<<<X1,Y1>>>(d_A,numElements, dimension);


    // int threadsPerBlock = 1024;
    // int blocksPerGrid =1;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    // matrix<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, numElements, dimension);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i=0; i<numElements; i++){
        printf("%f  ",h_A[i]);
        if ((i+1)%dimension == 0){
            printf("\n");
        }
    }


    // Verify that the result vector is correct
    // for (int i = 0; i < numElements; ++i)
    // {
    //     if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    //     {
    //         fprintf(stderr, "Result verification failed at element %d!\n", i);
    //         exit(EXIT_FAILURE);
    //     }
    // }

    // printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // err = cudaFree(d_B);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaFree(d_C);

    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Free host memory
    free(h_A);
    // free(h_B);
    // free(h_C);

    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
    return 0;
}
