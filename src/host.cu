#include "header.h"
/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Doing operations on matrix of size %d x %d with mask of size %d x %d\n", width, width, mask_width, mask_width);

    size_t mat_size = width*width*sizeof(float);
    size_t mask_size = mask_width*mask_width*sizeof(float);

    float *h_A = (float*)malloc(mat_size);

    // Verify that allocations succeeded
    if (h_A == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < width; ++i)
    {
        for(int j=0; j<width; j++)
            h_A[i*width+j] = rand()/(float)RAND_MAX;
    }

    printf("MATRIX:\n");
    for (int i = 0; i < width; ++i)
    {
        for(int j=0; j<width; j++)
            printf("%f ",h_A[i*width+j]);
        printf("\n");
    }

	float *h_B = (float*)malloc(mask_size);

    // Verify that allocations succeeded
    if (h_B == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < mask_width; ++i)
    {
        for(int j=0; j<mask_width; j++)
            h_B[i*mask_width+j] = rand()/(float)RAND_MAX;
    }

    printf("MASK:\n");
    for (int i = 0; i <mask_width; ++i)
    {
        for(int j=0; j<mask_width; j++)
            printf("%f ",h_B[i*mask_width+j]);
        printf("\n");
    }

    float *h_C = (float*)malloc(mat_size);

    // Verify that allocations succeeded
    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
    //*************************************************************************************************


    // Allocate the device input vector A
    // Every function with a "cuda" prefix has a error code returned which can be used to track error
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, mat_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Allocate the device input vector A
    // Every function with a "cuda" prefix has a error code returned which can be used to track error
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, mask_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, mat_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    ///////////////////////////////////// Operation 1 //////////////////////////////////////////////

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_A, mat_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, mask_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_C, h_C, mat_size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Process Kernel 1 
    dim3 grid(2,2,1);
    dim3 block(4,4,1);
    conv<<<grid,block>>>(d_A,d_B,d_C, mask_width, width, TILE_SIZE);



    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Operation completed");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    err = cudaMemcpy(h_C, d_C, width*width*sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("\n Convolved Matrix\n");
    for (int i = 0; i < width; ++i)
    {
        for(int j=0; j<width; j++)
            printf("%f ",h_C[i*width+j]);
        printf("\n");
    }  

    printf("%f\n", h_A[0*width+1+3]*h_B[1*mask_width+0] + h_A[0*width+2+3]*h_B[1*mask_width+1] + h_A[1*width+3+3]*h_B[1*mask_width+2] + h_A[1*width+1+3]*h_B[2*mask_width+0]+ h_A[1*width+2+3]*h_B[2*mask_width+1]+ h_A[1*width+3+3]*h_B[2*mask_width+2]);
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);

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

