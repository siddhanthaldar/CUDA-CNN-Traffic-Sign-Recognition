#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/kernel.cu"
#include <stdio.h>

using namespace std;

// class Conv2d
// {
// public:

// 	float ****weight; //channels_out, channels_in, kernel_size, kernel_size
// 	float bias;

// 	Conv2d(int channel_in, int channel_out, int kernel_size);
// 	void forward();
// 	void backward();
// };

class FC
{
public:

	float *weight;
	float *bias;
    float *d_in, *out;

	FC(int in_features, int out_features);
	void forward(float *in, int in_size, int out_size);
	void backward();
};

class max_pool
{
public:
	int *mask; //to remember the location
	float *out, *d_in;
    max_pool(int h, int w);
	void forward(float *in, int h, int w, int channel);
	void backward(float *d_out, int h, int w);
};

// class ReLU
// {
// public:
// 	void forward();
// 	void backward();
// };

// class softmax
// {
// public:
// 	void forward();
// 	void backward();
// };

// class dropout
// {
// public:
// 	bool** mask;
// 	void forward();
// 	void backward();
// };

// Conv2d::Conv2d(int channel_in, int channel_out, int kernel_size)
// {
// 	weight = new float***[channel_out];
// 	for(int i = 0; i < channel_out; i++)
// 	{
// 		weight[i] = new float**[channel_in];
// 		for(int j = 0; j < kernel_size; j++)
// 		{
// 			weight[i][j] = new float*[kernel_size];
// 			for(int k = 0; k < kernel_size; k++)
// 			{
// 				weight[i][j][k] = new float[kernel_size];
// 				for(int l = 0; l < kernel_size; l++)
// 					weight[i][j][k][l] = 0.0; //replace with random
// 			}
// 		}
// 	}

// 	bias = 0.0;//replace with random
// }

FC::FC(int in_size,int out_size)
{
    weight = (float*)malloc(out_size*in_size*sizeof(float));
    for(int i=0;i<out_size*in_size;i++)
        weight[i] = rand()/(float)RAND_MAX;

    bias = (float*)malloc(out_size*sizeof(float));
    for(int i=0;i<out_size;i++)
        bias[i] = rand()/(float)RAND_MAX;

    out = (float*)malloc(out_size*sizeof(float));
    d_in = (float*)malloc(in_size*sizeof(float));
}

void FC::forward(float *in, int in_size, int out_size)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_in = NULL;   // g stands for GPU
    size = in_size*sizeof(float);
    err = cudaMalloc((void **)&g_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_w = NULL;   
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_w, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_b = NULL;   
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = in_size*sizeof(float);
    err = cudaMemcpy(g_in, in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(float);
    err = cudaMemcpy(g_w, weight, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_b, bias, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    dim3 grid(1,1,1);
    dim3 block(1,out_size,1);  
    FC_fp<<<grid, block>>>(g_in,g_out,g_w,g_b,out_size,in_size,1);  // wx = (mxn)*(nxp)  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(int);
    err = cudaMemcpy(out, g_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(g_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_w);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

max_pool::max_pool(int h, int w)
{
	d_in = (float *)malloc(h*w * sizeof(float));
	out = (float *)malloc(h/2*w/2 * sizeof(float));
	mask = (int *)malloc(h/2*w/2 * sizeof(int));
}

void max_pool::forward(float *in, int h, int w, int channel)  // h and w are dim of input
{
	cudaError_t err = cudaSuccess;
	size_t size;

	float *g_in = NULL;   // g stands for GPU
	size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float *g_out = NULL;   // g stands for GPU
	size = h/2*w/2*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int *g_mask = NULL;   // g stands for GPU
	size = h/2*w/2*sizeof(int);
    err = cudaMalloc((void **)&g_mask, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*sizeof(float);
    err = cudaMemcpy(g_in, in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	
	// Launch the Vector Add CUDA Kernel
    dim3 grid(1,1,1);
    dim3 block(w/2,h/2,1);  
	maxpool_fp<<<grid, block>>>(g_in,g_out,g_mask,h,w);  
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h/2*w/2*sizeof(int);
    err = cudaMemcpy(mask, g_mask, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	size = h/2*w/2*sizeof(float);
    err = cudaMemcpy(out, g_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Free device global memory
    err = cudaFree(g_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_mask);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void max_pool::backward(float *d_out, int h, int w)  // h and w are dim of out
{
	cudaError_t err = cudaSuccess;
	size_t size;

	float *g_d_out = NULL;   // g stands for GPU
	size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float *g_d_in = NULL;   // g stands for GPU
	size = h*2*w*2*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	int *g_mask = NULL;   // g stands for GPU
	size = h*w*sizeof(int);
    err = cudaMalloc((void **)&g_mask, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    size = h*w*sizeof(float);
    err = cudaMemcpy(g_mask, mask, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

	// Launch the Vector Add CUDA Kernel
    dim3 grid(1,1,1);
    dim3 block(w,h,1);
	maxpool_bp<<<grid,block>>>(g_d_in,g_d_out,g_mask,h,w);  
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	size = h*2*w*2*sizeof(float);
    err = cudaMemcpy(d_in, g_d_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   // Free device global memory
    err = cudaFree(g_d_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_d_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_mask);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}