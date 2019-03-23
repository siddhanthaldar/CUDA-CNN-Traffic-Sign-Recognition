#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/kernel.cu"
#include <stdio.h>
#include <math.h>

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

class ReLU
{
public:
    float *out,*d_in;
    ReLU(int h, int w, int channel);
	void forward(float *in, int h, int w, int channel);
	void backward(float* d_out, int h, int w, int channel);
};

class Sigmoid
{
public:
    float *out,*d_in;
    Sigmoid(int h, int w, int channel);
    void forward(float *in, int h, int w, int channel);
    void backward(float* d_out, int h, int w, int channel);
};

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


ReLU::ReLU(int h, int w, int channel)
{
    out = (float*)malloc(h*w*channel*sizeof(float));
    d_in = (float*)malloc(h*w*channel*sizeof(float));
}

void ReLU::forward(float *in, int h, int w, int channel)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_in = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_in, in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 grid(channel,1,1);
    dim3 block(w,h,1);  
    ReLU_fp<<<grid, block>>>(g_out,g_in,h,w,channel);   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ReLU_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(int);
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
}

void ReLU::backward(float* d_out, int h, int w, int channel)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_d_in = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_d_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_out, out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 grid(channel,1,1);
    dim3 block(w,h,1);  
    ReLU_bp<<<grid, block>>>(g_d_in,g_d_out,g_out,h,w,channel);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ReLU_bp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(int);
    err = cudaMemcpy(d_in, g_d_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_in from device to host (error code %s)!\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to free device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

// ******************************************************************************************************************

Sigmoid::Sigmoid(int h, int w, int channel)
{
    out = (float*)malloc(h*w*channel*sizeof(float));
    d_in = (float*)malloc(h*w*channel*sizeof(float));
}

void Sigmoid::forward(float *in, int h, int w, int channel)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_in = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_in, in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 grid(channel,1,1);
    dim3 block(w,h,1);  
    Sigmoid_fp<<<grid, block>>>(g_out,g_in,h,w,channel);   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ReLU_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(int);
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
}

void Sigmoid::backward(float* d_out, int h, int w, int channel)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_d_in = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_d_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_out, out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 grid(channel,1,1);
    dim3 block(w,h,1);  
    Sigmoid_bp<<<grid, block>>>(g_d_in,g_d_out,g_out,h,w,channel);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ReLU_bp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(int);
    err = cudaMemcpy(d_in, g_d_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_in from device to host (error code %s)!\n", cudaGetErrorString(err));
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
        fprintf(stderr, "Failed to free device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}
  //*************************************************************************************************************
