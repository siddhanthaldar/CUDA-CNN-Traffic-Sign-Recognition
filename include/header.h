#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/kernel.cu"
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#include <stdio.h>
#include <math.h>

using namespace std;
__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width);
__global__ void conv_bp(float* d_del_weight, float* d_input, float* d_del_out, int channel_in,int channel_out,int kernel_size,int input_height,int input_width);
__global__ void rotate(float* d_weight_t, float* d_weight, int channel_in, int channel_out, int kernel_size);
__global__ void conv_bp_x(float* d_del_input, float* d_del_out, float* d_weight_t, int channel_in,int channel_out,int kernel_size,int input_height,int input_width);
__global__ void conv_step(float* d_weight, float* d_del_weight, float* d_del_vw, int channel_in, int channel_out, int kernel_size, float l_rate, float beeta, bool is_first);


class Conv2d
{
public:
	bool is_first; //For Momentum initialization
	float *weight; //channels_out, channels_in, kernel_size, kernel_size
	float bias;
	float* del_weight;
	float* del_vw; //For momentum weight
	int channel_in, channel_out, kernel_size;

	Conv2d(int channel_in, int channel_out, int kernel_size);
	float* forward(float* image, int img_width, int img_height);
	float* backward(float* del_out, float* input, int input_height, int input_width);
	void step(float l_rate, float beeta=0.9);
};

Conv2d::Conv2d(int channel_in, int channel_out, int kernel_size)
{
	this->channel_in=channel_in;
	this->channel_out = channel_out;
	this->kernel_size = kernel_size;
	is_first = 1;
	// cout<<"\n the weight is: ";
	weight = new float[channel_out*kernel_size*kernel_size*channel_in]();//Initialize the weights
	del_weight = new float[channel_out*kernel_size*kernel_size*channel_in];//Initialize the weights
	del_vw = new float[channel_out*kernel_size*kernel_size*channel_in];//Initialize the weights

	for(int i = 0; i < channel_out*kernel_size*kernel_size*channel_in; i++)
	{
		weight[i] = rand()*1.0/RAND_MAX;
		// del_weight[i] = 0;
		// cout<<weight[i]<<" ";
	}

	bias = 0.0;//replace with random
}

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

ReLU::ReLU(int h, int w, int channel)
{
    out = (float*)malloc(h*w*channel*sizeof(float));
    d_in = (float*)malloc(h*w*channel*sizeof(float));
}

class softmax_and_loss
{
public:
    float loss;
    float* forward(float* logits, int label, int n_classes);
    float* backward(float* out, int label, int n_classes);
};

class Dropout
{
public:
    bool* mask;
    float drop_prob, *d_in;
    int h, w, channel;
    Dropout(float drop_prob, int h, int w, int channel);
    float* forward(float*in);
    float* backward(float* d_out);
};

Dropout::Dropout(float drop_prob, int h, int w, int channel)
{
    this->drop_prob = drop_prob;
    this->h = h;
    this->w = w;
    this->channel = channel;
    mask = new bool[h*w];
    d_in = new float[h*w*channel];
}

float* Dropout::forward(float *in)
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

    bool *g_mask = NULL; 
    size = h*w*sizeof(bool);
    err = cudaMalloc((void **)&g_mask, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
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
    dim3 grid(1,1,channel);
    dim3 block(w,h,1);  
    dropout_fp<<<grid, block>>>(g_in,g_mask,drop_prob,h,w,channel);   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch dropout_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy Memory from device to host
    size = h*w*sizeof(bool);
    err = cudaMemcpy(mask, g_mask, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*sizeof(float);
    err = cudaMemcpy(in, g_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(g_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_mask);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceReset();
    return(in);
}

float* Dropout::backward(float* d_out)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_d_out = NULL; 
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_d_in = NULL; 
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    bool *g_mask = NULL; 
    size = h*w*sizeof(bool);
    err = cudaMalloc((void **)&g_mask, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy Memory from host to device
    size = h*w*channel*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*sizeof(bool);
    err = cudaMemcpy(g_mask, mask, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the CUDA Kernel
    dim3 grid(1,1,channel);
    dim3 block(w,h,1);  
    dropout_bp<<<grid, block>>>(g_d_out,g_d_in,g_mask,h,w,channel);   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch dropout_bp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy memory from device to host
    size = h*w*channel*sizeof(int);
    err = cudaMemcpy(d_in, g_d_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_in from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory

    err = cudaFree(g_d_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_d_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_d_in(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_mask);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceReset();
    return(d_in);

}
