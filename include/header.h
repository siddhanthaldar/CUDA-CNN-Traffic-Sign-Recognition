#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../src/kernel.cu"
#include <stdio.h>
#include <math.h>

#define ENHANCE_PARAM 45.986  //parameter which adjusts enhancement in log transformation(255/log(256) - scale b/w 0 - 255))
#define GAMMA 0.5  //parameter used for gamma correction

using namespace std;

__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width);
__global__ void conv_bp(float* d_del_weight, float* d_input, float* d_del_out, int channel_in,int channel_out,int kernel_size,int input_height,int input_width);
__global__ void rotate(float* d_weight_t, float* d_weight, int channel_in, int channel_out, int kernel_size);
__global__ void conv_bp_x(float* d_del_input, float* d_del_out, float* d_weight_t, int channel_in,int channel_out,int kernel_size,int input_height,int input_width);
__global__ void conv_step(float* d_weight, float* d_del_weight, float* d_del_vw, int channel_in, int channel_out, int kernel_size, float l_rate, float beeta, bool is_first);
/////******Add other function of Pool_Fc and Activation function******///////// 

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
        int sign = rand()%2;
        if(sign == 0)
            weight[i] *= -1;
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
    float *dw, *dw_old;
    float *db, *db_old;
    float *d_in, *out;
    int in_size, out_size,first;

    FC(int in_features, int out_features);
    float* forward(float *in);
    float* backward(float *in,float *d_out);
    void step(float lr, float beta);
};


class MaxPool
{
public:
    int *mask; //to remember the location
    float *out, *d_in;
    MaxPool(int h, int w,int channel);
    float* forward(float *in, int h, int w, int channel);
    float* backward(float *d_out, int h, int w,int channel);
};


class ReLU
{
public:
    float *out,*d_in;
    ReLU(int h, int w, int channel);
	float* forward(float *in, int h, int w, int channel);
	float* backward(float* d_out, int h, int w, int channel);
};

class Sigmoid
{
public:
    float *out,*d_in;
    Sigmoid(int h, int w, int channel);
    float* forward(float *in, int h, int w, int channel);
    float* backward(float* d_out, int h, int w, int channel);
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

////////////////////////////////// ReLU ///////////////////////////////////

ReLU::ReLU(int h, int w, int channel)
{
    out = (float*)malloc(h*w*channel*sizeof(float));
    d_in = (float*)malloc(h*w*channel*sizeof(float));
}

float* ReLU::forward(float *in, int h, int w, int channel)
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
    return out;
}

float* ReLU::backward(float* d_out, int h, int w, int channel)
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
    return d_in;
}

////////////////////////////////// Sigmoid /////////////////////////////////////// 

Sigmoid::Sigmoid(int h, int w, int channel)
{
    out = (float*)malloc(h*w*channel*sizeof(float));
    d_in = (float*)malloc(h*w*channel*sizeof(float));
}

float* Sigmoid::forward(float *in, int h, int w, int channel)
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
    return out;
}

float* Sigmoid::backward(float* d_out, int h, int w, int channel)
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
    return d_in;

}
  //*************************************************************************************************************
class softmax_cross_entropy_with_logits
{
public:
	float loss;
	float* forward(float* logits, int label, int n_classes);
	float* backward(float* out, int label, int n_classes);
};

float* softmax_cross_entropy_with_logits::forward(float* logits, int label, int n_classes)
{
	double sum = 0;
	float m = INT_MIN;
	for(int i = 0; i < n_classes; i++)
		m = max(m, logits[i]);
	for(int i=0; i< n_classes; i++){
		logits[i] -= m;
		sum += exp(logits[i]);
	}
	float* out = new float[n_classes];
	for(int i=0; i<n_classes; i++){
		out[i] = exp(logits[i])/sum;
	}
	loss = 0.0;
	for(int i = 0; i<n_classes; i++){
		if(i!= label) loss -= log(1-out[i] + 1e-7);
		else loss -= log(out[i] + 1e-7); 
	}
	return out;
}

float* softmax_cross_entropy_with_logits::backward(float* out, int label, int n_classes){
	float* del_out = new float[n_classes];
	for(int i=0; i<n_classes; i++){
		if(i == label) del_out[i] = out[i]-1;
		else del_out[i] = out[i];
	}
	return(del_out);
}
//************************************//
//Forward propagation for convolution//
//************************************//
float* Conv2d::forward(float* image, int img_height, int img_width)
{
	cudaError_t err = cudaSuccess;
	float* h_out = new float[img_height*img_width*channel_out];

	dim3 grid(1,1,channel_out);
	dim3 block(img_height,img_width,1);

	size_t size_img = img_width*img_height*channel_in*sizeof(float);
	size_t size_filter = kernel_size*kernel_size*channel_out*channel_in*sizeof(float);
	size_t size_out = img_height*img_width*channel_out*sizeof(float);

	float *d_img = NULL;
	err = cudaMalloc((void **)&d_img, size_img);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_filter = NULL;
	err = cudaMalloc((void **)&d_filter, size_filter);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_filter (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float* d_out = NULL;
	err = cudaMalloc((void **)&d_out, size_out);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input feature map from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_img, image, size_img, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_filter, weight, size_filter, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_filter (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//************************************//
	//Calling Kernel Function//
	//************************************//
	conv_fp <<<grid,block>>>(d_out, d_img, d_filter, channel_in, channel_out, kernel_size, img_height, img_width);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy h_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_filter);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_filter (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_img);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaFree(d_out);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaDeviceReset();
	// printf("Partyyyy\n");
	return (h_out);
}

//************************************//
//Backward propagation for convolution//
//************************************//

float* Conv2d::backward(float* del_out, float* input, int input_height, int input_width)
{
	cudaError_t err = cudaSuccess;
	dim3 grid(kernel_size, kernel_size, channel_out);
	dim3 block(1, 1, channel_in);

	size_t size_del_input = input_width*input_height*channel_in*sizeof(float);
	size_t size_del_weight = kernel_size*kernel_size*channel_out*channel_in*sizeof(float);
	size_t size_del_out = input_height*input_width*channel_out*sizeof(float);

	float *d_del_input = NULL;
	err = cudaMalloc((void **)&d_del_input, size_del_input);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_del_input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_del_out = NULL;
	err = cudaMalloc((void **)&d_del_out, size_del_out);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_del_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_del_weight = NULL;
	err = cudaMalloc((void **)&d_del_weight, size_del_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_del_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float* d_input = NULL;
	err = cudaMalloc((void **)&d_input, size_del_input);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input feature map from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_input, input, size_del_input, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_out, del_out, size_del_out, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_del_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//************************************//
	//Calling Kernel Function//
	//************************************//
	conv_bp <<<grid, block>>>(d_del_weight, d_input, d_del_out, channel_in, channel_out, kernel_size, input_height, input_width);


	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(del_weight, d_del_weight, size_del_weight, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy del_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_weight_t = NULL;
	err = cudaMalloc((void **)&d_weight_t, size_del_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_weight_t (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_weight = NULL;
	err = cudaMalloc((void **)&d_weight, size_del_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy weight data from the host to the CUDA memory\n");
	err = cudaMemcpy(d_weight, weight, size_del_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dim3 grid_t(kernel_size,kernel_size,channel_out);
	dim3 block_t(1, 1, channel_in);


	//************************************//
	//Calling Kernel Function//
	//************************************//
	rotate <<<grid_t, block_t>>>(d_weight_t, d_weight, channel_in, channel_out, kernel_size);


	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Now calculating gradients of input
	float* del_input = new float[input_width*input_height*channel_in]();
	dim3 grid_x(1, 1, channel_in); //order is z, x, y
	dim3 block_x(input_width, input_height, 1);

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_input, del_input, size_del_input, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to initialize by zero (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//************************************//
	//Calling Kernel Function//
	//************************************//
	conv_bp_x <<<grid_x, block_x>>> (d_del_input, d_del_out, d_weight_t, channel_in, channel_out, kernel_size, input_height, input_width);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(del_input, d_del_input, size_del_input, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy del_input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaFree(d_del_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_del_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_weight_t);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_weight_t (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_del_input);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_del_input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_del_out);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_del_out (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_input);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_input (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaDeviceReset();
	// printf("Partyyyy\n");
	return (del_input);
}

void Conv2d::step(float l_rate, float beeta)
{
	cudaError_t err = cudaSuccess;
	size_t size_weight = kernel_size*kernel_size*channel_out*channel_in*sizeof(float);

	float *d_weight = NULL;
	err = cudaMalloc((void **)&d_weight, size_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_del_weight = NULL;
	err = cudaMalloc((void **)&d_del_weight, size_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	float *d_del_vw = NULL;
	err = cudaMalloc((void **)&d_del_vw, size_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate d_del_vw (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_weight, weight, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_weight, del_weight, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_vw, del_vw, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_del_vw (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dim3 grid(kernel_size,kernel_size,channel_out);
	dim3 block(1, 1, channel_in);

	conv_step<<<grid, block>>>(d_weight, d_del_weight, d_del_vw, channel_in, channel_out, kernel_size, l_rate, beeta, is_first);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(weight, d_weight, size_weight, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(del_vw, d_del_vw, size_weight, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy del_vw (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_del_weight);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free del_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_del_vw);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free d_del_vw (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	is_first = 0;
}

///////////////////////////// Fully Connected /////////////////////////////////////

FC::FC(int in_features, int out_features)
{
    in_size = in_features;
    out_size = out_features;
    weight = (float*)malloc(out_size*in_size*sizeof(float));
    dw = (float*)malloc(out_size*in_size*sizeof(float));
    dw_old = (float*)malloc(out_size*in_size*sizeof(float));

    first = 1;
    for(int i=0;i<out_size*in_size;i++)
    {
        weight[i] = rand()/(float)RAND_MAX;
        dw_old[i] = 0;
        int sign = rand()%2;
        if(sign == 0)
            weight[i] *= -1;

    }

    bias = (float*)malloc(out_size*sizeof(float));
    db = (float*)malloc(out_size*sizeof(float));
    db_old = (float*)malloc(out_size*sizeof(float));
    for(int i=0;i<out_size;i++)
    {
        bias[i] = rand()/(float)RAND_MAX;
        int sign = rand()%2;
        if(sign == 0)
            bias[i] *= -1;

    }


    out = (float*)malloc(out_size*sizeof(float));
    d_in = (float*)malloc(in_size*sizeof(float));
}

float* FC::forward(float *in)//, int in_size, int out_size)
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
    return out;

}

float* FC::backward(float *in, float *d_out)//,int in_size, int out_size)
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

    float *g_d_in = NULL;   // g stands for GPU
    size = in_size*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_d_out = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_w = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_w, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_dw = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_dw, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_dw (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_w_transpose = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_w_transpose, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_w_transpose (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_b = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_db = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_db, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_db (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(float);
    err = cudaMemcpy(g_w,weight, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_w from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_b,bias, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_b from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = in_size*sizeof(float);
    err = cudaMemcpy(g_in,in, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_in from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid(in_size,max(out_size, in_size),1);
    dim3 block(1,1,1);  
    FC_bp<<<grid, block>>>(g_d_out,g_d_in,g_w,g_w_transpose,g_dw,g_b,g_db,g_in,out_size,in_size,1);

    size = in_size*sizeof(int);
    err = cudaMemcpy(d_in, g_d_in, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_infrom device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(int);
    err = cudaMemcpy(dw, g_dw, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_dw from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(int);
    err = cudaMemcpy(db, g_db, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_db from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_d_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_d_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_w);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_w_transpose);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_w_transpose (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_dw);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_dw (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_db);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_db (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_in);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return d_in;
}

void FC::step(float lr, float beta)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_w = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_w, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_dw = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_dw, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_dw (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_dw_old = NULL;   // g stands for GPU
    size = out_size*in_size*sizeof(float);
    err = cudaMalloc((void **)&g_dw_old, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_dw_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_b = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_db = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_db, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_db (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_db_old = NULL;   // g stands for GPU
    size = out_size*sizeof(float);
    err = cudaMalloc((void **)&g_db_old, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_db_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy to device

    size = out_size*in_size*sizeof(float);
    err = cudaMemcpy(g_w, weight, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector weight from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(float);
    err = cudaMemcpy(g_dw, dw, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector dw from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(float);
    err = cudaMemcpy(g_dw_old, dw_old, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector dw_old from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_b, bias, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector bias from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_db, db, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector db from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(float);
    err = cudaMemcpy(g_db_old, db_old, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector db_old from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    // Update weights
    dim3 grid(in_size,out_size,1);
    dim3 block(1,1,1);
    FC_step_w<<<grid, block>>>(g_w,g_dw,g_dw_old,lr,beta,out_size,in_size,1,first);

    // Update bias
    dim3 grid1(out_size,1,1);
    dim3 block1(1,1,1);
    FC_step_b<<<grid1, block1>>>(g_b,g_db,g_db_old,lr,beta,out_size,in_size,1,first);


    size = out_size*in_size*sizeof(int);
    err = cudaMemcpy(weight, g_w, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_w from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(int);
    err = cudaMemcpy(dw, g_dw, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_dw from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*in_size*sizeof(int);
    err = cudaMemcpy(dw_old, g_dw_old, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_dw_old from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(int);
    err = cudaMemcpy(bias, g_b, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_b from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(int);
    err = cudaMemcpy(db, g_db, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_db from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = out_size*sizeof(int);
    err = cudaMemcpy(db_old, g_db_old, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_db_old from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_w);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_w (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_dw);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_dw (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_dw_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_dw_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_db);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_db (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_db_old);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_db_old (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    first=0;
}
//////////////////////////////// MaxPool //////////////////////////////

MaxPool::MaxPool(int h, int w,int channel)
{
    d_in = (float *)malloc(h*w *channel* sizeof(float));
    out = (float *)malloc(h/2*w/2 *channel* sizeof(float));
    mask = (int *)malloc(h/2*w/2 *channel* sizeof(int));
}

float* MaxPool::forward(float *in, int h, int w, int channel)  // h and w are dim of input
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
    size = h/2*w/2*channel*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *g_mask = NULL;   // g stands for GPU
    size = h/2*w/2*channel*sizeof(int);
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

    
    // Launch the Vector Add CUDA Kernel
    dim3 grid(1,1,channel);
    dim3 block(w/2,h/2,1);  
    maxpool_fp<<<grid, block>>>(g_in,g_out,g_mask,h,w,channel);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h/2*w/2*channel*sizeof(int);
    err = cudaMemcpy(mask, g_mask, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h/2*w/2*channel*sizeof(float);
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
    return out;
}

float* MaxPool::backward(float *d_out, int h, int w,int channel)  // h and w are dim of out
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_d_out = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_d_in = NULL;   // g stands for GPU
    size = h*2*w*2*channel*sizeof(float);
    err = cudaMalloc((void **)&g_d_in, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_d_in (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *g_mask = NULL;   // g stands for GPU
    size = h*w*channel*sizeof(int);
    err = cudaMalloc((void **)&g_mask, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_mask (error code %s)!\n", cudaGetErrorString(err));
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
    err = cudaMemcpy(g_mask, mask, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    // Launch the Vector Add CUDA Kernel
    dim3 grid(1,1,channel);
    dim3 block(w,h,1);
    maxpool_bp<<<grid,block>>>(g_d_in,g_d_out,g_mask,h,w,channel);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*2*w*2*channel*sizeof(float);
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
    return d_in;
}

//////////////////////Preprocessing class/////////////////////

//////////////////////Preprocessing class/////////////////////

class preprocessing
{
public:
    
    float *gray_img, *hist_img, *trans_img, *norm_img;
    int h,w,channels;
    
    preprocessing(int h, int w, int channels);
    
    void BGR2GRAY(float* img);
    void Histogram_Equalization(float *img);
    void GrayLevel_Neg_Transformation(float *img);
    void GrayLevel_Log_Transformation(float *img);
    void GrayLevel_Gam_Transformation(float *img);
    void Normalization(float *img);
};

preprocessing::preprocessing(int h, int w, int channels)
{
    this->h = h;
    this->w = w;
    this->channels = channels;

    gray_img = new float[h*w];
    hist_img = new float[h*w];
    trans_img = new float[h*w];
    norm_img = new float[h*w];
}


void preprocessing::BGR2GRAY(float* img)
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_img = NULL;   // g stands for GPU
    size = h*w*channels*sizeof(float);
    
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_gray_img = NULL;  
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_gray_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_gray_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy Memory to device
    size = h*w*channels*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Call kernel function
    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    bgr2gray<<<grid, block>>>(g_img,g_gray_img,h,w,channels);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch maxpool_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from Device to host
    size = h*w*sizeof(int);
    err = cudaMemcpy(gray_img, g_gray_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_gray_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_gray_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_gray_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void preprocessing::Histogram_Equalization(float *img) // Here img is gray scale image
{
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    float *g_out_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_out_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy Memory to device
    size = h*w*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Call kernel function
    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    histogram_equalization<<<grid, block>>>(g_img, h, w, 256);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch histogram_equalization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    calcCDF<<<grid,block>>>(g_img, g_out_img, h, w, 256);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CDF calculation kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 
    // Copy from Device to host
    size = h*w*sizeof(float);
    err = cudaMemcpy(hist_img, g_out_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void preprocessing::GrayLevel_Neg_Transformation(float *img) // Here img is a gray scale image
{
    cudaError_t err = cudaSuccess;
    size_t size;

    //device input image
    float *g_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    //output image from device
    float *g_out_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_out_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy Memory to device
    size = h*w*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Call kernel function
    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    negative_transformation<<<grid, block>>>(g_img, g_out_img, h, w, 256);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch negative_transformation kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy from Device to host
    size = h*w*sizeof(float);
    err = cudaMemcpy(trans_img, g_out_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void preprocessing::GrayLevel_Log_Transformation(float *img) // Here img is a gray scale image
{
    cudaError_t err = cudaSuccess;
    size_t size;

    //device input image
    float *g_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    //output image from device
    float *g_out_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_out_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy Memory to device
    size = h*w*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Call kernel function
    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    log_transformation<<<grid, block>>>(g_img, g_out_img, h, w, 256 , ENHANCE_PARAM);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch logarithmic transformation kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy from Device to host
    size = h*w*sizeof(float);
    err = cudaMemcpy(trans_img, g_out_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

void preprocessing::GrayLevel_Gam_Transformation(float *img) // Here img is a gray scale image
{
    cudaError_t err = cudaSuccess;
    size_t size;

    //device input image
    float *g_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    //output image from device
    float *g_out_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_out_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy Memory to device
    size = h*w*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Call kernel function

    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    gamma_transformation<<<grid, block>>>(g_img, g_out_img, h, w, 256 , GAMMA);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch power law transformation kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy from Device to host
    size = h*w*sizeof(float);
    err = cudaMemcpy(trans_img, g_out_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}
void preprocessing::Normalization(float *img) // Here img is a gray scale image
{
    cudaError_t err = cudaSuccess;
    size_t size;

    //device input image
    float *g_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    //output image from device
    float *g_out_img = NULL;   
    size = h*w*sizeof(float);
    err = cudaMalloc((void **)&g_out_img, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy Memory to device
    size = h*w*sizeof(float);
    err = cudaMemcpy(g_img, img, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector img from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Call kernel function

    dim3 grid(1,1,1);
    dim3 block(w,h,1);  
    normalize_img<<<grid, block>>>(g_img, g_out_img, h, w, 256);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch normalization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    // Copy from Device to host
    size = h*w*sizeof(float);
    err = cudaMemcpy(norm_img, g_out_img, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out_img from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }   

    // Free device global memory
    err = cudaFree(g_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out_img);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out_img (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}

//////////////////////////////// Dropout /////////////////////////////////////

Dropout::Dropout(float drop_prob, int h, int w, int channel)
{
    this->drop_prob = drop_prob;
    this->h = h;
    this->w = w;
    this->channel = channel;
    mask = new bool[h*w*channel];
    d_in = new float[h*w*channel];
}

float* Dropout::forward(float *in)
{
    // Generate Random numbers
    curandState* devStates;
    int N = h*w*channel;
    cudaMalloc(&devStates, N*sizeof(curandState));

    // setup seeds
    dim3 grid0(1,1,channel);
    dim3 block0(w,h,1);
    setup_kernel<<<grid0,block0>>>(devStates,unsigned(time(NULL)),h,w);
    float* N2 = new float[N];
    float* N3;
    cudaMalloc((void**) &N3, sizeof(float)*N);

    dim3 grid1(1,1,channel);
    dim3 block1(w,h,1);
    kernel<<<grid1,block1>>> (N3, devStates, N,h,w);

    // Forward Prop
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
    size = h*w*channel*sizeof(bool);
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
    dropout_fp<<<grid, block>>>(g_in,g_mask,drop_prob,h,w,channel,N3);   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch dropout_fp kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Copy Memory from device to host
    size = h*w*channel*sizeof(bool);
    err = cudaMemcpy(mask, g_mask, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_mask from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = h*w*channel*sizeof(float);
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

    err = cudaFree(N3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector N3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(devStates);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector devStates (error code %s)!\n", cudaGetErrorString(err));
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
    size = h*w*channel*sizeof(bool);
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

    size = h*w*channel*sizeof(bool);
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


float* concat(float *a1,float *a2,float *a3,int size_a1,int size_a2,int size_a3)
{
    float *out = new float[(size_a1+size_a2+size_a3)];
    cudaError_t err = cudaSuccess;
    size_t size;

    float *g_a1 = NULL;   // g stands for GPU
    size = size_a1*sizeof(float);
    err = cudaMalloc((void **)&g_a1, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_a1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_a2 = NULL;   // g stands for GPU
    size = size_a2*sizeof(float);
    err = cudaMalloc((void **)&g_a2, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_a2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_a3 = NULL;   // g stands for GPU
    size = size_a3*sizeof(float);
    err = cudaMalloc((void **)&g_a3, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_a3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *g_out = NULL;   // g stands for GPU
    size = (size_a1+size_a2+size_a3)*sizeof(float);
    err = cudaMalloc((void **)&g_out, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = size_a1*sizeof(float);
    err = cudaMemcpy(g_a1,a1, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_a1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = size_a2*sizeof(float);
    err = cudaMemcpy(g_a2,a2, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_a2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = size_a3*sizeof(float);
    err = cudaMemcpy(g_a3,a3, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_a3 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int g = (ceil)((size_a1 + size_a2 + size_a3)/1024.0);
    dim3 grid(1,1,g);
    dim3 block(1,32,32);
    concat<<<grid,block>>>(g_a1,g_a2,g_a3,g_out,size_a1,size_a2,size_a3);  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch concat kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    size = (size_a1 + size_a2 + size_a3)*sizeof(float);
    err = cudaMemcpy(out,g_out, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_out from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

       // Free device global memory
    err = cudaFree(g_a1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_a1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_a2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_a2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_a3);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_a3 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(g_out);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector g_out (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return out;
}
