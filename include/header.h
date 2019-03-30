#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
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

	float **weight;
	float bias;

	FC(int in_features, int out_features);
	void forward();
	void backward();
};

class max_pool
{
public:
	int** mask; //to remember the location
	void forward();
	void backward();
};

class ReLU
{
public:
	void forward();
	void backward();
};

class softmax_and_loss
{
public:
	float loss;
	float* forward(float* logits, int label, int n_classes);
	float* backward(float* out, int label, int n_classes);
};

class dropout
{
public:
	bool** mask;
	void forward();
	void backward();
};

float* softmax_and_loss::forward(float* logits, int label, int n_classes)
{
	double sum = 0;
	float m = 0;
	for(int i = 0; i < n_classes; i++)
		m = max(m, logits[i]);
	for(int i=0; i<n_classes; i++){
		logits[i] -= m;
		sum += exp(logits[i]);
	}
	float* out = new float[n_classes];
	for(int i=0; i<n_classes; i++){
		out[i] = exp(logits[i])/sum;
	}
	loss = 0.0;
	for(int i = 0; i<n_classes; i++){
		if(i!= label) loss -= log(1-out[i] + 1e-5);
		else loss -= log(out[i] + 1e-5); 
	}
	return (out);
}

float* softmax_and_loss::backward(float* out, int label, int n_classes){
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

	printf("Copy input feature map from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_img, image, size_img, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy input weight filter from the host memory to the CUDA device\n");
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

	printf("Copy output data from the CUDA device to the host memory\n");
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
	printf("Partyyyy\n");
	return (h_out);
}

//************************************//
//Backward propagation for convolution//
//************************************//

float* Conv2d::backward(float* del_out, float* input, int input_height, int input_width)
{
	cudaError_t err = cudaSuccess;
	dim3 grid(1, 1, channel_out);
	dim3 block(kernel_size, kernel_size, channel_in);

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

	printf("Copy input feature map from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_input, input, size_del_input, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_img (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy input weight filter from the host memory to the CUDA device\n");
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

	printf("Copy output data from the CUDA device to the host memory\n");
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

	printf("Copy weight data from the host to the CUDA memory\n");
	err = cudaMemcpy(d_weight, weight, size_del_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dim3 grid_t(1,1,channel_out);
	dim3 block_t(kernel_size, kernel_size, channel_in);


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

	printf("Copy input weight filter from the host memory to the CUDA device\n");
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

	printf("Copy output data from the CUDA device to the host memory\n");
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
	printf("Partyyyy\n");
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

	printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_weight, weight, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_weight, del_weight, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_del_vw, del_vw, size_weight, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy d_del_vw (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	dim3 grid(1,1,channel_out);
	dim3 block(kernel_size, kernel_size, channel_in);

	conv_step<<<grid, block>>>(d_weight, d_del_weight, d_del_vw, channel_in, channel_out, kernel_size, l_rate, beeta, is_first);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch  kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(weight, d_weight, size_weight, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy weight (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("Copy output data from the CUDA device to the host memory\n");
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