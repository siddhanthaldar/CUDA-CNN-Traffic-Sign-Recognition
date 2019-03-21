#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;
__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width);

class Conv2d
{
public:

	float *weight; //channels_out, channels_in, kernel_size, kernel_size
	float bias;
	float* del_weight;
	int channel_in, channel_out, kernel_size;

	Conv2d(int channel_in, int channel_out, int kernel_size);
	float* forward(float* image, int img_width, int img_height);
	void backward();
};

Conv2d::Conv2d(int channel_in, int channel_out, int kernel_size)
{
	this->channel_in=channel_in;
	this->channel_out = channel_out;
	this->kernel_size = kernel_size;
	// cout<<"\n the weight is: ";
	weight = new float[channel_out*kernel_size*kernel_size*channel_in]();//Initialize the weights
	del_weight = new float[channel_out*kernel_size*kernel_size*channel_in]();//Initialize the weights
	for(int i = 0; i < channel_out*kernel_size*kernel_size*channel_in; i++)
	{
		weight[i] = 1;//rand()/RAND_MAX;
		del_weight[i] = 0;
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

class softmax
{
public:
	void forward();
	void backward();
};

class dropout
{
public:
	bool** mask;
	void forward();
	void backward();
};

float* Conv2d::forward(float* image, int img_height, int img_width)
{
	cudaError_t err = cudaSuccess;
	float* h_out = new float[img_height*img_width*channel_out];
	cout<<"\n THe input is: ";
    for (int i = 0; i<5*5*3;i++)
        cout<<image[i]<<" ";

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


void Conv2d::backward(float* del_out, float* input, int input_height, int input_width)
{
	cudaError_t err = cudaSuccess;

	dim3 grid(1, 1, channel_out);
	dim3 block(kernel_size, kernel_size, channel_in);

	size_t size_del_input = input_width*input_height*channel_in*sizeof(float);
	size_t size_del_weight = kernel_size*kernel_size*channel_out*channel_in*sizeof(float);
	size_t size_del_out = input_height*input_width*channel_out*sizeof(float);


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

    err = cudaFree(d_del_weight);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free d_del_weight (error code %s)!\n", cudaGetErrorString(err));
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
    return (h_out);
}
