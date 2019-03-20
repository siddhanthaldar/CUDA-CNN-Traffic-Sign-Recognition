#include <iostream>
#include <cuda.h>

__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width){

class Conv2d
{
public:

	float ****weight; //channels_out, channels_in, kernel_size, kernel_size
	float bias;
	int channel_in, channel_out, kernel_size;

	Conv2d(int channel_in, int channel_out, int kernel_size);
	void forward();
	void backward();
};

Conv2d::Conv2d(int channel_in, int channel_out, int kernel_size)
{
	this->channel_in=channel_in;
	this->channel_out = channel_out;
	this->kernel_size = kernel_size;
	// weight = new float***[channel_out];
	// for(int i = 0; i < channel_out; i++)
	// {
	// 	weight[i] = new float**[channel_in];
	// 	for(int j = 0; j < kernel_size; j++)
	// 	{
	// 		weight[i][j] = new float*[kernel_size];
	// 		for(int k = 0; k < kernel_size; k++)
	// 		{
	// 			weight[i][j][k] = new float[kernel_size];
	// 			for(int l = 0; l < kernel_size; l++)
	// 				weight[i][j][k][l] = 0.0; //replace with random
	// 		}
	// 	}
	// }
	weight = new float[channel_out*kernel_size*kernel_size*channel_in]();//Initialize the weights
	for(int i = 0; i < channel_out*kernel_size*kernel_size*channel_in; i++)
		weight[i] = rand()/RAND_MAX;
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

class max_pool()
{
public:
	int** mask; //to remember the location
	void forward();
	void backward();
};

class ReLU()
{
public:
	void forward();
	void backward();
};

class softmax()
{
public:
	void forward();
	void backward();
};

class dropout()
{
public:
	bool** mask;
	void forward();
	void backward();
};

void Conv2d::forward(float* image, int img_height, int img_width)
{
	float* h_out = new float[img_height*img_width*channel_out]();

	dim3 grid(1,1,channel_out);
	dim3 block(img_height,img_width,1);

	size_t size_img = img_width*img_height*channel_in;
	size_t size_filter = kernel_size*kernel_size*channel_out*channel_in;
	size_t size_out = img_height*img_width*channel_out;

	float *d_img = NULL;
	err = cudaMalloc((void **)&d_img, size_img);

	float *d_filter = NULL;
	err = cudaMalloc((void **)&d_filter, size_filter);

	float* d_out = NULL;
	err = cudaMalloc((void **)&d_out, size_out);

	printf("Copy input feature map from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_img, image, size_img, cudaMemcpyHostToDevice);

	printf("Copy input weight filter from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_filter, weight, size_filter, cudaMemcpyHostToDevice);

	conv_fp <<<grid,block>>>(d_out, d_img, d_filter, channel_in, channel_out, kernel_size, img_height, img_width);

	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    err = cudaFree(d_filter);
    err = cudaFree(d_img);
    err = cudaFree(d_out);

    err = cudaDeviceReset();
    cout<<"Partyyyy\n";
}