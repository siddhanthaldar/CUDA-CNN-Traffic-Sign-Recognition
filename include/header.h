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

// class FC
// {
// public:

// 	float **weight;
// 	float bias;

// 	FC(int in_features, int out_features);
// 	void forward();
// 	void backward();
// };

class max_pool
{
public:
	int *mask; //to remember the location
	float *out, *d_in;
    // int h,w;
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

max_pool::max_pool(int h, int w)
{
	d_in = (float *)calloc(h*w, sizeof(float));
    for(int x=0;x<h*w;x++)
        d_in[x] = 2;
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
    // int blocksPerGrid = (int)(h/2*w/2)/1024;
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
    
    d_out[0] = 5;
    for(int x=0;x<h*w;x++)
        cout << d_out[x] << " ";
    cout << "\n";

    size = h*w*sizeof(float);
    err = cudaMemcpy(g_d_out, d_out, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g_d_out from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    // for(int x=0; x<h*w;x++)
    //     cout<<mask[x]<<"  ";
    // cout<<endl;

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