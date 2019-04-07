#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "../src/kernel.cu"

#define ENHANCE_PARAM 45.986  //parameter which adjusts enhancement in log transformation(255/log(256) - scale b/w 0 - 255))
#define GAMMA 0.5  //parameter used for gamma correction

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

class preprocessing
{
public:
    float *gray_img, *hist_img, *trans_img;
    int h,w,channels;
    preprocessing(int h, int w, int channels);
    void BGR2GRAY(float* img);
    void Histogram_Equalization(float *img);
    void GrayLevel_Neg_Transformation(float *img);
    void GrayLevel_Log_Transformation(float *img);
    void GrayLevel_Gam_Transformation(float *img);
};

preprocessing::preprocessing(int h, int w, int channels)
{
    this->h = h;
    this->w = w;
    this->channels = channels;

    gray_img = new float[h*w];
    hist_img = new float[h*w];
    trans_img = new float[h*w];
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
