#include <iostream>
#include <cuda.h>

class Conv2d
{
public:

	float ****weight; //channels_out, channels_in, kernel_size, kernel_size
	float bias;

	Conv2d(int channel_in, int channel_out, int kernel_size);
	void forward();
	void backward();
};

Conv2d::Conv2d(int channel_in, int channel_out, int kernel_size)
{
	weight = new float***[channel_out];
	for(int i = 0; i < channel_out; i++)
	{
		weight[i] = new float**[channel_in];
		for(int j = 0; j < kernel_size; j++)
		{
			weight[i][j] = new float*[kernel_size];
			for(int k = 0; k < kernel_size; k++)
			{
				weight[i][j][k] = new float[kernel_size];
				for(int l = 0; l < kernel_size; l++)
					weight[i][j][k][l] = 0.0; //replace with random
			}
		}
	}

	bias = 0.0;//replace with random
}

class FC
{
public:

	float **weights;
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