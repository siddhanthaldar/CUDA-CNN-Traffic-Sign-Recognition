#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>


__global__
void dropout_fp(float *in, bool *mask, float drop_prob, int h, int w, int channels)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;
	int k = blockIdx.z * blockDim.z + threadIdx.z ;

	if(i<h && j<w && k<channels)
	{
		if(rand()*1.0/(float)RAND_MAX < drop_prob)
		{
			in[k*h*w + i*w+j] = 0;
			mask[i*w+j] = 1;
		}	
	}
}

__global__
void dropout_bp(float *d_out, float* d_in, bool *mask, int h, int w, int channels)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;
	int k = blockIdx.z * blockDim.z + threadIdx.z ;

	if(i<h && j<w && k<channels)
	{
		if(mask[i*w+j] == 1)
			d_in[k*h*w + i*w+j] = d_out[k*h*w + i*w+j];
		else	
			d_in[k*h*w + i*w+j] = 0;
	}
}

__device__ 
float generate( curandState* globalState, int ind ) 
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}