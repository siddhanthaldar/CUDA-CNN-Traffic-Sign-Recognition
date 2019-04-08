#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ 
float generate( curandState* globalState, int ind ) 
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ 
void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ 
void kernel(float* N, curandState* globalState, int n)
{
    // generate random numbers
    for(int i=0;i<n;i++)
    {
        float k = generate(globalState, i);
        N[i] = k;
    }
}

__global__
void dropout_fp(float *in, bool *mask, float drop_prob, int h, int w, int channels, float *rand)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;
	int k = blockIdx.z * blockDim.z + threadIdx.z ;

	if(i<h && j<w && k<channels)
	{
		if(rand[k*h*w+i*w+j] < drop_prob)
		{
			in[k*h*w+i*w+j] = 0;
			mask[k*h*w+i*w+j] = 1;
		}	
	}
}


// __global__
// void dropout_fp(float *in, bool *mask, float drop_prob, int h, int w, int channels)
// {
// 	int i = blockIdx.y * blockDim.y + threadIdx.y ;
// 	int j = blockIdx.x * blockDim.x + threadIdx.x ;
// 	int k = blockIdx.z * blockDim.z + threadIdx.z ;

// 	if(i<h && j<w && k<channels)
// 	{
// 		if(rand()*1.0/(float)RAND_MAX < drop_prob)
// 		{
// 			in[k*h*w + i*w+j] = 0;
// 			mask[i*w+j] = 1;
// 		}	
// 	}
// }

__global__
void dropout_bp(float *d_out, float* d_in, bool *mask, int h, int w, int channels)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;
	int k = blockIdx.z * blockDim.z + threadIdx.z ;

	if(i<h && j<w && k<channels)
	{
		if(mask[k*h*w+i*w+j] == 0)
			d_in[k*h*w + i*w+j] = d_out[k*h*w + i*w+j];
		else	
			d_in[k*h*w + i*w+j] = 0;
	}
}

