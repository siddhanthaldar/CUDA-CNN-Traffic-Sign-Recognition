__global__ void
ReLU_fp(float *out, float *in, int h, int w, int channel)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(i<h && j<w*channel)// && k<channel)
	{
		out[i*w*channel+j] = in[i*w*channel+j]>0?in[i*w*channel+j]:0;   //k*w*h+
	}
}

__global__ void
ReLU_bp(float *d_in, float *d_out, float*out, int h, int w, int channel)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(i<h && j<w*channel)
	{
		d_in[i*w*channel+j] = out[i*w*channel+j]>0?d_out[i*w*channel+j]:0;   //k*w*h+
	}
}

__global__ void
Sigmoid_fp(float *out, float *in, int h, int w, int channel)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(i<h && j<w*channel)
	{
		out[i*w*channel+j] = 1.0/(1+exp(-1.0*in[i*w*channel+j]));   //k*w*h+
	}
}

__global__ void
Sigmoid_bp(float *d_in, float *d_out, float*out, int h, int w, int channel)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(i<h && j<w*channel)
	{
		d_in[i*w*channel+j] = out[i*w*channel+j]*(1-out[i*w*channel+j])*d_out[i*w*channel+j];   //k*w*h+
	}
}
