
__global__ void 
maxpool_fp(float *in, float *out, int *mask, int h, int w) // h and w are sizes of in matrix
{	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	// Mask contains position from where the value is obtained
	// 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
	if(i<h/2 && j<w/2)
	{		
		out[i*w/2+j] = in[2*i*w+j*2];
		mask[i*w/2+j] = 0;

		if(out[i*w/2+j]<in[2*i*w+j*2+1])
		{
			out[i*w/2+j] = in[2*i*w+j*2+1];
			mask[i*w/2+j] = 1;
		}

		if(out[i*w/2+j]<in[(2*i+1)*w+j*2])
		{
			out[i*w/2+j] = in[(2*i+1)*w+j*2];
			mask[i*w/2+j] = w;
		}

		if(out[i*w/2+j]<in[(2*i+1)*w+j*2+1])
		{
			out[i*w/2+j] = in[(2*i+1)*w+j*2+1];
			mask[i*w/2+j] = w+1;
		}

	}
}

__global__ void 
maxpool_bp(float *d_in, float *d_out, int *mask, int h, int w)  // h and w are dim of d_out
{	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockDim.y*blockDim.y + threadIdx.y;

	if(i<h && j<w)
	{
		d_in[2*i*2*w+j*2+mask[i*w+j]] = d_out[i*w+j];
	}
	d_in[0] = mask[i*w+j];
}

