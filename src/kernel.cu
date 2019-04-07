
__global__ 
void bgr2gray(float *in_img, float *gray_img, int h, int w, int channel)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if(i<h && j<w)
	{
		// R -> 0.299, G -> 0.587, B -> 0.114
		gray_img[i*w+j] = 0.299*in_img[0*h*w+i*w+j] + 0.587*in_img[1*h*w+i*w+j] + 0.114*in_img[2*h*w+i*w+j];		
	}

}
__device__ int hist[256];
__global__
void histogram_equalization(float *in_img, int h, int w, int num_levels)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;


	if(i*w+j<num_levels)
		hist[i*w+j] = 0;

	 __syncthreads();

	 if(i<h && j<w)
	 {
	 	atomicAdd(&(hist[(int)in_img[i*w+j]]),1);
	 }
	 __syncthreads();

}
__global__ void calcCDF(float* in_img , float* out_img, int h, int w, int num_levels)
{
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	//Sequential Implementation -- can be modified to parallel using partial scan algorithm
	__shared__ float cdf[256];
	if(i*w + j == 0)
	{
		cdf[0] = (float)hist[i*w+j]/(h*w);
		for( int k = 1; k < num_levels; k++)
		{	
			cdf[k] = cdf[k-1] + (float)hist[k]/(h*w);
		}
	}
	__syncthreads();
	if(i*w+j<num_levels)
	 	cdf[i*w+j] = (float)(cdf[i*w+j] * (num_levels-1));
	__syncthreads();
	if(i<h && j<w)
	 	out_img[i*w+j] = (int)cdf[(int)in_img[i*w+j]];
}