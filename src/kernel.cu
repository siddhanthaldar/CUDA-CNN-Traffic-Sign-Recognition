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