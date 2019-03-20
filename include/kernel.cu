__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width){
	int t_x = blockIdx.x*blockDim.x + threadIdx.x;
	int t_y = blockIdx.y*blockDim.y + threadIdx.y;
	int t_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	for(int i = ty - kernel_size/2; i <= ty + kernel_size/2; i++)
	{
		for(int j = tx - kernel_size/2; j <= tx + kernel_size/2; j++)
		{
			if(i < 0 || j < 0 || i >= img_height || j >= img_width)
				continue;
			d_out[t_z*(img_width*img_height) + i*img_width + j] = 0;
			for(int k = 0; k < channel_in; k++)
			{
				d_out[t_z*(img_width*img_height) + i*img_width + j] += d_img[k*(img_width*img_height) + i*img_width + j]*d_filter[t_z*(kernel_size*kernel_size*channel_in) + k*(kernel_size*kernel_size) + (i - ty + kernel_size/2) + j - tx + kernel_size/2];
			}
		}
	}
}