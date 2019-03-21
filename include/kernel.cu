__global__ void conv_fp(float* d_out, float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width){
	int t_x = blockIdx.x*blockDim.x + threadIdx.x;
	int t_y = blockIdx.y*blockDim.y + threadIdx.y;
	int t_z = blockIdx.z*blockDim.z + threadIdx.z;
	
	d_out[t_z*(img_width*img_height) + t_y*img_width + t_x] = 0;
	
	for(int i = t_y - kernel_size/2; i <= t_y + kernel_size/2; i++)
	{
		for(int j = t_x - kernel_size/2; j <= t_x + kernel_size/2; j++)
		{
			if(i < 0 || j < 0 || i >= img_height || j >= img_width)
				continue;
			
			
			for(int k = 0; k < channel_in; k++)
			{
				d_out[t_z*(img_width*img_height) + t_y*img_width + t_x] += d_img[k*(img_width*img_height) + i*img_width + j]*d_filter[t_z*(kernel_size*kernel_size*channel_in) + k*(kernel_size*kernel_size) + i - t_y + kernel_size/2 + j - t_x + kernel_size/2];
			}
		}
	}
}

__global__ void conv_bp(float* d_del_weight, float* d_input, float* d_del_out, int channel_in,int channel_out,int kernel_size,int input_height,int input_width)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tz = blockIdx.z*blockDim.z + threadIdx.z;

	int w_index = blockIdx.z*(kernel_size*kernel_size*channel_in) + threadIdx.z*(kernel_size*kernel_size) + ty*input_width tx;
	d_del_out[w_index] = 0;
	
	for(int i = ty - kernel_size/2; i < ty + input_height - kernel_size/2; i++)
	{
		for(int j = tx - kernel_size/2; j < tx + input_width - kernel_size/2; j++)
		{
			if(i < 0 || j < 0 || i >= input_height || j >= input_width)
				continue;
			int input_index = threadIdx.z*(input_height*input_width) + i*input_width + j;
			int del_out_index = blockIdx.z*(input_height*input_width) + (i - ty + kernel_size/2)*input_width + (j - tx + kernel_size/2);
			d_del_weight[w_index] += d_input[input_index]*d_del_out[del_out_index];
		}
	}
}
