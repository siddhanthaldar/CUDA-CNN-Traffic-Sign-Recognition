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
=======
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
				int out_index = t_z*(img_width*img_height) + t_y*img_width + t_x;
				int img_index = k*(img_width*img_height) + i*img_width + j;
				int filter_index = t_z*(kernel_size*kernel_size*channel_in) + k*(kernel_size*kernel_size) + i - t_y + kernel_size/2 + j - t_x + kernel_size/2;
				d_out[out_index] += d_img[img_index]*d_filter[filter_index];
			}
		}
	}
}

__global__ void conv_bp(float* d_del_weight, float* d_input, float* d_del_out, int channel_in,int channel_out,int kernel_size,int input_height,int input_width)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;

	int w_index = blockIdx.z*(kernel_size*kernel_size*channel_in) + threadIdx.z*(kernel_size*kernel_size) + ty*kernel_size + tx;
	d_del_weight[w_index] = 0;
	
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

__global__ void rotate(float* d_weight_t, float* d_weight, int channel_in, int channel_out, int kernel_size)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tz = blockIdx.z*blockDim.z + threadIdx.z;

	d_weight_t[tz*kernel_size*kernel_size + ty*kernel_size + tx] = d_weight[tz*kernel_size*kernel_size + (kernel_size-ty - 1)*kernel_size + kernel_size-tx - 1];
}


__global__ void conv_bp_x(float* d_del_input, float* d_del_out, float* d_weight_t, int channel_in,int channel_out,int kernel_size,int input_height,int input_width)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	// int tz = blockIdx.z*blockDim.z + threadIdx.z;

	// for(int i = ty - kernel_size/2; i <= ty + kernel_size/2; i++)
	// {
	// 	for(int j = tx - kernel_size/2; j <= tx + kernel_size/2; j++)
	// 	{
	// 		if(i < 0 || j < 0 || i >= input_height || j >= input_width)
	// 			continue;
	// 		int input_index = (blockIdx.z % channel_in)*(input_height*input_width) + ty*input_width + tx;
	// 		int out_index = (blockIdx.z / channel_in)*(input_height*input_width) + i*input_width + j;
	// 		int filter_index = blockIdx.z*kernel_size*kernel_size + (i - ty + kernel_size/2)*kernel_size + j - tx + kernel_size/2;
	// 		d_del_input[input_index] += d_del_out[out_index]*d_weight_t[filter_index];
	// 	}
	// }
	for(int i = ty - kernel_size/2; i <= ty + kernel_size/2; i++)
	{
		for(int j = tx - kernel_size/2; j <= tx + kernel_size/2; j++)
		{
			if(i < 0 || j < 0 || i >= input_height || j >= input_width)
				continue;
			int input_index = (blockIdx.z)*(input_height*input_width) + ty*input_width + tx;
			for(int k=0; k<channel_out; k++)
			{
				int out_index = k*(input_height*input_width) + i*input_width + j;
				int filter_index = k*kernel_size*kernel_size*channel_in + (i - ty + kernel_size/2)*kernel_size + j - tx + kernel_size/2;
				d_del_input[input_index] += d_del_out[out_index]*d_weight_t[filter_index];
			}
		}
	}

}



__global__ void conv_step(float* d_weight, float* d_del_weight, float* d_del_vw, int channel_in, int channel_out, int kernel_size, float l_rate, float beeta, bool is_first)
{
	int index = blockIdx.z*kernel_size*kernel_size*channel_in + threadIdx.z*kernel_size*kernel_size + threadIdx.y*kernel_size + threadIdx.x;
	if(is_first)
		d_del_vw[index] = d_del_weight[index];
	else
		d_del_vw[index] = beeta*d_del_vw[index] + (1 - beeta)*d_del_weight[index];

	d_weight[index] -= l_rate*d_del_vw[index];
}
