__global__ void conv_fp(float* d_img,float* d_filter,int channel_in,int channel_out,int kernel_size,int img_height,int img_width){
	int t_x = blockIdx.x*blockDim.x + threadIdx.x;
	int t_y = blockIdx.y*blockDim.y + threadIdx.y;
	int t_z = blockIdx.z*blockDim.z + threadIdx.z;
	
}