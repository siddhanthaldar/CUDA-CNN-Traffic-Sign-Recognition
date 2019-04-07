//////////********Actibation Function************////////////////
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
/////////////**************Convolution*****************//////////
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
/////////////***********Max Pool and fully connected**********///////////
__global__ void 
maxpool_fp(float *in, float *out, int *mask, int h, int w,int channel) // h and w are sizes of in matrix
{	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	// Mask contains position from where the value is obtained
	// 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
	if(i<h/2 && j<w/2 && z<channel)
	{		
		out[z*h/2*w/2+i*w/2+j] = in[z*h*w+2*i*w+j*2];
		mask[z*h/2*w/2+i*w/2+j] = 0;

		if(out[z*h/2*w/2+i*w/2+j]<in[z*h*w+2*i*w+j*2+1])
		{
			out[z*h/2*w/2+i*w/2+j] = in[z*h*w+2*i*w+j*2+1];
			mask[z*h/2*w/2+i*w/2+j] = 1;
		}

		if(out[z*h/2*w/2+i*w/2+j]<in[z*h*w+(2*i+1)*w+j*2])
		{
			out[z*h/2*w/2+i*w/2+j] = in[z*h*w+(2*i+1)*w+j*2];
			mask[z*h/2*w/2+i*w/2+j] = w;
		}

		if(out[z*h/2*w/2+i*w/2+j]<in[z*h*w+(2*i+1)*w+j*2+1])
		{
			out[z*h/2*w/2+i*w/2+j] = in[z*h*w+(2*i+1)*w+j*2+1];
			mask[z*h/2*w/2+i*w/2+j] = w+1;
		}

	}
}
__global__ void 
maxpool_bp(float *d_in, float *d_out, int *mask, int h, int w,int channel)  // h and w are dim of d_out
{	
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if(i<h && j<w && z<channel)
	{
		d_in[z*4*h*w+2*i*2*w+j*2+mask[z*h*w+i*w+j]] = d_out[z*h*w+i*w+j];
	}
}

__global__ void
FC_fp(float *in, float *out, float *w, float *b, int m,int n,int p) //m = out_size, n = in_size, p = num _data
{
	// dim3 grid(1,1,1);
	// dim3 block(p,m,1);
	// if(threadIdx.x==0)
	// {
	// 	matrix_mul(w,in,out,m,n,p); //<<<grid,block>>>
	// 	// cudaThreadSynchronize();
	// }	

	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(( i < m ) && (j < p )){
		float Pvalue = 0.0;
		for (int k = 0; k < n ; ++ k){
			Pvalue += in[k]* w[i*n+k];
		}
		out[ i * p + j ] = Pvalue ;
	}

	__syncthreads();

	if(i<m && j<p)
	{
		out[i*p+j] = out[i*p+j] + b[i];
	}
}

__global__ void
FC_bp(float *d_out, float *d_in, float *w, float *w_transpose,float *dw,float *b,float *db,float *in,int m,int n,int p)  // same as in fp
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	if(i<m && j<n)
	{
        int index_in  = i*n+j;
        int index_out = j*m+i;
        w_transpose[index_out] = w[index_in]; 
    }
    __syncthreads();

    if(i<n && j<p){
		float Pvalue = 0.0;
		for (int k = 0; k < m ; ++ k){
			Pvalue += w_transpose[i*m+k] * d_out[k*p + j];
		}
		d_in[i*p+j] = Pvalue ;
	}
    __syncthreads();

    if(i<m && j<n){
		float Pvalue = 0.0;
		for (int k = 0; k < p; ++ k){
			Pvalue += d_out[i*p+k]* in[k*n+j];
		}
		dw[i*n+j] = Pvalue ;
	}
    __syncthreads();

    if(i<m && j<p)
    {
    	db[i*p+j] = d_out[i*p+j];
    }

    __syncthreads();
}

__global__ void
FC_step_w(float *w, float *dw, float *dw_old, float lr, float beta, int m, int n, int p)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	dw[i*n+j] = beta * dw[i*n+j] + (1-beta) * dw_old[i*n+j];
	dw_old[i*n+j] = dw[i*n+j];

	w[i*n+j] -= lr*dw[i*n+j];

}

__global__ void
FC_step_b(float *bias, float *db, float *db_old, float lr, float beta, int m, int n, int p)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	db[i*n+j] = beta * db[i*n+j] + (1-beta) * db_old[i*n+j];
	db_old[i*n+j] = db[i*n+j];
	bias[i*n+j] -= lr*db[i*n+j];
}