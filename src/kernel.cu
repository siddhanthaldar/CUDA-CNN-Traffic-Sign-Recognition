__device__ void 
matrix_mul(float *d_M, float *d_N, float *d_P , int m, int n, int p)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;
	if(( i < m ) && (j < p )){
		float Pvalue = 0.0;
		for (int k = 0; k < n ; ++ k){
			Pvalue += d_M [ i * n + k ]* d_N [ k * p + j ];
		}
		d_P [ i * p + j ] = Pvalue ;
	}
}

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
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	if(i<h && j<w)
	{
		d_in[2*i*2*w+j*2+mask[i*w+j]] = d_out[i*w+j];
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
FC_bp(float *d_out, float *w, float *b, float *in, int m,int n,int p)  // same as in fp
{
	int i = blockIdx.y * blockDim.y + threadIdx.y ;
	int j = blockIdx.x * blockDim.x + threadIdx.x ;

	// if(i<m && j<p)
	// {
		
	// }

}





// const int TILE_WIDTH = 1;

// __global__ void 
// matrix_mul(float *d_M, float *d_N, float *d_P, int Width)
// {
// 	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
// 	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
// 	int bx = blockIdx.x ;
// 	int by = blockIdx.y ;
// 	int tx = threadIdx.x ;
// 	int ty = threadIdx.y ;

// 	int Row = by * TILE_WIDTH + ty ;
// 	int Col = bx * TILE_WIDTH + tx ;
// 	float Pvalue = 0;
// 	for(int m=0; m<Width/TILE_WIDTH;++m) 
// 	{
// 		Mds[ty][tx] = d_M[Row*Width+m*TILE_WIDTH + tx];
// 		Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * Width + Col];
// 		__syncthreads();
// 		for(int k = 0; k<TILE_WIDTH; ++k)
// 			Pvalue += Mds[ty][k] * Nds[k][tx];
// 		__syncthreads() ;
// 	}
// 	d_P[ Row * Width + Col] = Pvalue ;
// }
