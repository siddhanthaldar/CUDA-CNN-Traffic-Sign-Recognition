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
