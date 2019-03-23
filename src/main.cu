#include <iostream>
#include <stdlib.h>
#include "../include/header.h"

using namespace std;

// Max pooling

// int h = 4;
// int w = 4;

// int main()
// {
//     float *M=(float*)malloc(h*w*sizeof(float));
//     for(int i=0;i<h*w;i++)
//    	{
//    		M[i]=i;
//    		// cout<<M[i]<<" ";	
//    	} 
//    	// cout<<endl;

//     max_pool obj(h,w);

//     obj.forward(M,h,w,1);

//     cout<<"d_out : \n";
//     float *dM=(float*)malloc(h/2*w/2*sizeof(float));
//     for(int i=0;i<h/2*w/2;i++)
//    	{
//    		dM[i]=i+1;
//    	  cout<<M[i]<<" ";	
//    	} 
//     cout<<"\n\n";

//     cout<<"Mask : \n";
//     for(int i=0;i<h*w/4;i++)
//       cout << obj.mask[i] << "  ";
//     cout<<"\n\n";

//     obj.backward(dM,h/2,w/2);

//     cout<<"d_in : \n";
//     for(int i=0;i<h*w;i++)
//       cout << obj.d_in[i] << "  ";
//     cout<<"\n";

// 	return 0;
// }


// // Matrix multiplication

// int m = 4;

// int main()
// {
//   float *M1=(float*)malloc(m*m*sizeof(float)); 
//   float *M2=(float*)malloc(m*m*sizeof(float)); 
//   float *M3=(float*)malloc(m*m*sizeof(float));

//   for(int i=0;i<m*m;i++)
//   {
//     M1[i] = i+1;
//     M2[i] = m*m-i;
//   }

//   cudaError_t err = cudaSuccess;
//   size_t size;
//   size = m*m*sizeof(float);

//   float *d_M1 = NULL;
//   err = cudaMalloc((void **)&d_M1, size);

//   float *d_M2 = NULL;
//   err = cudaMalloc((void **)&d_M2, size);

//   float *d_M3 = NULL;
//   err = cudaMalloc((void **)&d_M3, size);


//   err = cudaMemcpy(d_M1, M1, size, cudaMemcpyHostToDevice);
//   err = cudaMemcpy(d_M2, M2, size, cudaMemcpyHostToDevice);

// // Launch the Vector Add CUDA Kernel
//   dim3 grid(1,1,1);
//   dim3 block(4,4,1);  
//   matrix_mul<<<grid, block>>>(d_M1,d_M2,d_M3,m,m,m);  
//   err = cudaMemcpy(M3, d_M3, size, cudaMemcpyDeviceToHost);

//   for(int i=0;i<m*m;i++)
//     cout<<M3[i]<<"  ";
//   cout<<"\n";

//   return 0;
// }

// Fully Connected Layer

int main()
{
  int in_size = 4;
  int out_size = 2;
  float *in=(float*)malloc(in_size*sizeof(float));
  for(int i=0; i<in_size; i++)
    in[i] = i+1;

  float *d_out=(float*)malloc(out_size*sizeof(float));
  for(int i=0; i<out_size; i++)
    d_out[i] = i+1;

  FC obj(in_size,out_size);
  obj.forward(in, in_size, out_size);

  cout<<"In : \n";
  for(int i=0;i<in_size;i++)
    cout<<in[i]<<"  ";
  cout<<"\n\n";

  cout<<"Weights : \n";
  for(int i=0;i<out_size*in_size;i++)
    cout<<obj.weight[i]<<"  ";
  cout<<"\n\n";
  
  cout<<"Bias : \n";
  for(int i=0; i<out_size;i++)
    cout<<obj.bias[i]<<"  ";
  cout<<"\n\n";

  cout<<"Out : \n";
  for(int i=0; i<out_size;i++)
    cout<<obj.out[i]<<" ";
  cout<<"\n\n";

  obj.backward(in,d_out,in_size,out_size);

  cout<<"d_in : \n";
  for(int i=0; i<in_size;i++)
    cout<<obj.d_in[i]<<" ";
  cout<<"\n\n";

  cout<<"dW : \n";
  for(int i=0;i<out_size*in_size;i++)
    cout<<obj.dw[i]<<"  ";
  cout<<"\n\n";

  cout<<"db : \n";
  for(int i=0; i<out_size;i++)
    cout<<obj.db[i]<<"  ";
  cout<<"\n\n";


  return 0;
}