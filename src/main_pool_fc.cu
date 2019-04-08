#include <iostream>
#include <stdlib.h>
#include "../include/header.h"

using namespace std;

int h = 4;
int w = 4;
int channel = 3;

// int main()
// {
//     float *M=(float*)malloc(h*w*channel*sizeof(float));
//     cout << "in: " << endl;
//     for(int i=0;i<h*w*channel;i++)
//     {
//       M[i]=rand()/(float)RAND_MAX;
//       cout<<M[i]<<" ";  
//     } 
//     cout<<endl;

//     max_pool obj(h,w,channel);

//     obj.forward(M,h,w,channel);
//     cout << "out:" << endl;
//     for(int i=0;i<h/2*w/2*channel;i++)
//     {
//       cout << obj.out[i] << " "; 
//     } 
//     cout << endl;
//     cout<<"d_out : \n";
//     float *dM=(float*)malloc(h/2*w/2*channel*sizeof(float));
//     for(int i=0;i<h/2*w/2*channel;i++)
//     {
//       dM[i]=i+1;
//       cout<<dM[i]<<" "; 
//     } 
//     cout<<"\n\n";

//     cout<<"Mask : \n";
//     for(int i=0;i<h*w/4*channel;i++)
//       cout << obj.mask[i] << "  ";
//     cout<<"\n\n";

//     obj.backward(dM,h/2,w/2,channel);

//     cout<<"d_in : \n";
//     for(int i=0;i<h*w*channel;i++)
//       cout << obj.d_in[i] << "  ";
//     cout<<"\n";

//   return 0;
// }

int main()
{
    float *M1=(float*)malloc(5*sizeof(float));
    float *M2=(float*)malloc(6*sizeof(float));
    float *M3=(float*)malloc(7*sizeof(float));

    cout << "in1: " << endl;
    for(int i=0;i<5;i++)
    {
      M1[i]=i+1;
      cout<<M1[i]<<" ";  
    }
    cout << endl;

    cout << "in2: " << endl;
    for(int i=0;i<6;i++)
    {
      M2[i]=i+1;
      cout<<M2[i]<<" ";  
    }
    cout << endl;

    cout << "in3: " << endl;
    for(int i=0;i<7;i++)
    {
      M3[i]=i+1;
      cout<<M3[i]<<" ";  
    }

    float *out = concat(M1,M2,M3,5,6,7);
    cout << "out: " << endl;
    for(int i=0;i<(5+6+7);i++)
    {
      cout << out[i] << " ";
    }
    cout << endl;
}


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

// int main()
// {
//   int in_size = 4;
//   int out_size = 2;
//   float *in=(float*)malloc(in_size*sizeof(float));
//   for(int i=0; i<in_size; i++)
//     in[i] = i+1;

//   float *d_out=(float*)malloc(out_size*sizeof(float));
//   for(int i=0; i<out_size; i++)
//     d_out[i] = i+1;

//   FC obj(in_size,out_size);
//   obj.forward(in);

//   cout<<"In : \n";
//   for(int i=0;i<in_size;i++)
//     cout<<in[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"Weights : \n";
//   for(int i=0;i<out_size*in_size;i++)
//     cout<<obj.weight[i]<<"  ";
//   cout<<"\n\n";
  
//   cout<<"Bias : \n";
//   for(int i=0; i<out_size;i++)
//     cout<<obj.bias[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"Out : \n";
//   for(int i=0; i<out_size;i++)
//     cout<<obj.out[i]<<" ";
//   cout<<"\n\n";

//   obj.backward(in,d_out);

//   cout<<"d_in : \n";
//   for(int i=0; i<in_size;i++)
//     cout<<obj.d_in[i]<<" ";
//   cout<<"\n\n";

//   cout<<"dW : \n";
//   for(int i=0;i<out_size*in_size;i++)
//     cout<<obj.dw[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"db : \n";
//   for(int i=0; i<out_size;i++)
//     cout<<obj.db[i]<<"  ";
//   cout<<"\n\n";

//   obj.step(0.001,0.9);

//   cout<<"After optimizing : \n\n";

//   cout<<"dW : \n";
//   for(int i=0;i<out_size*in_size;i++)
//     cout<<obj.dw[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"Weights : \n";
//   for(int i=0;i<out_size*in_size;i++)
//     cout<<obj.weight[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"db : \n";
//   for(int i=0; i<out_size;i++)
//     cout<<obj.db[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"Bias : \n";
//   for(int i=0; i<out_size;i++)
//     cout<<obj.bias[i]<<"  ";
//   cout<<"\n\n";


//   return 0;
// }