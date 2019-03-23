#include <iostream>
#include <stdlib.h>
#include "../include/header.h"

using namespace std;

// // ReLU

// int main()
// {
//   int h = 4;
//   int w = 4;
//   int channel = 2;
//   float *in=(float*)malloc(h*w*channel*sizeof(float));
//   for(int i=0; i<h*w*channel; i++)
//   {
//   	if(i%2) in[i] = -1.0*i;
//   	else in[i] = i;
//   } 

//   ReLU obj(h,w,channel);
//   obj.forward(in,h,w,channel);

//   cout<<"In : \n";
//   for(int i=0;i<h*w*channel;i++)
//     cout<<in[i]<<"  ";
//   cout<<"\n\n";

//   cout<<"Out : \n";
//   for(int i=0; i<h*w*channel;i++)
//     cout<<obj.out[i]<<" ";
//   cout<<"\n\n";

//   float *d_out=(float*)malloc(h*w*channel*sizeof(float));
//   for(int i=0; i<h*w*channel; i++)
//   {
//   	d_out[i] = h*w*channel-i;
//   } 

//   obj.backward(d_out,h,w,channel);

//   cout<<"d_in : \n";
//   for(int i=0; i<h*w*channel;i++)
//     cout<<obj.d_in[i]<<" ";
//   cout<<"\n\n";


//   return 0;
// }

// Sigmoid

int main()
{
  int h = 4;
  int w = 4;
  int channel = 2;
  float *in=(float*)malloc(h*w*channel*sizeof(float));
  for(int i=0; i<h*w*channel; i++)
  {
  	if(i%2) in[i] = -1.0*i;
  	else in[i] = i;
  } 

  Sigmoid obj(h,w,channel);
  obj.forward(in,h,w,channel);

  cout<<"In : \n";
  for(int i=0;i<h*w*channel;i++)
    cout<<in[i]<<"  ";
  cout<<"\n\n";

  cout<<"Out : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<obj.out[i]<<" ";
  cout<<"\n\n";

  float *d_out=(float*)malloc(h*w*channel*sizeof(float));
  for(int i=0; i<h*w*channel; i++)
  {
  	d_out[i] = h*w*channel-i;
  } 

  obj.backward(d_out,h,w,channel);

  cout<<"d_in : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<obj.d_in[i]<<" ";
  cout<<"\n\n";


  return 0;
}