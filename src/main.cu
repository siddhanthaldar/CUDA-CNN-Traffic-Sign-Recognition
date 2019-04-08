#include <iostream>
#include <stdlib.h>
#include "../include/header.h"


using namespace std;


int main()
{
  int h = 4;
  int w = 4;
  int channel = 2;
  float drop_prob = 0.5;
  float *in=(float*)malloc(h*w*channel*sizeof(float));
  float *out=(float*)malloc(h*w*channel*sizeof(float));
  for(int i=0; i<h*w*channel; i++)
  {
  	in[i] = i;
  } 

  cout<<"In : \n";
  for(int i=0;i<h*w*channel;i++)
    cout<<in[i]<<"  ";
  cout<<"\n\n";

  Dropout obj(drop_prob,h,w,channel);
  out = obj.forward(in);

  cout<<"Out : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<out[i]<<" ";
  cout<<"\n\n";

  cout<<"Mask : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<obj.mask[i]<<" ";
  cout<<"\n\n";

  float *d_in=(float*)malloc(h*w*channel*sizeof(float));
  float *d_out=(float*)malloc(h*w*channel*sizeof(float));
  for(int i=0; i<h*w*channel; i++)
  {
  	d_out[i] = h*w*channel-i;
  } 

  d_in = obj.backward(d_out);

  cout<<"d_out : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<d_out[i]<<" ";
  cout<<"\n\n";


  cout<<"d_in : \n";
  for(int i=0; i<h*w*channel;i++)
    cout<<d_in[i]<<" ";
  cout<<"\n\n";


  return 0;
}
