#include <iostream>
#include <stdlib.h>
#include "../include/header.h"

using namespace std;

int h = 4;
int w = 4;

int main()
{
    float *M=(float*)malloc(h*w*sizeof(float));
    for(int i=0;i<h*w;i++)
   	{
   		M[i]=i;
   		// cout<<M[i]<<" ";	
   	} 
   	// cout<<endl;

    max_pool obj(h,w);

    obj.forward(M,h,w,1);

    float *dM=(float*)malloc(h/2*w/2*sizeof(float));
    for(int i=0;i<h/2*w/2;i++)
   	{
   		dM[i]=i;
   		// cout<<M[i]<<" ";	
   	} 

    obj.backward(dM,h/2,w/2);

    for(int i=0;i<h*w;i++)
    cout << obj.d_in[i] << "\n";

	return 0;
}