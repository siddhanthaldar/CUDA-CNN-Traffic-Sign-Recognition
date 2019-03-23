#include "../include/header.h"
/**
 * Host main routine
 */

int main(void)
{
    cout<<"Convolution and update"<<endl;
    int input_size = 32;
    int kernel_size = 3;
    int channel_in = 3;
    int channel_out = 16;
    int numElements = input_size*input_size;
    float *image = new float[numElements*channel_in];
    float *del_out = new float[numElements*channel_out];
    for(int i = 0; i<numElements*channel_in; i++){
        image[i] = 1;
    }
    for(int i = 0; i<numElements*channel_out; i++){
        del_out[i] = 1;
    }
    Conv2d h(channel_in,channel_out,kernel_size);
    float* out = h.forward(image, input_size, input_size);
    cout<<"Forward output : \n";
    for(int i = 0; i<numElements*channel_out; i++){
        if(i%input_size == 0) cout<<endl;
        cout<<out[i]<<" ";
    }
    cout<<endl<<"The derivative of input: \n";
    float* del_input = h.backward(del_out, image, input_size, input_size);
    for(int i = 0; i<numElements*channel_in; i++){
        if(i%input_size == 0) cout<<endl;
        cout<<del_input[i]<<" ";
    }
    cout<<"\nThe derivative of weight: \n";
    for(int i = 0; i<kernel_size*kernel_size*channel_in*channel_out; i++){
        if(i%kernel_size== 0) cout<<endl;
        cout<<h.del_weight[i]<<" ";
    }
    cout<<endl;
    h.step(0.0001);
    cout<<"The updated weight : \n";
    for(int i = 0; i<kernel_size*kernel_size*channel_in*channel_out; i++){
        if(i%kernel_size== 0) cout<<endl;
        cout<<h.weight[i]<<" ";
    }
    cout<<endl;
    // h.step(0.1);
    // cout<<"The updated weight : \n";
    // for(int i = 0; i<kernel_size*kernel_size*channel_in*channel_out; i++){
    //     if(i%kernel_size== 0) cout<<endl;
    //     cout<<h.weight[i]<<" ";
    // }


}