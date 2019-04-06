#include "../include/header.h"
/**
 * Host main routine
 */

int main(void)
{
    int input_size = 3;
    int channel_in = 3;
    int channel_out= 1;
    int numElements = input_size*input_size;
    float *image = new float[numElements*channel_in];
    float *gray_img = new float[numElements*channel_out];
    for(int i = 0; i<numElements*channel_in; i++){
        image[i] = i;
    }

    preprocessing gray(input_size, input_size, channel_in);

    gray.BGR2GRAY(image);

    cout<<"Image : \n";
    for(int j=0; j<channel_in; j++){
        for(int i = 0; i<numElements; i++)
            cout<<image[j*numElements + i]<<"   ";
        cout<<"\n";    
    }        
    cout<<"\n";

    cout<<"Gray Image: \n";
    for(int i = 0; i<numElements; i++)
        cout<<gray.gray_img[i]<<"   ";
    cout<<"\n\n";

    // gray.Histogram_Equalization(gray.gray_img);

    // cout<<"Histogram Equalized Image: \n";
    // for(int i = 0; i<numElements; i++)
    //     cout<<gray.hist_img[i]<<"   ";
    // cout<<"\n\n";


    return 0;

}