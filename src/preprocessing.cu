#include "../include/header.h"
/**
 * Host main routine
 */
#define TEST
int main(void)
{
    int input_size = 32;
    int channel_in = 3;
    int channel_out= 1;
    int numElements = input_size*input_size;
    float *image = new float[numElements*channel_in];
    float *gray_img = new float[numElements*channel_out];
    #ifdef TEST
    {
        for(int i = 0; i<numElements*channel_in; i++){
            image[i] = i%100;// +123 ;
        }
    }
    #endif

    preprocessing gray(input_size, input_size, channel_in);

    gray.BGR2GRAY(image);
    
    #ifdef TEST
    {
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
    }
    #endif
    /* //for checking
    float *tmp = new float[numElements*channel_out];
    float tmp1[16] = {4,1,3,2,3,1,1,1,0,1,5,2,1,1,2,2};
    tmp = tmp1;
    */
    gray.Histogram_Equalization(gray.gray_img);
    
    #ifdef TEST
    {
        cout<<"Histogram Equalized Image: \n";
        for(int i = 0; i<numElements; i++)
            cout<<gray.hist_img[i]<<"   ";
        cout<<"\n\n";
    }
    #endif

    gray.GrayLevel_Neg_Transformation(gray.hist_img); //can send any of gray-scale image,hist-equalized image or transformed(if called before) 
    #ifdef TEST
    {
        cout<<"Negative Transformed Image: \n";
        for(int i = 0; i<numElements; i++)
            cout<<gray.trans_img[i]<<"   ";
        cout<<"\n\n";
    }
    #endif
    gray.GrayLevel_Log_Transformation(gray.hist_img);
    #ifdef TEST
    {
        cout<<"Logarithmic Transformed Image: \n";
        for(int i = 0; i<numElements; i++)
            cout<<gray.trans_img[i]<<"   ";
        cout<<"\n\n";
    }
    #endif
    gray.GrayLevel_Gam_Transformation(gray.hist_img);
    #ifdef TEST
    {
        cout<<"Gamma Transformed Image: \n";
        for(int i = 0; i<numElements; i++)
            cout<<gray.trans_img[i]<<"   ";
        cout<<"\n\n";
    }
    #endif
    return 0;

}