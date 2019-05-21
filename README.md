# CUDA-CNN

We have implemented the deep neural network consisting of CNN and FC layers in CUDA. Each part has it's own class and their functions are explained below. We have particularly used this for training a traffic sign classifier model but it's scope is not limited to this application of Neural network.  

## Layout of the code

'''
/include/header.h - Contains all the classes such as Conv2d, Fully connected, Max pool, relu etc.
/include/src/kernel.cu - Contains all the kernel function which runs on Cuda
/include/Traffic_sign.cu - Contains our model and data loader for traffic sign recognition
'''

### Prerequisites
Cuda version 9.0
gcc compiler


## Pipeline

### Preprocessing

Class Definition
'''
class preprocessing
{
public:
    
    float *gray_img, *hist_img, *trans_img, *norm_img;
    int h,w,channels;
    
    preprocessing(int h, int w, int channels);
    
    void BGR2GRAY(float* img);
    void Histogram_Equalization(float *img);
    void GrayLevel_Neg_Transformation(float *img);
    void GrayLevel_Log_Transformation(float *img);
    void GrayLevel_Gam_Transformation(float *img);
    void Normalization(float *img);
};
'''
The parameters of different functions for this class are explained below -:

'''
void bgr2gray(float *in_img, float *gray_img, int h, int w, int channel);
void histogram_equalization(float *in_img, int h, int w, int num_levels);
void calcCDF(float* in_img , float* out_img, int h, int w, int num_levels);
void negative_transformation(float *in_img, float* out_img, int h, int w, int num_levels);
void log_transformation(float *in_img, float* out_img, int h, int w, int num_levels, float param);
void gamma_transformation(float *in_img, float* out_img, int h, int w, int num_levels, float gamma);
''' 

In all the upcoming classes there will be three functions:<br />

Forward - The froward propogation of the model<br />
Backward - Computing the gradients wrt the computer loss (gradient wrt to input and parameters)<br />
Step - This is to update the learnable parameters <br />


### Convolution

Class definition
'''
class Conv2d
{
public:
	bool is_first; //For Momentum initialization
	float *weight; //channels_out, channels_in, kernel_size, kernel_size
	float bias;
	float* del_weight;
	float* del_vw; //For momentum weight
	int channel_in, channel_out, kernel_size;

	Conv2d(int channel_in, int channel_out, int kernel_size);
	float* forward(float* image, int img_width, int img_height);
	float* backward(float* del_out, float* input, int input_height, int input_width);
	void step(float l_rate, float beeta=0.9);
};
'''

### Fully connected

Class definition
'''
class FC
{
public:

    float *weight;
    float *bias;
    float *dw, *dw_old;
    float *db, *db_old;
    float *d_in, *out;
    int in_size, out_size,first;

    FC(int in_features, int out_features);
    float* forward(float *in);
    float* backward(float *in,float *d_out);
    void step(float lr, float beta);
};
'''
### Relu

Class definition
'''
class ReLU
{
public:
    float *out,*d_in;
    ReLU(int h, int w, int channel);
	float* forward(float *in, int h, int w, int channel);
	float* backward(float* d_out, int h, int w, int channel);
};
'''
### Sigmoid

Class definition
'''
class Sigmoid
{
public:
    float *out,*d_in;
    Sigmoid(int h, int w, int channel);
    float* forward(float *in, int h, int w, int channel);
    float* backward(float* d_out, int h, int w, int channel);
};
'''

### Dropout

Class definition
'''
class Dropout
{
public:
    bool* mask;
    float drop_prob, *d_in;
    int h, w, channel;
    Dropout(float drop_prob, int h, int w, int channel);
    float* forward(float*in);
    float* backward(float* d_out);
};
'''



Add additional notes about how to deploy this on a live system


## Contributing

In case if you find any error it would be great if you can create an issue, and if you solve it too or add some other classes, feel free to send PR.

