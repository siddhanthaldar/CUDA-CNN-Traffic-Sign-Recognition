#include "../include/header.h"
#include <fstream>
#define NUM_IMAGE 50
#define IMAGE_DIM 32
#define n_classes 43


void parser(float** out_img, int* label){
	ifstream infile;
	infile.open("../dataset.txt");

	preprocessing gray(IMAGE_DIM, IMAGE_DIM, 3);
	float** img = new float*[NUM_IMAGE]; 
	

	for (int i = 0; i<NUM_IMAGE; i++){
		img[i] = new float[IMAGE_DIM*IMAGE_DIM*3];
	}
	for (int i = 0; i<NUM_IMAGE;i++){
		for (int j=0; j<IMAGE_DIM*IMAGE_DIM*3+1;j++){
			if(j == 0){
				infile>>label[i]; 
				// cout<<"The label "<<label[i]<<endl;
			}
			else {
				infile >> img[i][j-1];
				// cout<<img[i][j-1];
			}
		}
		gray.BGR2GRAY(img[i]);
		gray.Histogram_Equalization(gray.gray_img);
		gray.Normalization(gray.hist_img);
		out_img[i] = gray.norm_img;
		// cout<<endl;
	}
	cout<<"Data Loaded"<<endl;

infile.close();

}


int main(void)
{
	float** out_img = new float*[NUM_IMAGE]; 
	for(int i=0; i< NUM_IMAGE; i++){
		out_img[i] = new float[IMAGE_DIM*IMAGE_DIM*1];//this is to store gray scale normalized image
	}
	int* label = new int[NUM_IMAGE];
	parser(out_img, label);
	// for(int i = 0; i<IMAGE_DIM*IMAGE_DIM; i++)
	// 	cout<<out_img[0][i]<<" ";
	// float** images = new float*[n_images];

	// for(int i = 0; i < n_images; i++)
	// {
	// 	images[i] = new float[IMAGE_DIM*IMAGE_DIM*1];
	// 	for(int j = 0; j < IMAGE_DIM*IMAGE_DIM*1; j++)
	// 	{
	// 		images[i][j] = rand()*1.0/(float)RAND_MAX;
	// 		images[i][j] *= (rand()%2) == 0 ? -1 : 1;
	// 		// cout<<img[i]<<' ';
	// 	}
	// }
	// int label = 4;
	float sum_loss = 0;


	//Defining Model
	Conv2d C1(1, 32, 5);
	ReLU R1(IMAGE_DIM, IMAGE_DIM, 32);
	MaxPool M1(IMAGE_DIM, IMAGE_DIM, 32);
	
	Conv2d C2(32, 64, 5);
	ReLU R2(IMAGE_DIM/2, IMAGE_DIM/2, 64);
	MaxPool M2(IMAGE_DIM/2, IMAGE_DIM/2, 64);
	// Dropout D1(0.2, IMAGE_DIM/4, IMAGE_DIM/4, 64);

	Conv2d C3(64, 128, 5);
	ReLU R3(IMAGE_DIM/4, IMAGE_DIM/4, 128);
	MaxPool M3(IMAGE_DIM/4, IMAGE_DIM/4, 128);
	// Dropout D2(0.2, IMAGE_DIM/8, IMAGE_DIM/8, 128);

	FC F1((IMAGE_DIM*IMAGE_DIM/64)*128, 1024);
	ReLU R4(1, 1, 1024);
	// Dropout D3(0.2, 1, 1, 1024);

	FC F2(1024, n_classes);
	Sigmoid S1(1,1,n_classes);

	softmax_cross_entropy_with_logits S;

	for(int epoch = 0; epoch < 100; epoch++)
	{
		float acc = 0;
		float epoch_loss = 0;
		for(int idx = 0; idx < NUM_IMAGE; idx++)
		{
			float* img = out_img[idx];

			float* out_C1 = C1.forward(img, IMAGE_DIM, IMAGE_DIM);
			float* out_R1 = R1.forward(out_C1, IMAGE_DIM, IMAGE_DIM, 32);
			float* out_M1 = M1.forward(out_R1, IMAGE_DIM, IMAGE_DIM, 32);

			float* out_C2 = C2.forward(out_M1, IMAGE_DIM/2, IMAGE_DIM/2);
			float* out_R2 = R2.forward(out_C2, IMAGE_DIM/2, IMAGE_DIM/2, 64);
			float* out_M2 = M2.forward(out_R2, IMAGE_DIM/2, IMAGE_DIM/2, 64);

			float* out_C3 = C3.forward(out_M2, IMAGE_DIM/4, IMAGE_DIM/4);
			float* out_R3 = R3.forward(out_C3, IMAGE_DIM/4, IMAGE_DIM/4, 128);
			float* out_M3 = M3.forward(out_R3, IMAGE_DIM/4, IMAGE_DIM/4, 128);

			float* out_F1 = F1.forward(out_M3);
			float* out_R4 = R4.forward(out_F1, 1, 1, 1024);
			float* out_F2 = F2.forward(out_R4);
			// float* out_S1 = S1.forward(out_F2, 1,1,n_classes); 


			// for(int i = 0; i < n_classes; i++)
			// 	cout<<out_F2[i]<<' ';
			// cout<<endl;


			float* out_S = S.forward(out_F2, label[idx], n_classes);

			float m = -1;
			int ind = -1;
			for(int i = 0; i < n_classes; i++)
				if(m < out_S[i])
				{
					m = out_S[i];
					ind = i;
				}
			if(label[idx] == ind)
				acc += 1;

			float loss = S.loss;
			epoch_loss += S.loss;
			cout<<"Loss : "<<loss<<endl;
			// cout<<"Relu : \n";
			// for(int i = 0; i < n_classes; i++)
			// 	cout<<out_R4[i]<<' ';
			// cout<<endl;

			// cout<<"FC : \n";
			// for(int i = 0; i < n_classes; i++)
			// 	cout<<out_F2[i]<<' ';
			// cout<<endl;

			// cout<<"Sigmoid : \n";
			// for(int i = 0; i < n_classes; i++)
			// 	cout<<out_S[i]<<' ';
			// cout<<endl;


			sum_loss += loss;
			if(idx % 25 == 0){
			cout<<"Epoch : "<<epoch;//<<' '<<"Out : "<<out_S[0]<<' '<<out_S[1]<<endl;
			cout<<" Iteration : "<<idx;
			cout<<" Label: "<<label[idx];
			cout<<"Running Loss : "<<(sum_loss/25)<<endl;
			sum_loss = 0;
			}

			float* del_out = S.backward(out_S, label[idx], n_classes);
			// del_out = S1.backward(del_out, 1, 1, n_classes);

			del_out = F2.backward(out_F1, del_out);
			del_out = R4.backward(del_out, 1, 1, 1024);
			del_out = F1.backward(out_M3, del_out);

			del_out = M3.backward(del_out, IMAGE_DIM/8, IMAGE_DIM/8, 128);
			del_out = R3.backward(del_out, IMAGE_DIM/4, IMAGE_DIM/4, 128);
			del_out = C3.backward(del_out, out_M2, IMAGE_DIM/4, IMAGE_DIM/4);
			// for(int i = 0; i < 10; i++)
			// 	cout<<del_out[i]<<' ';
			// cout<<endl;


			del_out = M2.backward(del_out, IMAGE_DIM/4, IMAGE_DIM/4, 64);
			del_out = R2.backward(del_out, IMAGE_DIM/2, IMAGE_DIM/2, 64);
			del_out = C2.backward(del_out, out_M1, IMAGE_DIM/2, IMAGE_DIM/2);

			del_out = M1.backward(del_out, IMAGE_DIM/2, IMAGE_DIM/2, 32);
			del_out = R1.backward(del_out, IMAGE_DIM, IMAGE_DIM, 32);
			del_out = C1.backward(del_out, img, IMAGE_DIM, IMAGE_DIM);


			F2.step(1e-2, 0.9);
			F1.step(1e-2, 0.9);
			C1.step(1e-2, 0.9);
			C2.step(1e-2, 0.9);
			C3.step(1e-2, 0.9);
		}
		cout<<"\nEpoch : "<<epoch<<", Avg Loss : "<<epoch_loss/NUM_IMAGE<<", Accuracy : "<<acc/NUM_IMAGE<<endl;
	}
}