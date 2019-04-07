#include "../include/header.h"

int main(void)
{
	cout<<"Convolution and update"<<endl;
	int input_size = 6;

	float* img = new float[input_size*input_size*3];
	for(int i = 0; i < input_size*input_size*3; i++)
	{
		img[i] = rand()*1.0/(float)RAND_MAX;
		img[i] *= (rand()%2) == 0 ? -1 : 1;
		// cout<<img[i]<<' ';
	}

	// cout<<endl;

	Conv2d C1(3, 3, 3);
	Conv2d C2(3, 1, 3);

	ReLU R1(input_size, input_size, 3);
	FC F1(input_size*input_size/4, 12);
	FC F2(12, 2);
	max_pool M1(input_size, input_size, 3);	
	softmax_and_loss S;

	for(int epoch = 0; epoch < 100; epoch++)
	{
		float* out_C1 = C1.forward(img, input_size, input_size);
		// for(int i = 0; i < 3*3*3*3; i++)
		// 	cout<<C1.weight[i]<<' ';

		float* out_R1 = R1.forward(out_C1, input_size, input_size, 3);

		float* out_M1 = M1.forward(out_R1, input_size, input_size, 3);
		// for(int i = 0; i < (input_size*input_size*3)/4; i++)
		// 	cout<<out_M1[i]<<' ';
		float* out_C2 = C2.forward(out_M1, input_size/2, input_size/2);

		float* out_F1 = F1.forward(out_C2);

		float* out_F2 = F2.forward(out_F1);

		float* out_S = S.forward(out_F2, 1, 2);
		float loss = S.loss;
		cout<<"Epoch : "<<epoch<<' '<<"Out : "<<out_S[0]<<' '<<out_S[1]<<endl;
		cout<<"Loss : "<<loss<<endl;


		float* del_out = S.backward(out_S, 1, 2);
		del_out = F2.backward(out_F1, del_out);
		del_out = F1.backward(out_C2, del_out);
		del_out = C2.backward(del_out, out_R1, input_size/2, input_size/2);
		del_out = M1.backward(del_out, input_size/2, input_size/2, 3);
		del_out = R1.backward(del_out, input_size, input_size, 3);
		del_out = C1.backward(del_out, img, input_size, 3);

		// for(int i = 0; i < 2; i++)
		// 	cout<<F2.db[i]<<' ';
		// cout<<endl;

		F2.step(1e-3, 0.9);
		F1.step(1e-3, 0.9);
		C1.step(1e-3, 0.9);
		C2.step(1e-3, 0.9);
	}


	// float* out = C1.forward(img, input_size, input_size);
	// // cout<<"Conv1 output : \n";
	// // for(int i = 0; i < input_size*input_size*3; i++)
	// // 	cout<<out[i]<<' ';
	// // cout<<endl;

	// R1.forward(out, input_size, input_size, 3);
	// out = R1.out;
	// // cout<<"Relu Output\n";
	// // for(int i = 0; i < input_size*input_size*3; i++)
	// // 	cout<<out[i]<<' ';
	// // cout<<endl;

	// Conv2d C2(3, 1, 3);
	// out = C2.forward(out, input_size, input_size);
	// // cout<<"Conv2 output : \n";
	// // for(int i = 0; i < input_size*input_size; i++)
	// // 	cout<<out[i]<<' ';
	// // cout<<endl;


	// F1.forward(out);
	// out = F1.out;
	// // cout<<"FC1 output\n";
	// // for(int i = 0; i < 12; i++)
	// // 	cout<<out[i]<<' ';
	// // cout<<endl;


	// F2.forward(out);
	// out = F2.out;

	// out = S.forward(out, 1, 2);

	// cout<<"Final output\n";
	// cout<<out[0]<<' '<<out[1]<<endl;

	// cout<<"Loss : "<<S.loss;


}