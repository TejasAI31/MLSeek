#pragma once
#include <iostream>
#include <vector>
using namespace std;


class Layer
{
public:

	typedef struct layerparams
	{
		float LeakyReluAlpha = 0.01;
	} layerparams;

	typedef enum layertype {
		Input2D,
		Input,
		Linear,
		Sigmoid,
		Relu,
		LeakyRelu,
		Tanh,
		Softmax,
		Conv,
		Pool2D,
		Dropout,
	} layertype;


	vector<vector<vector<vector<float>>>> kernels;
	vector<vector<vector<vector<float>>>> pre_activation_values2D;
	vector<vector<vector<vector<float>>>> values2D;
	vector<vector<vector<vector<float>>>> values2Dderivative;
	vector<vector<vector<vector<float>>>> deltakernel;
	vector<vector<vector<vector<float>>>> momentum2D;
	vector<vector<vector<vector<float>>>> rmsp2D;


	bool flattenweights = false;

	int stride = 1;
	int dilation = 1;
	int padding = 0;
	int kernelnumber = 0;
	int kernelsize = 0;
	int number = 0;

	vector<float> softmaxsum;
	float dropout = 0.0;
	vector<vector<float>> values;
	vector<vector<float>> pre_activation_values;

	layerparams parameters;
	layertype type;
	std::string neurontype;

	Layer(int num, std::string neuronname);
	Layer(int kernalnumber, int size, std::string layertype);
	Layer(int kernalnumber, int size, int dilation, std::string layertype);
	Layer(int kernalnumber, int size, int dilation, int stridenum, std::string layertype);
	Layer(float drop, std::string layertype);
};