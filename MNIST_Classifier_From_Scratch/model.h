#pragma once
#include "layer.h"
#include <random>
#include <thread>

using namespace std;

class Network
{
public:

	//Deep Learning

	typedef enum scheduler_type {
		No_Scheduler,
		Step_LR,
		Multi_Step_LR,
		Constant_LR,
		Linear_LR,
		Exponential_LR,
		Reduce_LR_On_Plateau,

	}scheduler_type;

	typedef struct lr_scheduler {
		scheduler_type type = No_Scheduler;
		string mode = "min";
		double threshold = 0.1;
		int patience = 5;
		double final_lr = 0;
		double min_lr = 0;
		double gamma = 0.5;
		vector<int> milestones = {};
		int iterations = 0;
		int step = 5;

		double lineardiff = 0;
	}lr_scheduler;

	typedef enum regularizer_type {
		No_Regularizer,
		L1,
		L2,
		Elastic_Net
	}regularizer_type;

	typedef struct regularizer {
		regularizer_type type = No_Regularizer;
		double L1_Lambda = 0.01;
		double L2_Lambda = 0.1;
		double Elastic_Net_Alpha = 0.5;
	}regularizer;

	typedef enum gradtype {
		Stochastic,
		Batch,
		Mini_Batch,
	}gradtype;

	typedef enum losstype {
		Mean_Squared,
		Mean_Absolute,
		Mean_Biased,
		Root_Mean_Squared,
		Binary_Cross_Entropy,
		Categorical_Cross_Entropy
	}losstype;

	typedef enum optimizer {
		No_Optimizer,
		Momentum,
		RMSProp,
		Adam,
	}optimizer;

	typedef enum weightinitializer {
		Glorot,
		He,
		Random
	}weightinitializer;

	typedef enum showparams {
		Visual,
		Text
	}showparams;

	gradtype gradient_descent_type;
	losstype model_loss_type;
	showparams displayparameters;

	lr_scheduler LR_Scheduler;
	regularizer Regularizer;
	optimizer Optimizer;
	weightinitializer WeightInitializer;

	vector<Layer> layers;

	double*** weights;
	double** biases;
	double*** momentum1D;
	double*** rmsp1D;
	vector<double> errors;
	vector<double> derrors;


	double lr = 1e-5;
	double momentumbeta = 0.9;
	double rmspropbeta = 0.999;
	double rmspropepsilon = 10e-10;

	int batchsize;
	long long int totalinputsize;
	int totalepochs;
	int batchnum = 1;

	int threadcounter = 0;
	int batchcounter = 0;
	int epochcounter = 0;
	double epochloss = 0;

	bool verbosetraining = true;
	bool verboseloss = false;
	bool cleanerrors = true;
	bool fullcarryvalues = false;
	bool input1D = false;

	//Utitlity
	vector<double> OnesMatrix(int num);
	vector<double> ZerosMatrix(int num);

	//Image
	double MatrixAverage(vector<vector<double>>* mat);
	vector<vector<double>> PixelDistances(vector<vector<double>>* mat1, vector<vector<double>>* mat2);
	vector<vector<vector<double>>> SobelEdgeDetection(vector<vector<vector<double>>>* images);
	vector<vector<vector<double>>> PrewittEdgeDetection(vector<vector<vector<double>>>* images);
	vector<vector<vector<double>>> NNInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight);
	vector<vector<vector<double>>> BilinearInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight);
	vector<vector<vector<double>>> EmptyUpscale(vector<vector<vector<double>>>* image, int finalwidth, int finalheight);

	//Network
	string GetInitializerName();
	string GetRegularizerName();
	string GetOptimizerName();
	string GetLRSchedulerName();
	void InitializeValueMatrices(int batchsize);
	void InitializePredictedMatrix(vector<vector<double>>* predicted);
	double Activation(double x, int i);
	double DActivation(double x, int i);
	void AddLayer(Layer l);
	void SetDisplayParameters(string s);
	void SetLRScheduler(string s);
	void SetRegularizer(string s);
	void UpdateLearningRate(int epoch);
	void Summary();
	void PrintParameters();
	void Compile(string type, int batch_size);
	void Compile(string type);
	void Initialize();
	double WeightInitialization(int fan_in, int fan_out);
	void Predict(vector<vector<double>>* input, vector<vector<double>>* predicted);
	void Predict(vector<vector<vector<double>>>* input, vector<vector<double>>* predicted);
	void Train(vector<vector<double>>* inputs, vector<vector<double>>* actual, int epochs, string loss);
	void Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, int epochs, string loss);
	void ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual, int i);
	void ForwardPropogation(int samplenum, vector<vector<double>> sample, vector<double> actualvalue);
	void ErrorCalculation(int samplenum, vector<double> actualvalue);
	void AccumulateErrors();
	double DError(double predictedvalue, double actualvalue, int neuronnum);
	void CleanErrors();
	void BackPropogation();
	void LeakyReluParameters(double i, double a);
	void SetOptimizer(string opt);
	void SetInitializer(string init);
	vector<vector<double>> FullConvolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel, int stride);
	vector<vector<double>> Convolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel, int stride);
	vector<vector<double>> Dilate2D(vector<vector<double>>* input, int dilation);
	vector<vector<double>> Rotate(vector<vector<double>>* input);
	vector<vector<double>> Zero2DMatrix(int x, int y);
	void AddVectors(vector<vector<double>>* v1, vector<vector<double>>* v2);
	void UpdateKernel(vector<vector<double>>* v1, vector<vector<double>>* v2, vector<vector<double>>* momentumkernel, vector<vector<double>>* rmspkernel);
	vector<vector<double>> InitializeKernel(int kernelsize, int dilation);
	vector<vector<double>> Relu2D(vector<vector<double>>* input);
	void MaxPooling2D(vector<vector<double>>* input, short int padnum, vector<vector<double>>* outputdest, vector<vector<double>>* chosendest);
	void CleanLayers();
};