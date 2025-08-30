#pragma once
#include "layer.h"
#include <random>
#include <thread>
#include <mutex>
#include <iomanip>

using namespace std;

class Network
{
public:

	//Deep Learning
	
	typedef enum model_mode {
		train,
		eval
	}model_mode;

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
		float threshold = 0.1;
		int patience = 5;
		float final_lr = 0;
		float min_lr = 0;
		float gamma = 0.5;
		vector<int> milestones = {};
		int iterations = 0;
		int step = 5;

		float lineardiff = 0;
	}lr_scheduler;

	typedef enum regularizer_type {
		No_Regularizer,
		L1,
		L2,
		Elastic_Net
	}regularizer_type;

	typedef struct regularizer {
		regularizer_type type = No_Regularizer;
		float L1_Lambda = 0.01;
		float L2_Lambda = 0.1;
		float Elastic_Net_Alpha = 0.5;
	}regularizer;

	typedef enum gradtype {
		Stochastic,
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

	model_mode Model_Mode;

	vector<Layer> layers;

	float*** weights;
	float*** delta_weights;
	float** biases;
	float** delta_biases;
	float*** momentum1D;
	float*** rmsp1D;
	vector<vector<float>> errors;
	vector<vector<float>> derrors;


	float lr = 1e-5;
	float momentumbeta = 0.9;
	float rmspropbeta = 0.999;
	float rmspropepsilon = 10e-10;

	int batchsize=0;
	int totalinputsize=0;
	int totalepochs=0;
	int batchnum = 1;

	int threadcounter = 0;
	int batchcounter = 0;
	int epochcounter = 0;
	float epochloss = 0;

	bool verbosetraining = true;
	bool verboseloss = false;
	bool cleanerrors = true;
	bool fullcarryvalues = false;
	bool input1D = false;

	//Mutex
	mutex* thread_counter_mutex = new mutex();
	mutex* delta_mutex = new mutex();
	mutex* conv_mutex = new mutex();

	//Utitlity
	vector<float> OnesMatrix(int num);
	vector<float> ZerosMatrix(int num);

	//Image
	float MatrixAverage(vector<vector<float>>* mat);
	vector<vector<float>> PixelDistances(vector<vector<float>>* mat1, vector<vector<float>>* mat2);
	vector<vector<vector<float>>> SobelEdgeDetection(vector<vector<vector<float>>>* images);
	vector<vector<vector<float>>> PrewittEdgeDetection(vector<vector<vector<float>>>* images);
	vector<vector<vector<float>>> NNInterpolation(vector<vector<vector<float>>>* image, int finalwidth, int finalheight);
	vector<vector<vector<float>>> BilinearInterpolation(vector<vector<vector<float>>>* image, int finalwidth, int finalheight);
	vector<vector<vector<float>>> EmptyUpscale(vector<vector<vector<float>>>* image, int finalwidth, int finalheight);

	//Network
	string GetInitializerName();
	string GetRegularizerName();
	string GetOptimizerName();
	string GetLRSchedulerName();
	void InitializeValueMatrices(int batchsize);
	void InitializePredictedMatrix(vector<vector<float>>* predicted);
	float Activation(float x, int i);
	float DActivation(float x, int i);
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
	float WeightInitialization(int fan_in, int fan_out);
	void Predict(vector<vector<float>>* input, vector<vector<float>>* predicted);
	void Predict(vector<vector<vector<float>>>* input, vector<vector<float>>* predicted);
	void Train(vector<vector<float>>* inputs, vector<vector<float>>* actual, int epochs, string loss);
	void Train(vector<vector<vector<float>>>* inputs, vector<vector<float>>* actual, int epochs, string loss);
	void ShowTrainingStats(vector<vector<float>>* inputs, vector<vector<float>>* actual, int i);
	void ForwardPropogation(int samplenum, vector<vector<float>> sample, vector<float> actualvalue);
	void BackPropogation(int samplenum);
	void ErrorCalculation(int samplenum, vector<float>& actualvalue);
	void AccumulateErrors();
	void UpdateParameters();
	float DError(float predictedvalue, float actualvalue, int neuronnum);
	void CleanErrors();
	void LeakyReluParameters(float i, float a);
	void SetOptimizer(string opt);
	void SetInitializer(string init);
	void FullConvolve2D(vector<vector<float>>& input, vector<vector<float>>& kernel, int stride,vector<vector<float>>& dst);
	void Convolve2D(vector<vector<float>>& input, vector<vector<float>>& kernel, int stride,vector<vector<float>>& dst);
	vector<vector<float>> Dilate2D(vector<vector<float>>* input, int dilation);
	vector<vector<float>> Rotate(vector<vector<float>>* input);
	vector<vector<float>> Zero2DMatrix(int x, int y);
	void AddVectors(vector<vector<float>>& v1, vector<vector<float>>&  v2);
	void UpdateKernel(vector<vector<float>>* v1, vector<vector<float>>* v2, vector<vector<float>>* momentumkernel, vector<vector<float>>* rmspkernel);
	vector<vector<float>> InitializeKernel(int kernelsize, int dilation);
	vector<vector<float>> Relu2D(vector<vector<float>>* input);
	void MaxPooling2D(vector<vector<float>>* input, short int padnum, vector<vector<float>>* outputdest, vector<vector<float>>* chosendest);
	void CleanLayers(int samplenum);
	void DisplayTensor(vector<vector<float>>& tensor);
};