#include <iostream>
#include "model.h"
#include <fstream>

int testdatasize = 5000;

void CreateMnistDataset(vector<vector<double>>* input, vector<vector<double>>* actual, int i)
{
	if (i > 60000)
	{
		cout << "Only 60000 Instances Are Present" << endl;
		return;
	}
	cout << "LOADING DATASET" << endl;

	ifstream file("../Data/train-images.idx3-ubyte", ios::binary);

	vector<unsigned char> header(16);
	file.read((char*)(header.data()), 16);
	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(28 * 28);
		vector<double> sample;
		file.read((char*)(temp.data()), 28 * 28);
		for (int i = 0; i < 28 * 28; i++)
		{
			sample.push_back((double)temp[i] / (double)255);
		}
		input->push_back(sample);

	}
	file.close();

	ifstream file2("../Data/train-labels.idx1-ubyte");

	vector<unsigned char> header2(8);
	file2.read((char*)(header2.data()), 8);

	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(1);

		file2.read((char*)(temp.data()), 1);
		vector<double> label(10);
		label[(double)temp[0]] = 1;

		actual->push_back(label);
	}
	file2.close();

	cout << "\nDATASET LOADED\n\n";
}

void CreateMnistDataset2D(vector<vector<vector<double>>>* input, vector<vector<double>>* actual, int i)
{
	if (i > 60000)
	{
		cout << "Only 60000 Instances Are Present" << endl;
		return;
	}
	cout << "LOADING DATASET" << endl;

	ifstream file("../Data/train-images.idx3-ubyte", ios::binary);

	vector<unsigned char> header(16);
	file.read((char*)(header.data()), 16);
	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(28 * 28);
		vector<vector<double>> sample2D;
		file.read((char*)(temp.data()), 28 * 28);

		for (int i = 0; i < 28; i++)
		{
			vector<double> row;
			for (int j = 0; j < 28; j++)
			{
				row.push_back((double)(temp[i * 28 + j]) / (double)255);
			}
			sample2D.push_back(row);
		}
		input->push_back(sample2D);
	}
	file.close();

	ifstream file2("../Data/train-labels.idx1-ubyte");

	vector<unsigned char> header2(8);
	file2.read((char*)(header2.data()), 8);

	for (int x = 0; x < i; x++)
	{
		vector<unsigned char> temp(1);

		file2.read((char*)(temp.data()), 1);
		vector<double> label(10);
		label[(double)temp[0]] = 1;

		actual->push_back(label);
	}
	file2.close();

	cout << "DATASET LOADED\n\n";
}

Network CreateClassifier()
{
	Network model;
	model.AddLayer(Layer(28 * 28, "Input"));
	model.AddLayer(Layer(10, "Softmax"));

	model.lr = 1e-3;
	model.SetOptimizer("Adam");

	model.Compile("Mini B",1);
	model.Summary();

	return model;
}

int main()
{
	vector<vector<double>> input;
	vector<vector<double>> actual;
	vector<vector<double>> predictions;
	CreateMnistDataset(&input, &actual, 5000);

	//Training
	Network model = CreateClassifier();
	model.Train(&input, &actual, 3, "CCE");

	vector<vector<double>> testdata;
	for (int i = 0; i < testdatasize; i++)
	{
		testdata.push_back(input[i]);
	}

	cout << "Calculating Accuracy:" << endl;

	//Predictions
	model.Predict(&testdata, &predictions);

	//Accuracy
	int counter = 0;
	for (int i = 0; i < predictions.size(); i++)
	{
		double pred = 0;
		for (int j = 0; j < 10; j++)
		{
			if (predictions[i][j] > predictions[i][pred])
				pred = j;
		}

		double act = 0;
		for (int j = 0; j < 10; j++)
		{
			if (actual[i][j] == 1)
			{
				act = j;
				break;
			}
		}

		if (act == pred)
			counter++;
	}
	
	cout << "Accuracy: " << counter * 100 / (double)testdatasize;
}