#include "model.h"

using namespace std;

//Utility
vector<double> Network::OnesMatrix(int num)
{
	vector<double> out(num, 1);
	return out;
}

vector<double> Network::ZerosMatrix(int num)
{
	vector<double> out(num, 0);
	return out;
}

//Image Transformations
double Network::MatrixAverage(vector<vector<double>>* mat)
{
	double sum = 0;
	for (int x = 0; x < mat->size(); x++)
	{
		for (int y = 0; y < (*mat)[0].size(); y++)
		{
			sum += (*mat)[x][y];
		}
	}
	return sum / (double)(mat->size() * (*mat)[0].size());
}

vector<vector<double>> Network::PixelDistances(vector<vector<double>>* mat1, vector<vector<double>>* mat2)
{
	vector<vector<double>> distancemat;
	if (mat1->size() != mat2->size() || mat1->empty())
		return {};

	for (int x = 0; x < mat1->size(); x++)
	{
		vector<double> row;
		for (int y = 0; y < (*mat1)[0].size(); y++)
		{
			row.push_back(sqrt(pow((*mat1)[x][y], 2) + pow((*mat2)[x][y], 2)));
		}
		distancemat.push_back(row);
	}

	return distancemat;
}

vector<vector<vector<double>>> Network::EmptyUpscale(vector<vector<vector<double>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<double>>> finalimage;

	for (int i = 0; i < image->size(); i++)
	{
		vector<vector<double>> upscaled;
		int rowsize = (*image)[i][0].size();
		int columnsize = (*image)[i].size();
		int rowstep = floor(finalheight / columnsize);
		int columnstep = floor(finalwidth / rowsize);

		int ydeficit = finalheight - columnsize;
		int ycounter = 0;
		for (int x = 0; x < finalheight; x++)
		{
			//Last Column
			if (ycounter == columnsize - 1)
			{
				for (int y = 0; y < finalheight - x - 1; y++)
				{
					vector<double> empty(finalwidth, 123.456);
					upscaled.push_back(empty);
				}
				break;
			}

			//Enter Full Row
			vector<double> row(finalwidth, 123.456);
			int xdeficit = finalwidth - rowsize;
			int xcounter = 0;
			for (int y = 0; y < finalwidth; y++)
			{
				if (xcounter == rowsize - 1)break;
				row[y] = (*image)[i][ycounter][xcounter];
				if (xdeficit > 0)y += columnstep;
				xdeficit--;
				xcounter++;
			}
			row[finalwidth - 1] = (*image)[i][ycounter][rowsize - 1];
			upscaled.push_back(row);

			//Empty Row
			if (ydeficit > 0)
			{
				for (int z = 0; z < rowstep; z++)
				{
					vector<double> empty(finalwidth, 123.456);
					upscaled.push_back(empty);
				}
				x += rowstep;
			}
			ydeficit--;
			ycounter++;
		}

		//Final Row
		vector<double> row(finalwidth, 123.456);
		int xdeficit = finalwidth - rowsize;
		int xcounter = 0;
		for (int y = 0; y < finalwidth; y++)
		{
			if (xcounter == rowsize - 1)break;
			row[y] = (*image)[i][columnsize - 1][xcounter];
			if (xdeficit > 0)y++;
			xdeficit--;
			xcounter++;
		}
		row[finalwidth - 1] = (*image)[i][ycounter][rowsize - 1];
		upscaled.push_back(row);

		//Send Image
		finalimage.push_back(upscaled);
	}
	return finalimage;
}

vector<vector<vector<double>>> Network::SobelEdgeDetection(vector<vector<vector<double>>>* image)
{
	static vector<vector<double>> sobelx = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};

	static vector<vector<double>> sobely = {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1}
	};

	vector<vector<vector<double>>> edgeimages;

	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<double>> xedges = Convolve2D(&((*image)[x]), &sobelx, 1);
		vector<vector<double>> yedges = Convolve2D(&((*image)[x]), &sobely, 1);
		vector<vector<double>> magnitude = PixelDistances(&xedges, &yedges);

		double threshold = MatrixAverage(&magnitude);

		for (int x = 0; x < magnitude.size(); x++)
		{
			for (int y = 0; y < magnitude[0].size(); y++)
			{
				magnitude[x][y] = (magnitude[x][y] < threshold) ? 0 : 1;
			}
		}

		edgeimages.push_back(magnitude);
	}
	return edgeimages;
}

vector<vector<vector<double>>> Network::PrewittEdgeDetection(vector<vector<vector<double>>>* image)
{
	static vector<vector<double>> prewittx = {
		{-1,0,1},
		{-1,0,1},
		{-1,0,1}
	};

	static vector<vector<double>> prewitty = {
		{1,1,1},
		{0,0,0},
		{-1,-1,-1}
	};

	vector<vector<vector<double>>> edgeimages;

	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<double>> xedges = Convolve2D(&((*image)[x]), &prewittx, 1);
		vector<vector<double>> yedges = Convolve2D(&((*image)[x]), &prewitty, 1);
		vector<vector<double>> magnitude = PixelDistances(&xedges, &yedges);

		double threshold = MatrixAverage(&magnitude);

		for (int x = 0; x < magnitude.size(); x++)
		{
			for (int y = 0; y < magnitude[0].size(); y++)
			{
				magnitude[x][y] = (magnitude[x][y] < threshold) ? 0 : 1;
			}
		}

		edgeimages.push_back(magnitude);
	}
	return edgeimages;
}

vector<vector<vector<double>>> Network::NNInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<double>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			//Empty Row
			if (emptyupscaled[i][x][0] == 123.456)continue;

			double value = emptyupscaled[i][x][0];
			for (int y = 1; y < emptyupscaled[i][x].size(); y++)
			{
				if (emptyupscaled[i][x][y] == 123.456)
					emptyupscaled[i][x][y] = value;
				else
					value = emptyupscaled[i][x][y];
			}
		}

		//Column Interpolation
		for (int x = 0; x < emptyupscaled[i][0].size(); x++)
		{
			double value = emptyupscaled[i][0][x];
			for (int y = 1; y < emptyupscaled[i].size(); y++)
			{
				if (emptyupscaled[i][y][x] == 123.456)
					emptyupscaled[i][y][x] = value;
				else
					value = emptyupscaled[i][y][x];
			}
		}
	}
	return emptyupscaled;
}

vector<vector<vector<double>>> Network::BilinearInterpolation(vector<vector<vector<double>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<double>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			double value = emptyupscaled[i][x][0];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i][x].size(); y++)
			{
				if (emptyupscaled[i][x][y] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][x][valuecount + z] = value * (denom - z) / (double)denom + emptyupscaled[i][x][y] * z / (double)denom;
					}
					value = emptyupscaled[i][x][y];
					valuecount = y;
				}
			}
		}

		//Column Interpolation
		for (int x = 0; x < emptyupscaled[i][0].size(); x++)
		{
			double value = emptyupscaled[i][0][x];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i].size(); y++)
			{
				if (emptyupscaled[i][y][x] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][valuecount + z][x] = value * (denom - z) / (double)denom + emptyupscaled[i][y][x] * z / (double)denom;
					}
					value = emptyupscaled[i][y][x];
					valuecount = y;
				}
			}
		}
	}
	return emptyupscaled;
}

//Convolution Functions
vector<vector<double>> Network::Convolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel, int stride)
{
	short int kernelsize = kernel->size();
	short int columns = input->size();
	short int rows = (*input)[0].size();

	vector<vector<double>> output;

	for (int y = 0; y <= columns - columns % kernelsize; y += stride)
	{
		vector<double> row;
		for (int x = 0; x <= rows - rows % kernelsize; x += stride)
		{
			double value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (y + i >= columns || x + j >= rows)
						continue;
					else
						value += (*kernel)[i][j] * (*input)[y + i][x + j];
				}
			}
			row.push_back(value);
		}
		output.push_back(row);
	}

	return output;
}

vector<vector<double>> Network::FullConvolve2D(vector<vector<double>>* input, vector<vector<double>>* kernel, int stride)
{
	short int kernelsize = kernel->size();
	short int columns = input->size();
	short int rows = (*input)[0].size();
	short int rowpadding = kernelsize - rows % kernelsize;
	short int columnpadding = kernelsize - columns % kernelsize;

	vector<vector<double>> output;

	for (int y = 1 - columns; y < 2 * columns - kernelsize + columnpadding; y += stride)
	{
		vector<double> row;
		for (int x = 1 - rows; x < 2 * rows - kernelsize + rowpadding; x += stride)
		{
			double value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (y + i >= columns || x + j >= rows || y + i < 0 || x + j < 0)
						continue;
					else
						value += (*kernel)[i][j] * (*input)[y + i][x + j];
				}
			}
			row.push_back(value);
		}
		output.push_back(row);
	}

	return output;
}

vector<vector<double>> Network::Relu2D(vector<vector<double>>* input)
{
	vector<vector<double>> output;
	for (int i = 0; i < input->size(); i++)
	{
		vector<double> row;
		for (int j = 0; j < (*input)[0].size(); j++)
		{
			row.push_back((*input)[i][j] > 0 ? (*input)[i][j] : 0);
		}
		output.push_back(row);
	}
	return output;
}

vector<vector<double>> Network::Rotate(vector<vector<double>>* input)
{
	vector<vector<double>> rotated;
	for (int i = input->size() - 1; i >= 0; i--)
	{
		vector<double> row;
		for (int j = (*input)[i].size() - 1; j >= 0; j--)
		{
			row.push_back((*input)[i][j]);
		}
		rotated.push_back(row);
	}
	return rotated;
}

void Network::AddVectors(vector<vector<double>>* v1, vector<vector<double>>* v2)
{
	if (v1->size() == 0)
	{
		(*v1) = (*v2);
		return;
	}
	else
	{
		for (int i = 0; i < v1->size(); i++)
		{
			for (int j = 0; j < (*v1)[0].size(); j++)
			{
				(*v1)[i][j] += (*v2)[i][j];
			}
		}
	}
}

void Network::MaxPooling2D(vector<vector<double>>* input, short int padnum, vector<vector<double>>* outputdest, vector<vector<double>>* chosendest)
{
	int columns = input->size();
	int rows = (*input)[0].size();
	short int rowpadding = padnum - rows % padnum;
	short int columnpadding = padnum - columns % padnum;

	vector<vector<double>> output;
	vector<vector<double>> chosenvalues(columns, vector<double>(rows));


	for (int y = 0; y <= columns + columnpadding; y += padnum)
	{
		vector<double> row;
		for (int x = 0; x <= rows + rowpadding; x += padnum)
		{
			short int chosenx = 0;
			short int choseny = 0;
			double maxval = 0;
			for (int i = 0; i < padnum; i++)
			{
				for (int j = 0; j < padnum; j++)
				{
					if (y + i >= columns || x + j >= rows)
						continue;
					else if ((*input)[y + i][x + j] > maxval)
					{
						maxval = (*input)[y + i][x + j];
						chosenx = x + j;
						choseny = y + i;
					}
				}
			}
			chosenvalues[choseny][chosenx] = 1;
			row.push_back(maxval);
		}
		output.push_back(row);
	}

	*outputdest = output;
	*chosendest = chosenvalues;
}

void Network::UpdateKernel(vector<vector<double>>* v1, vector<vector<double>>* v2, vector<vector<double>>* momentumkernel, vector<vector<double>>* rmspkernel)
{
	for (int i = 0; i < v1->size(); i++)
	{
		for (int j = 0; j < (*v1)[0].size(); j++)
		{
			switch (Optimizer)
			{
			case No_Optimizer:
				(*v1)[i][j] += (*v2)[i][j] * lr;
				break;
			case Momentum:
				(*momentumkernel)[i][j] = momentumbeta * (*momentumkernel)[i][j] + (1 - momentumbeta) * (*v2)[i][j];
				(*v1)[i][j] += (*momentumkernel)[i][j] * lr;
				break;
			case RMSProp:
				(*rmspkernel)[i][j] = rmspropbeta * (*rmspkernel)[i][j] + (1 - momentumbeta) * pow((*v2)[i][j], 2);
				(*v1)[i][j] += (*v2)[i][j] / sqrt((*rmspkernel)[i][j] + rmspropepsilon) * lr;
				break;
			case Adam:
				(*momentumkernel)[i][j] = momentumbeta * (*momentumkernel)[i][j] + (1 - momentumbeta) * (*v2)[i][j];
				(*rmspkernel)[i][j] = rmspropbeta * (*rmspkernel)[i][j] + (1 - momentumbeta) * pow((*v2)[i][j], 2);
				(*v1)[i][j] += (*momentumkernel)[i][j] / sqrt((*rmspkernel)[i][j] + rmspropepsilon) * lr;
				break;
			}
		}
	}
	return;
}

vector<vector<double>> Network::Dilate2D(vector<vector<double>>* input, int dilation)
{
	vector<vector<double>> output;

	int rows = input->size();
	int cols = (*input)[0].size();

	int rowpos = 0;
	int colpos = 0;

	for (int j = 0; j < rows + (dilation - 1) * (rows - 1); j++)
	{
		vector<double> row;
		for (int i = 0; i < cols + (dilation - 1) * (cols - 1); i++)
		{
			if (i % dilation == 0 && j % dilation == 0)
			{
				row.push_back((*input)[rowpos][colpos]);
				colpos++;
			}
			else
				row.push_back(0);
		}
		colpos = 0;
		if (j % dilation == 0)
			rowpos++;
		output.push_back(row);
	}
	return output;
}

vector<vector<double>> Network::Zero2DMatrix(int x, int y)
{
	vector<vector<double>> mat;
	for (int ay = 0; ay < y; ay++)
	{
		vector<double> zerorow(x);
		mat.push_back(zerorow);
	}
	return mat;
}

vector<vector<double>> Network::InitializeKernel(int kernelsize, int dilation)
{
	vector<vector<double>> kernel;

	for (int j = 0; j < kernelsize + (dilation - 1) * (kernelsize - 1); j++)
	{
		vector<double> row;
		for (int k = 0; k < kernelsize + (dilation - 1) * (kernelsize - 1); k++)
		{
			if (k % dilation == 0 && j % dilation == 0)
				row.push_back((float)rand() / (float)RAND_MAX);
			else
				row.push_back(0);
		}
		kernel.push_back(row);
	}
	return kernel;
}

void Network::ForwardPropogation(int samplenum, vector<vector<double>> sample, vector<double> actualvalue)
{

	//Check Validity of input layer
	if (layers[0].type != Layer::Input2D && layers[0].type != Layer::Input)
	{
		cerr << "\n\nLayer 0 is not Input\n\n";
		return;
	}

	//Insert Sample Into Input Layer
	short int convend = 1;
	switch (layers[0].type)
	{
	case Layer::Input2D:
		layers[0].values2D[samplenum][0] = sample;
		break;

	case Layer::Input:
		//Shape Mismatch
		if (sample[0].size() != layers[0].number)
		{
			cerr << "\n\nInput Size " << layers[0].number << " Does Not Match Sample Size " << sample[0].size() << endl;
			return;
		}

		for (int i = 0; i < sample[0].size(); i++)
		{
			layers[0].values[samplenum][i] = sample[0][i];
		}
	}


	//Calculate Convolutions
	for (int i = 1; i < layers.size(); i++)
	{
		if (layers[0].type == Layer::Input)
			break;

		if (layers[i].type == Layer::Conv)
		{
			//For all previous convolutions
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				for (int k = 0; k < layers[i - 1].kernelnumber; k++)
				{
					vector<vector<double>> convolution = Convolve2D(&layers[i - 1].values2D[samplenum][k], &layers[i].kernels[j][k], layers[i].stride);
					AddVectors(&layers[i].pre_activation_values2D[samplenum][j], &convolution);
				}
				layers[i].values2D[samplenum][j] = Relu2D(&layers[i].pre_activation_values2D[samplenum][j]);
			}
		}
		else if (layers[i].type == Layer::Pool2D)
		{
			//For all previous convolutions
			for (int j = 0; j < layers[i - 1].kernelnumber; j++)
			{
				MaxPooling2D(&layers[i - 1].values2D[samplenum][j], layers[i].padding, &layers[i].values2D[samplenum][j], &layers[i].pre_activation_values2D[samplenum][j]);
			}
		}

		else
		{
			//FLATTEN
			if (layers[i - 1].values[samplenum].size() == 0)
			{
				vector<double> temp(layers[i - 1].values2D[samplenum].size() * layers[i - 1].values2D[samplenum][0].size() * layers[i - 1].values2D[samplenum][0][0].size());
				layers[i - 1].values[samplenum] = temp;
				layers[i - 1].number = layers[i - 1].values2D[samplenum].size() * layers[i - 1].values2D[samplenum][0].size() * layers[i - 1].values2D[samplenum][0][0].size();
			}

			unsigned long long int counter = 0;
			for (int j = 0; j < layers[i - 1].values2D[samplenum].size(); j++)
			{
				for (int k = 0; k < layers[i - 1].values2D[samplenum][j].size(); k++)
				{
					for (int l = 0; l < layers[i - 1].values2D[samplenum][j][k].size(); l++)
					{
						layers[i - 1].values[samplenum][counter] = layers[i - 1].values2D[samplenum][j][k][l];
						counter++;
					}
				}
			}

			//Weight Initialization
			if (!layers[i - 1].flattenweights)
			{
				weights[i - 1] = (double**)malloc(sizeof(double*) * layers[i - 1].number);
				momentum1D[i - 1] = (double**)malloc(sizeof(double*) * layers[i - 1].number);
				rmsp1D[i - 1] = (double**)malloc(sizeof(double*) * layers[i - 1].number);

				for (int j = 0; j < layers[i - 1].number; j++)
				{
					weights[i - 1][j] = (double*)malloc(sizeof(double) * layers[i].number);
					momentum1D[i - 1][j] = (double*)malloc(sizeof(double) * layers[i].number);
					rmsp1D[i - 1][j] = (double*)malloc(sizeof(double) * layers[i].number);

					for (int k = 0; k < layers[i].number; k++)
					{
						weights[i - 1][j][k] = (double)rand() / (double)RAND_MAX;
						momentum1D[i - 1][j][k] = 0.0;
						rmsp1D[i - 1][j][k] = 0.0;
					}
				}
				layers[i - 1].flattenweights = true;
			}

			convend = i;
			break;
		}
	}

	//Calculate Forward Prop
	for (int i = convend; i < layers.size(); i++)
	{
		//Check For Dropout
		if (layers[i].type == Layer::Dropout)
		{
			double multrate = 1 / (double)(1 - layers[i].dropout);
			int totaloff = layers[i].dropout * layers[i].number;
			int off = 0;

			//Initialization
			for (int j = 0; j < layers[i].number; j++)
			{
				layers[i].values[samplenum][j] = layers[i - 1].values[samplenum][j];
				if (layers[i].values[samplenum][j] == 0)layers[i].values[samplenum][j] = 0.0001;
			}
			//Turn off neurons
			if (totaloff > 0)
			{
				bool flag = 0;
				while (!flag)
					for (int j = 0; j < layers[i].number; j++)
					{
						if (rand() / (double)RAND_MAX < layers[i].dropout && layers[i].values[samplenum][j] != 0)
						{
							layers[i].values[samplenum][j] = 0;
							if (++off > totaloff)
							{
								flag = 1;
								break;
							}
						}
					}
			}
			//Scale Values
			for (int j = 0; j < layers[i].number; j++)
			{
				if (layers[i].values[samplenum][j] == 0.0001)layers[i].values[samplenum][j] = 0;
				if (layers[i].values[samplenum][j] != 0)layers[i].values[samplenum][j] *= multrate;
			}
		}

		else if (layers[i].type == Layer::Softmax)
		{
			//Calculate Pre Activation Values
			for (int j = 0; j < layers[i].number; j++)
			{
				double sum = 0;
				for (int k = 0; k < layers[i - 1].number; k++)
				{
					sum += weights[i - 1][k][j] * layers[i - 1].values[samplenum][k];
				}
				layers[i].pre_activation_values[samplenum][j] = sum;
			}

			//Calculate Softsum
			double softsum = 0;
			for (int j = 0; j < layers[i].number; j++)
				softsum += exp(layers[i].pre_activation_values[samplenum][j]);
			layers[i].softmaxsum[samplenum] = softsum;

			//Calculate Activation Values
			for (int j = 0; j < layers[i].number; j++)
				layers[i].values[samplenum][j] = Activation(layers[i].pre_activation_values[samplenum][j], i);
		}

		//Sigmoid,Tanh,Relu,etc Cases
		else
		{
			for (int j = 0; j < layers[i].number; j++)
			{
				double sum = 0;
				for (int k = 0; k < layers[i - 1].number; k++)
				{
					sum += weights[i - 1][k][j] * layers[i - 1].values[samplenum][k];
				}
				sum += biases[i][j];

				layers[i].pre_activation_values[samplenum][j] = sum;
				layers[i].values[samplenum][j] = Activation(sum, i);
			}
		}
	}

	//Calculate Error
	ErrorCalculation(samplenum, actualvalue);

	//Increment ThreadCounter
	threadcounter++;
	return;
}

void Network::BackPropogation()
{
	//Initial Error
	short int final_layer = layers.size() - 1;

	//Shift errors to values
	layers.back().values[0] = derrors;

	int convstart = 0;
	for (int i = final_layer; i > 0; i--)
	{
		//Check for Convolution Start
		if (layers[i].type == Layer::Conv || layers[i].type == Layer::Pool2D)
		{
			convstart = i;
			//Calculate derivate
			for (long long unsigned int j = 0; j < layers[i].number; j++)
			{
				long long int sum = 0;
				for (int k = 0; k < layers[i + 1].number; k++)
				{
					sum += layers[i + 1].values[0][k] * weights[i][j][k];
				}
				layers[i].values[0][j] = sum;
			}
			break;
		}

		//Derivative
		for (int j = 0; j < layers[i].number; j++)
		{
			//Dropout Case
			if (layers[i].type == Layer::Dropout)
			{
				//Off Neuron
				if (layers[i].values[0][j] == 0)
				{
					layers[i - 1].values[0][j] = 0;
					continue;
				}

				double sum = 0;
				for (int k = 0; k < layers[i + 1].number; k++)
				{
					sum += layers[i + 1].values[0][k] * weights[i][j][k];
				}

				layers[i].values[0][j] = DActivation(layers[i].pre_activation_values[0][j], i) * sum;
				layers[i - 1].values[0][j] = layers[i].values[0][j];

			}

			else
			{
				//Calculate Delta Values 
				if (i < final_layer)
					if (layers[i + 1].type != Layer::Dropout)
					{
						double sum = 0;
						for (int k = 0; k < layers[i + 1].number; k++)
						{
							sum += layers[i + 1].values[0][k] * weights[i][j][k];
						}
						layers[i].values[0][j] = DActivation(layers[i].pre_activation_values[0][j], i) * sum;
					}


				for (int k = 0; k < layers[i - 1].number; k++)
				{

					double prevweight = weights[i - 1][k][j];

					//Weight Updates
					switch (Optimizer)
					{
					case No_Optimizer:
						weights[i - 1][k][j] += lr * layers[i].values[0][j] * layers[i - 1].values[0][k];
						break;
					case Momentum:
						momentum1D[i - 1][k][j] = momentumbeta * momentum1D[i - 1][k][j] + (1 - momentumbeta) * (layers[i].values[0][j] * layers[i - 1].values[0][k]);
						weights[i - 1][k][j] += lr * momentum1D[i - 1][k][j];
						break;
					case RMSProp:
						rmsp1D[i - 1][k][j] = rmspropbeta * rmsp1D[i - 1][k][j] + (1 - rmspropbeta) * pow((layers[i].values[0][j] * layers[i - 1].values[0][k]), 2);
						weights[i - 1][k][j] += lr * (layers[i].values[0][j] * layers[i - 1].values[0][k]) / (sqrt(rmsp1D[i - 1][k][j]) + rmspropepsilon);
						break;
					case Adam:
						momentum1D[i - 1][k][j] = momentumbeta * momentum1D[i - 1][k][j] + (1 - momentumbeta) * (layers[i].values[0][j] * layers[i - 1].values[0][k]);
						rmsp1D[i - 1][k][j] = rmspropbeta * rmsp1D[i - 1][k][j] + (1 - rmspropbeta) * pow((layers[i].values[0][j] * layers[i - 1].values[0][k]), 2);
						weights[i - 1][k][j] += lr * momentum1D[i - 1][k][j] / (sqrt(rmsp1D[i - 1][k][j]) + rmspropepsilon);
						break;
					}

					switch (Regularizer.type)
					{
					case L1:
						weights[i - 1][k][j] += (prevweight < 0) ? lr * Regularizer.L1_Lambda : lr * (-Regularizer.L1_Lambda);
						break;
					case L2:
						weights[i - 1][k][j] -= lr * Regularizer.L2_Lambda * prevweight;
						break;
					case Elastic_Net:
						weights[i - 1][k][j] += (prevweight < 0) ? lr * Regularizer.L1_Lambda * Regularizer.Elastic_Net_Alpha : lr * (-Regularizer.L1_Lambda) * Regularizer.Elastic_Net_Alpha;
						weights[i - 1][k][j] -= lr * Regularizer.L2_Lambda * (1 - Regularizer.Elastic_Net_Alpha) * prevweight;
						break;
					}

				}

				//Bias Updates
				if (layers[i].type != Layer::Softmax)
					biases[i][j] += lr * layers[i].values[0][j];
			}
		}
	}

	//STOP IF NO CONVOLUTIONS
	if (convstart == 0)
	{
		if (fullcarryvalues)
		{
			for (int j = 0; j < layers[0].number; j++)
			{
				if (layers[1].type != Layer::Dropout)
				{
					double sum = 0;
					for (int k = 0; k < layers[1].number; k++)
					{
						sum += layers[1].values[0][k] * weights[0][j][k];
					}
					layers[0].values[0][j] = sum;
				}
			}
		}

		return;
	}


	//UNFLATTEN
	unsigned long long int counter = 0;
	for (int i = 0; i < layers[convstart].values2D[0].size(); i++)
	{
		vector<vector<double>> derivative;
		for (int j = 0; j < layers[convstart].values2D[0][i].size(); j++)
		{
			vector<double> row;
			for (int k = 0; k < layers[convstart].values2D[0][i][j].size(); k++)
			{
				row.push_back(layers[convstart].values[0][counter]);
				counter++;
			}
			derivative.push_back(row);
		}
		layers[convstart].values2Dderivative[i] = derivative;
	}

	//2D Derivative
	for (int i = convstart; i > 0; i--)
	{
		//Convolution Case
		if (layers[i].type == Layer::Conv)
		{
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				for (int k = 0; k < layers[i - 1].kernelnumber; k++)
				{

					vector<vector<double>> deltay = layers[i - 1].values2D[0][k];
					vector<vector<double>> rotatedfilter = Rotate(&layers[i].kernels[j][k]);
					//Check for Strides
					if (layers[i].stride > 1)
					{
						deltay = Dilate2D(&deltay, layers[i].stride);
						rotatedfilter = Dilate2D(&rotatedfilter, layers[i].stride);
					}

					layers[i].deltakernel[j][k] = Convolve2D(&deltay, &layers[i].values2Dderivative[j], 1);
					vector<vector<double>> delta2D = FullConvolve2D(&rotatedfilter, &layers[i].values2Dderivative[j], 1);

					//Update Change
					AddVectors(&layers[i - 1].values2Dderivative[k], &delta2D);
					UpdateKernel(&layers[i].kernels[j][k], &layers[i].deltakernel[j][k], &layers[i].momentum2D[j][k], &layers[i].rmsp2D[j][k]);
				}
			}
		}

		//Pooling Case
		else if (layers[i].type == Layer::Pool2D)
		{
			for (int h = 0; h < layers[i].values2D[0].size(); h++)
			{
				if (layers[i - 1].values2Dderivative[h].size() == 0)
				{
					for (int m = 0; m < layers[i - 1].values2D[0][h].size(); m++)
					{
						vector<double> row(layers[i - 1].values2D[0][h][m].size());
						layers[i - 1].values2Dderivative[h].push_back(row);
					}
				}

				for (int j = 0; j < layers[i].values2D[0][0].size(); j++)
				{
					for (int k = 0; k < layers[i].values2D[0][0][0].size(); k++)
					{
						for (int l = 0; l < layers[i].padding; l++)
						{
							for (int m = 0; m < layers[i].padding; m++)
							{
								if (layers[i].pre_activation_values2D[0][h][j + l][k + m])
									layers[i - 1].values2Dderivative[h][j + l][k + m] = layers[i].values2Dderivative[h][j][k];
							}
						}
					}
				}
			}
		}
	}

	CleanLayers();
}

//Generic Functions
string Network::GetInitializerName()
{
	switch (WeightInitializer)
	{
	case Random:
		return "Random";
	case Glorot:
		return "Glorot";
	case He:
		return "He";
	}
}

string Network::GetOptimizerName()
{
	switch (Optimizer)
	{
	case No_Optimizer:
		return "No Optimizer";
	case Momentum:
		return "Momentum";
	case RMSProp:
		return "RMSProp";
	case Adam:
		return "Adam";
	}
}

string Network::GetRegularizerName()
{
	switch (Regularizer.type)
	{
	case No_Regularizer:
		return "No Regularizer";
	case L1:
		return "L1";
	case L2:
		return "L2";
	case Elastic_Net:
		return "Elastic Net";
	}
}

string Network::GetLRSchedulerName()
{
	switch (LR_Scheduler.type)
	{
	case No_Scheduler:
		return "No Scheduler";
	case Step_LR:
		return "Step LR";
	case Multi_Step_LR:
		return "Multi Step LR";
	case Constant_LR:
		return "Constant LR";
	case Linear_LR:
		return "Linear LR";
	case Exponential_LR:
		return "Exponential LR";
	case Reduce_LR_On_Plateau:
		return "Reduce LR On Plateau";
	}
}

//Layer Functions
void Network::AddLayer(Layer l)
{
	layers.push_back(l);
}

void Network::CleanLayers()
{
	for (int b = 0; b < batchsize; b++)
		for (int i = 0; i < layers.size(); i++)
		{
			if (layers[i].type == Layer::Conv)
			{
				for (int j = 0; j < layers[i].pre_activation_values2D[b].size(); j++)
				{
					for (int k = 0; k < layers[i].pre_activation_values2D[b][j].size(); k++)
					{
						for (int l = 0; l < layers[i].pre_activation_values2D[b][j][k].size(); l++)
						{
							layers[i].pre_activation_values2D[b][j][k][l] = 0;
						}
					}
				}
			}
		}
}

//Settings
void Network::SetOptimizer(string opt)
{
	if (!opt.compare("Momentum"))
		Optimizer = Momentum;
	else if (!opt.compare("RMSProp"))
		Optimizer = RMSProp;
	else if (!opt.compare("Adam"))
		Optimizer = Adam;
}

void Network::SetInitializer(string init)
{
	if (!init.compare("Glorot"))
		WeightInitializer = Glorot;
	else if (!init.compare("He"))
		WeightInitializer = He;
	else if (!init.compare("Random"))
		WeightInitializer = Random;

	return;
}

void Network::SetLRScheduler(string s)
{
	if (!s.compare("No_Scheduler"))
		LR_Scheduler.type = No_Scheduler;
	else if (!s.compare("Step_LR"))
		LR_Scheduler.type = Step_LR;
	else if (!s.compare("Multi_Step_LR"))
		LR_Scheduler.type = Multi_Step_LR;
	else if (!s.compare("Constant_LR"))
		LR_Scheduler.type = Constant_LR;
	else if (!s.compare("Linear_LR"))
		LR_Scheduler.type = Linear_LR;
	else if (!s.compare("Exponential_LR"))
		LR_Scheduler.type = Exponential_LR;
	else if (!s.compare("Reduce_LR_On_Plateau"))
		LR_Scheduler.type = Reduce_LR_On_Plateau;
}

void Network::SetRegularizer(string s)
{
	if (!s.compare("L1"))
		Regularizer.type = L1;
	else if (!s.compare("L2"))
		Regularizer.type = L2;
	else if (!s.compare("Elastic_Net"))
		Regularizer.type = Elastic_Net;
}

void Network::UpdateLearningRate(int epoch)
{
	static double prevloss = 0;
	static int patience = 0;

	bool update = false;

	switch (LR_Scheduler.type)
	{
	case No_Scheduler:
		return;

	case Step_LR:
		if (epoch % LR_Scheduler.step == 0)
		{
			lr *= LR_Scheduler.gamma;
			update = true;
		}
		break;

	case Multi_Step_LR:
		for (auto& i : LR_Scheduler.milestones)
			if (epoch == i)
			{
				lr = lr * LR_Scheduler.gamma;
				update = true;
				break;
			}
		break;

	case Constant_LR:
		if (epoch == LR_Scheduler.iterations)
		{
			lr *= LR_Scheduler.gamma;
			update = true;
		}
		break;

	case Linear_LR:
		if (LR_Scheduler.lineardiff == 0)
			LR_Scheduler.lineardiff = (LR_Scheduler.final_lr - lr) / (double)LR_Scheduler.iterations;
		if (LR_Scheduler.iterations-- > 0)
		{
			lr += LR_Scheduler.lineardiff;
			update = true;
		}
		break;

	case Exponential_LR:
		lr *= LR_Scheduler.gamma;
		update = true;
		break;

	case Reduce_LR_On_Plateau:

		if (prevloss == 0)
		{
			prevloss = epochloss;
			return;
		}

		double diff = epochloss - prevloss;
		if (abs(diff) < LR_Scheduler.threshold)
		{
			if (!LR_Scheduler.mode.compare("min") && diff < 0)
				patience++;
			else if (!LR_Scheduler.mode.compare("max") && diff > 0)
				patience++;
			else
				patience = 0;
		}
		else
			patience = 0;

		if (patience >= LR_Scheduler.patience)
		{
			lr *= LR_Scheduler.gamma;
			update = true;
		}

		prevloss = epochloss;
		break;
	}


	if (update)
	{
		if (lr < LR_Scheduler.min_lr)
			lr = LR_Scheduler.min_lr;

		cout << "\nNew Learning Rate: " << lr;
	}

}

void Network::PrintParameters()
{
	cout << "\n\n";
	for (int x = 0; x < layers.size() - 1; x++)
	{
		cout << "Layer " << x + 1 << " Weights:\n===============\n";
		int counter = 1;
		for (int y = 0; y < layers[x].number; y++)
		{
			for (int z = 0; z < layers[x + 1].number; z++)
			{
				//Prints Weights
				cout << counter++ << ". " << weights[x][y][z] << endl;
			}
		}
		cout << endl;
	}
}

void Network::Summary()
{
	//Network Summary
	cout << "NETWORK SUMMARY\n===============\n\n";
	for (int x = 0; x < layers.size(); x++)
	{
		if (layers[x].type == Layer::Pool2D)
			cout << "LAYER: " << "Pool2D" << "\t\tDIMENSIONS: " << layers[x].padding << "\n\n";
		else if (layers[x].type == Layer::Conv)
			cout << "LAYER: " << "Conv2D" << "\t\tKERNEL (SIZE: " << layers[x].kernelsize + (layers[x].dilation - 1) * (layers[x].kernelsize - 1) << ",NUMBER: " << layers[x].kernelnumber << ",DILATION: " << layers[x].dilation << ",STRIDE: " << layers[x].stride << ")" << "\n\n";
		else if (layers[x].type == Layer::Dropout)
			cout << "LAYER: " << "Dropout" << "\t\tRATE: " << layers[x].dropout << "\n\n";
		else
			cout << "LAYER: " << layers[x].neurontype << "\t\tNUMBER: " << layers[x].number << "\n\n";
	}
	cout << "\n";

	cout << "Weight Initializer:	 " << GetInitializerName() << "\n";
	cout << "Optimizer:		 " << GetOptimizerName() << "\n";
	cout << "Regularizer:		 " << GetRegularizerName() << "\n";
	cout << "LR Scheduler:		 " << GetLRSchedulerName() << "\n";
	cout << "\n\n";
}

//Initializings
void Network::Initialize()
{
	//Layer 0 Initialisation
	vector<vector<double>> temp;
	layers[0].kernelnumber = 1;
	layers[0].values2Dderivative.push_back(temp);

	//Parameter Initialisation
	weights = (double***)malloc(sizeof(double**) * layers.size() - 1);
	biases = (double**)malloc(sizeof(double*) * layers.size());
	momentum1D = (double***)malloc(sizeof(double**) * layers.size() - 1);
	rmsp1D = (double***)malloc(sizeof(double**) * layers.size() - 1);

	for (int x = 0; x < layers.size() - 1; x++)
	{
		//Check for Convolution Layer
		if (layers[x].type == Layer::Conv)
		{
			//Kernels
			for (int i = 0; i < layers[x].kernelnumber; i++)
			{
				vector<vector<vector<double>>> kernelset;
				layers[x].kernels.push_back(kernelset);
				layers[x].deltakernel.push_back(kernelset);
				layers[x].momentum2D.push_back(kernelset);
				layers[x].rmsp2D.push_back(kernelset);

				vector<vector<double>> difftemp;
				layers[x].values2Dderivative.push_back(temp);

				for (int j = 0; j < layers[x - 1].kernelnumber; j++)
				{
					vector<vector<double>> kernel = InitializeKernel(layers[x].kernelsize, layers[x].dilation);
					vector<vector<double>> deltaker;

					//Optimizer params
					vector<vector<double>> momentumkernel = Zero2DMatrix(layers[x].kernelsize + (layers[x].dilation - 1) * (layers[x].kernelsize - 1), layers[x].kernelsize + (layers[x].dilation - 1) * (layers[x].kernelsize - 1));

					layers[x].kernels[i].push_back(kernel);
					layers[x].deltakernel[i].push_back(deltaker);

					layers[x].momentum2D[i].push_back(momentumkernel);
					layers[x].rmsp2D[i].push_back(momentumkernel);
				}
			}
		}

		//Check for Pooling2D
		else if (layers[x].type == Layer::Pool2D)
		{
			layers[x].kernelnumber = layers[x - 1].kernelnumber;
			for (int i = 0; i < layers[x - 1].kernelnumber; i++)
			{
				vector<vector<double>> temp;
				layers[x].values2Dderivative.push_back(temp);
			}
		}

		//Other Cases
		else
		{
			//Check for Dropout1D
			if (layers[x].type == Layer::Dropout)
			{
				if (x == 0)
				{
					cout << "Cannot Add Dropout As First Layer" << endl;
					exit(0);
				}
				layers[x].number = layers[x - 1].number;

				//Weights
				weights[x] = (double**)malloc(sizeof(double*) * layers[x].number);
				biases[x] = (double*)malloc(sizeof(double) * layers[x].number);

				//Optimizer Params
				momentum1D[x] = (double**)malloc(sizeof(double*) * layers[x].number);
				rmsp1D[x] = (double**)malloc(sizeof(double*) * layers[x].number);

				for (int y = 0; y < layers[x].number; y++)
				{
					weights[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);
					momentum1D[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);
					rmsp1D[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);

					biases[x][y] = 0;
					for (int z = 0; z < layers[x + 1].number; z++)
					{
						weights[x][y][z] = WeightInitialization(layers[x].number, layers[x + 1].number);
						momentum1D[x][y][z] = 0.0;
						rmsp1D[x][y][z] = 0.0;
					}
				}

			}
			else
			{

				//Weights
				weights[x] = (double**)malloc(sizeof(double*) * layers[x].number);
				biases[x] = (double*)malloc(sizeof(double) * layers[x].number);

				//Optimizer Params
				momentum1D[x] = (double**)malloc(sizeof(double*) * layers[x].number);
				rmsp1D[x] = (double**)malloc(sizeof(double*) * layers[x].number);

				for (int y = 0; y < layers[x].number; y++)
				{
					weights[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);
					momentum1D[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);
					rmsp1D[x][y] = (double*)malloc(sizeof(double) * layers[x + 1].number);

					biases[x][y] = 0;
					for (int z = 0; z < layers[x + 1].number; z++)
					{
						weights[x][y][z] = WeightInitialization(layers[x].number, layers[x + 1].number);
						momentum1D[x][y][z] = 0.0;
						rmsp1D[x][y][z] = 0.0;
					}
				}
			}
		}
	}

	//Last Layer Exceptions
	if (layers.back().type == Layer::Dropout)
	{
		cout << "Cannot Add Dropout to Last Layer" << endl;
		exit(0);
	}

	//Last Layer
	biases[layers.size() - 1] = (double*)malloc(sizeof(double) * layers.back().number);
	for (int j = 0; j < layers.back().number; j++)
	{
		biases[layers.size() - 1][j] = (double)rand() / (double)RAND_MAX;
	}

	//Errors
	for (int x = 0; x < layers[layers.size() - 1].number; x++)
	{
		errors.push_back(0);
		derrors.push_back(0);
	}

}

void Network::InitializeValueMatrices(int batchsize)
{
	for (int t = 0; t < batchsize; t++)
	{
		//Row 0
		vector<vector<vector<double>>> temp2Dsetinit;

		vector<vector<double>>temp2D;
		temp2Dsetinit.push_back(temp2D);

		layers[0].values2D.push_back(temp2Dsetinit);
		layers[0].pre_activation_values2D.push_back(temp2Dsetinit);

		for (int i = 0; i < layers.size(); i++)
		{
			vector<double> temp(layers[i].number);
			layers[i].values.push_back(temp);
			layers[i].pre_activation_values.push_back(temp);
			layers[i].softmaxsum.push_back(0);

			vector<vector<vector<double>>> temp2Dset;
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				vector<vector<double>>temp2D;
				temp2Dset.push_back(temp2D);
			}
			layers[i].values2D.push_back(temp2Dset);
			layers[i].pre_activation_values2D.push_back(temp2Dset);
		}
	}
}

void Network::InitializePredictedMatrix(vector<vector<double>>* predicted)
{
	for (int j = 0; j < totalepochs; j++)
		for (int i = 0; i < totalinputsize; i++)
		{
			vector<double> temp(layers.back().number);
			predicted->push_back(temp);
		}
}

double Network::WeightInitialization(int fan_in, int fan_out)
{
	static default_random_engine generator;
	normal_distribution<double> glorotdist(0, 2 / (double)(fan_in + fan_out));
	normal_distribution<double> hedist(0, 2 / (double)(fan_in));
	double val;

	switch (WeightInitializer)
	{
	case Random:
		val = rand() / (double)RAND_MAX;
		return (rand() / (double)RAND_MAX > 0.5) ? val : -val;
	case Glorot:
		return glorotdist(generator);
	case He:
		return hedist(generator);
	}
}

//Compiling
void Network::Compile(string type)
{
	if (!type.compare("Stochastic"))
	{
		gradient_descent_type = Stochastic;
		batchsize = 1;
	}
	else if (!type.compare("Batch"))
	{
		gradient_descent_type = Batch;
	}
	else
	{
		cerr << "Mini Batch Gradient Descent Requires A Defined Batch Size" << endl;
		exit(0);
	}
	Initialize();
	InitializeValueMatrices(batchsize);
}

void Network::Compile(string type, int input_batch_size)
{
	gradient_descent_type = Mini_Batch;
	batchsize = input_batch_size;
	Initialize();
	InitializeValueMatrices(batchsize);
}

//Activations
double Network::DActivation(double x, int i)
{
	static double temp;
	switch (layers[i].type)
	{
	case Layer::Sigmoid:
		temp = Activation(x, i);
		return temp * (1 - temp);
	case Layer::Linear:
		return 1;
	case Layer::Relu:
		return (x > 0) ? 1 : 0;
	case Layer::LeakyRelu:
		return (x > 0) ? 1 : layers[i].parameters.LeakyReluAlpha;
	case Layer::Tanh:
		return 1 - pow(tanh(x), 2);
	case Layer::Softmax:
		temp = Activation(x, i);
		return temp * (1 - temp);
	}
}

double Network::Activation(double x, int i)
{
	switch (layers[i].type)
	{
	case Layer::Sigmoid:
		return 1 / (double)(1 + exp(-x));
	case Layer::Linear:
		return x;
	case Layer::Relu:
		return (x > 0) ? x : 0;
	case Layer::LeakyRelu:
		return (x > 0) ? x : layers[i].parameters.LeakyReluAlpha * x;
	case Layer::Tanh:
		return tanh(x);
	case Layer::Softmax:
		return exp(x) / (double)layers[i].softmaxsum[0];
	}
}

//Errors
void Network::ErrorCalculation(int samplenum, vector<double> actualvalue)
{
	static double avgerror = 0;
	static int counter = 0;
	static int totalcounter = 0;

	for (int i = 0; i < layers.back().number; i++)
	{
		double error = 0;
		double val = layers.back().values[samplenum][i];

		//Individual Errors
		switch (model_loss_type)
		{
		case Mean_Squared:
			error = pow(val - actualvalue[i], 2) / 2;
			break;
		case Mean_Absolute:
			error = abs(val - actualvalue[i]);
			break;
		case Mean_Biased:
			error = actualvalue[i] - val;
			break;
		case Root_Mean_Squared:
			error = pow(val - actualvalue[i], 2) / 2;
			break;
		case Binary_Cross_Entropy:
			error = -actualvalue[i] * log(val) - (1 - actualvalue[i]) * log(1 - val);
			break;
		case Categorical_Cross_Entropy:
			error = -actualvalue[i] * log(val);
			break;
		}

		if (gradient_descent_type == Stochastic)
		{
			errors[i] = error;
			derrors[i] = DActivation(layers.back().pre_activation_values[samplenum][i], layers.size() - 1) * DError(layers.back().values[samplenum][i], actualvalue[i], i);
		}
		else
		{
			errors[i] += error;
			derrors[i] += DActivation(layers.back().pre_activation_values[samplenum][i], layers.size() - 1) * DError(layers.back().values[samplenum][i], actualvalue[i], i);

		}


		avgerror += errors[i];
	}


	counter++;
	totalcounter++;
}

void Network::AccumulateErrors()
{
	double errorsum = 0;
	for (int i = 0; i < layers.back().number; i++)
	{
		switch (model_loss_type)
		{
		case Root_Mean_Squared:
			errors[i] = sqrt(errors[i] / (double)(2 * batchsize));
			derrors[i] /= (double)(batchsize * 2 * errors[i]);
			errorsum += errors[i];
			break;
		case Mean_Squared:
		case Mean_Absolute:
		case Mean_Biased:
		case Binary_Cross_Entropy:
		case Categorical_Cross_Entropy:
			derrors[i] /= (double)batchsize;
			errorsum += errors[i];
			break;
		}
	}
	epochloss += errorsum / (double)totalinputsize;
}

double Network::DError(double predictedvalue, double actualvalue, int neuronnum)
{
	switch (model_loss_type)
	{
	case Mean_Squared:
		return actualvalue - predictedvalue;
	case Mean_Absolute:
		return -predictedvalue / abs(predictedvalue);
	case Mean_Biased:
		return -predictedvalue;
	case Root_Mean_Squared:
		return actualvalue - predictedvalue;
	case Binary_Cross_Entropy:
		return (actualvalue - predictedvalue) / (double)(predictedvalue * (1 - predictedvalue));
	case Categorical_Cross_Entropy:
		return actualvalue / (double)predictedvalue;
	}
}

void Network::CleanErrors()
{
	for (int x = 0; x < layers.back().number; x++)
	{
		errors[x] = 0;
		derrors[x] = 0;
	}
}

//Params
void Network::LeakyReluParameters(double i, double a)
{
	if (i<1 || i>layers.size())
	{
		cout << "Layer Number Out Of Range" << endl;
		return;
	}
	layers[i - 1].parameters.LeakyReluAlpha = a;
}

void Network::ShowTrainingStats(vector<vector<double>>* inputs, vector<vector<double>>* actual, int i)
{
	//cout << "Inputs: ";
	//for (int j = 0; j < (*inputs)[i].size(); j++)
	//{
	//	cout << (*inputs)[i][j] << " ";
	//}

	cout << "\tPredicted: ";
	for (int j = 0; j < layers[layers.size() - 1].number; j++)
	{
		cout << layers[layers.size() - 1].values[0][j] << " ";
	}

	cout << "\tActual: ";
	for (int j = 0; j < (*actual)[i].size(); j++)
	{
		cout << (*actual)[i][j] << " ";
	}

	cout << "\tErrors: ";
	for (int j = 0; j < layers[layers.size() - 1].number; j++)
	{
		cout << errors[j] << " ";
	}

	cout << endl;
}

void Network::SetDisplayParameters(string s)
{
	if (!s.compare("Visual"))
		displayparameters = Visual;
	else if (!s.compare("Text"))
		displayparameters = Text;
}

//Predicting
void Network::Predict(vector<vector<double>>* input, vector<vector<double>>* predicted)
{
	input1D = true;

	vector<vector<double>> tempinput = *input;
	vector<vector<vector<double>>> inputs = { tempinput };
	Predict(&inputs, predicted);
}

void Network::Predict(vector<vector<vector<double>>>* input, vector<vector<double>>* predicted)
{

	CleanErrors();

	int totalsize;
	if (input1D)
	{
		totalsize = (*input)[0].size();
	}
	else
	{
		totalsize = input->size();
	}

	
	for (int i = 0; i < totalsize; i++)
	{
		//Taking the sample
		vector<vector<double>> sample;

		//Check for 1D or 2D
		if (input1D)
		{
			sample = { (*input)[0][i] };
		}
		else
		{
			sample = (*input)[i];
		}

		vector<double> actual(layers.back().number);

		//Calculate Prop
		ForwardPropogation(0, sample, actual);

		//Get Prediction
		vector<double>prediction = layers.back().values[0];
		predicted->push_back(prediction);

		//Clean Errors
		CleanErrors();
	}
	
}


//Training
void Network::Train(vector<vector<double>>* inputs, vector<vector<double>>* actual, int epochs, string losstype)
{
	vector<vector<double>> tempinput = *inputs;
	vector<vector<vector<double>>> input = { tempinput };
	Train(&input, actual, epochs, losstype);
}

void Network::Train(vector<vector<vector<double>>>* inputs, vector<vector<double>>* actual, int epochs, string loss)
{
	totalinputsize = inputs->size();
	if (totalinputsize == 1)
		totalinputsize = (*inputs)[0].size();

	totalepochs = epochs;

	if (!loss.compare("MSE"))
	{
		model_loss_type = Mean_Squared;
	}
	else if (!loss.compare("MAE"))
	{
		model_loss_type = Mean_Absolute;
	}
	else if (!loss.compare("MBE"))
	{
		model_loss_type = Mean_Biased;
	}
	else if (!loss.compare("RMSE"))
	{
		model_loss_type = Root_Mean_Squared;
	}
	else if (!loss.compare("BCE"))
	{
		model_loss_type = Binary_Cross_Entropy;
	}
	else if (!loss.compare("CCE"))
	{
		model_loss_type = Categorical_Cross_Entropy;
	}

	if (verbosetraining)
		cout << "Training\n========\n" << endl;

	bool initializationpass = false;
	threadcounter = 0;

	switch (gradient_descent_type)
	{
	case Batch:batchsize = totalinputsize;
	case Stochastic:
	case Mini_Batch:

		//Calculate Batch Sizes
		batchnum = totalinputsize / (float)batchsize;

		//Verbosity
		if (verbosetraining)
		{
			cout << "Total Batches= " << batchnum << "\n";
			cout << "Batch Size= " << batchsize << "\n\n";

			cout << "\nTraining Started";
			cout << "\n----------------";
		}

		int threaddeaths = 0;
		double averagetime = 0;
		for (int l = 0; l < epochs; l++)
		{
			if (verbosetraining)
				cout << "\n\nEpoch: " << l + 1 << "\n";

			for (int j = 0; j < batchnum; j++)
			{
				for (int i = 0; i < batchsize; i++)
				{

					//Taking the sample
					vector<vector<double>> sample;

					//Check for 1D or 2D
					if (inputs->size() == 1)
					{
						sample = { (*inputs)[0][j * batchsize + i] };
					}
					else
					{
						sample = (*inputs)[j * batchsize + i];
					}

					//Actual Value
					vector<double> actualvalue = (*actual)[j * batchsize + i];

					//First Pass for flattening weights
					if (!initializationpass)
					{
						ForwardPropogation(i, sample, actualvalue);
						initializationpass = true;
					}
					else
					{
						thread t(&Network::ForwardPropogation, this, i, sample, actualvalue);
						t.detach();
					}
					//if (displayparameters == Text)
					//ShowTrainingStats(inputs, actual, i);

				}

				//Show Batch Status
				if (verbosetraining)
					cout << "\r" << "Batch " << j + 1 << " / " << batchnum;

				//Check Thread Deaths
				double batchtime = 0;
				while (threadcounter < batchsize) {
					if (averagetime > 0 && batchtime > 10 * averagetime)
					{
						threaddeaths++;
						break;
					}
					batchtime += 10e-10;
				};
				averagetime = batchtime;

				//Reset Counters
				threadcounter = 0;
				batchcounter++;

				//Backprop
				AccumulateErrors();
				BackPropogation();

				if (cleanerrors)
					CleanErrors();
			}

			//Print Loss
			if (verbosetraining || verboseloss)
				cout << "\nLoss: " << epochloss;

			UpdateLearningRate(l + 1);

			//Alter counters
			epochloss = 0;
			batchcounter = 0;
			epochcounter++;
		}

		//Display Thread Deaths
		if (verbosetraining)
		{
			cout << "\n\nTraining Completed";
			cout << "\n------------------\n\n";
		}
		break;
	}

	epochcounter = 0;
}
//