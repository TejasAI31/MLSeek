#include "model.h"

using namespace std;

//Utility
void Network::DisplayTensor(vector<vector<float>>& tensor)
{
	cout << '\n';
	for (int i = 0; i < tensor.size(); i++)
	{
		for (int j = 0; j < tensor[0].size(); j++)
		{
			cout << std::setprecision(2) << tensor[i][j] << " ";
		}
		cout << "\n";
	}
	cout << "\n";
}

vector<float> Network::OnesMatrix(int num)
{
	vector<float> out(num, 1);
	return out;
}

vector<float> Network::ZerosMatrix(int num)
{
	vector<float> out(num, 0);
	return out;
}

//Image Transformations
float Network::MatrixAverage(vector<vector<float>>* mat)
{
	float sum = 0;
	for (int x = 0; x < mat->size(); x++)
	{
		for (int y = 0; y < (*mat)[0].size(); y++)
		{
			sum += (*mat)[x][y];
		}
	}
	return sum / (float)(mat->size() * (*mat)[0].size());
}

vector<vector<float>> Network::PixelDistances(vector<vector<float>>* mat1, vector<vector<float>>* mat2)
{
	vector<vector<float>> distancemat;
	if (mat1->size() != mat2->size() || mat1->empty())
		return {};

	for (int x = 0; x < mat1->size(); x++)
	{
		vector<float> row;
		for (int y = 0; y < (*mat1)[0].size(); y++)
		{
			row.push_back(sqrt(pow((*mat1)[x][y], 2) + pow((*mat2)[x][y], 2)));
		}
		distancemat.push_back(row);
	}

	return distancemat;
}

vector<vector<vector<float>>> Network::EmptyUpscale(vector<vector<vector<float>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<float>>> finalimage;

	for (int i = 0; i < image->size(); i++)
	{
		vector<vector<float>> upscaled;
		int rowsize = (int)(*image)[i][0].size();
		int columnsize = (int)(*image)[i].size();
		int rowstep = (int)floor(finalheight / columnsize);
		int columnstep = (int)floor(finalwidth / rowsize);

		int ydeficit = finalheight - columnsize;
		int ycounter = 0;
		for (int x = 0; x < finalheight; x++)
		{
			//Last Column
			if (ycounter == columnsize - 1)
			{
				for (int y = 0; y < finalheight - x - 1; y++)
				{
					vector<float> empty(finalwidth, 123.456);
					upscaled.push_back(empty);
				}
				break;
			}

			//Enter Full Row
			vector<float> row(finalwidth, 123.456);
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
					vector<float> empty(finalwidth, 123.456);
					upscaled.push_back(empty);
				}
				x += rowstep;
			}
			ydeficit--;
			ycounter++;
		}

		//Final Row
		vector<float> row(finalwidth, 123.456);
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

vector<vector<vector<float>>> Network::SobelEdgeDetection(vector<vector<vector<float>>>* image)
{
	static vector<vector<float>> sobelx = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};

	static vector<vector<float>> sobely = {
		{1,2,1},
		{0,0,0},
		{-1,-2,-1}
	};

	vector<vector<vector<float>>> edgeimages;

	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<float>> xedges;
		vector<vector<float>> yedges;

		Convolve2D((*image)[x], sobelx, 1, xedges);
		Convolve2D((*image)[x], sobely, 1, yedges);

		vector<vector<float>> magnitude = PixelDistances(&xedges, &yedges);

		float threshold = MatrixAverage(&magnitude);

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

vector<vector<vector<float>>> Network::PrewittEdgeDetection(vector<vector<vector<float>>>* image)
{
	static vector<vector<float>> prewittx = {
		{-1,0,1},
		{-1,0,1},
		{-1,0,1}
	};

	static vector<vector<float>> prewitty = {
		{1,1,1},
		{0,0,0},
		{-1,-1,-1}
	};

	vector<vector<vector<float>>> edgeimages;

	for (int x = 0; x < image->size(); x++)
	{
		vector<vector<float>> xedges;
		vector<vector<float>> yedges;

		Convolve2D((*image)[x], prewittx, 1,xedges);
		Convolve2D((*image)[x], prewitty, 1,yedges);

		vector<vector<float>> magnitude = PixelDistances(&xedges, &yedges);

		float threshold = MatrixAverage(&magnitude);

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

vector<vector<vector<float>>> Network::NNInterpolation(vector<vector<vector<float>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<float>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			//Empty Row
			if (emptyupscaled[i][x][0] == 123.456)continue;

			float value = emptyupscaled[i][x][0];
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
			float value = emptyupscaled[i][0][x];
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

vector<vector<vector<float>>> Network::BilinearInterpolation(vector<vector<vector<float>>>* image, int finalwidth, int finalheight)
{
	vector<vector<vector<float>>> emptyupscaled = EmptyUpscale(image, finalwidth, finalheight);
	for (int i = 0; i < emptyupscaled.size(); i++)
	{
		//Row Interpolation
		for (int x = 0; x < emptyupscaled[i].size(); x++)
		{
			float value = emptyupscaled[i][x][0];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i][x].size(); y++)
			{
				if (emptyupscaled[i][x][y] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][x][valuecount + z] = value * (denom - z) / (float)denom + emptyupscaled[i][x][y] * z / (float)denom;
					}
					value = emptyupscaled[i][x][y];
					valuecount = y;
				}
			}
		}

		//Column Interpolation
		for (int x = 0; x < emptyupscaled[i][0].size(); x++)
		{
			float value = emptyupscaled[i][0][x];
			int valuecount = 0;
			for (int y = 1; y < emptyupscaled[i].size(); y++)
			{
				if (emptyupscaled[i][y][x] != 123.456)
				{
					int denom = y - valuecount;
					for (int z = 1; z < denom; z++)
					{
						emptyupscaled[i][valuecount + z][x] = value * (denom - z) / (float)denom + emptyupscaled[i][y][x] * z / (float)denom;
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
void Network::Convolve2D(vector<vector<float>>& input, vector<vector<float>>& kernel, int stride,vector<vector<float>>& dst)
{
	short int kernelsize = (int)kernel.size();
	short int rows = (int)input.size();
	short int columns = (int)input[0].size();


	//Create Dummy Array
	if (dst.size() == 0)
	{
		for (int x = 0; x <= (rows - kernelsize)/stride; x += stride)
		{
			vector<float> row;
			for (int y = 0; y <= (columns - kernelsize)/stride; y += stride)
			{
				row.push_back(0.0);
			}
			dst.push_back(row);
		}
	}

	//Convolve
	for (int x = 0; x <= (rows - kernelsize)/stride; x += stride)
	{
		for (int y = 0; y <= (columns - kernelsize)/stride; y += stride)
		{
			float value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (x + i >= rows || y + j >= columns)
						continue;
					else
						value += kernel[i][j] * input[x + i][y + j];
				}
			}
			dst[y][x] += value;
		}
	}

	return;
}

void Network::FullConvolve2D(vector<vector<float>>& input, vector<vector<float>>& kernel, int stride,vector<vector<float>>&dst)
{
	short int kernelsize = (int)kernel.size();
	short int rows = (int)input.size();
	short int columns = (int)input[0].size();

	if (dst.size() == 0)
	{
		for (int x = 1 - rows; x < kernelsize; x += stride)
		{
			vector<float> row;
			for (int y = 1 - columns; y < kernelsize ; y += stride)
			{
				row.push_back(0);
			}
			dst.push_back(row);
		}
	}


	for (int x = 0; x < rows-1+kernelsize; x += stride)
	{
		for (int y = 0; y < columns-1+kernelsize; y += stride)
		{
			float value = 0;
			for (int i = 0; i < kernelsize; i++)
			{
				for (int j = 0; j < kernelsize; j++)
				{
					if (- 1 - x+i >=0 || -1-y+j>=0||rows-1-x+i<0||columns-1-y+j<0)
						continue;
					else
						value += kernel[i][j] * input[rows-1-x+i][columns-1-y+j];
				}
			}
			dst[x][y]+=value;
		}
	}

}

vector<vector<float>> Network::Relu2D(vector<vector<float>>* input)
{
	vector<vector<float>> output;
	for (int i = 0; i < input->size(); i++)
	{
		vector<float> row;
		for (int j = 0; j < (*input)[0].size(); j++)
		{
			row.push_back((*input)[i][j] > 0 ? (*input)[i][j] : 0);
		}
		output.push_back(row);
	}
	return output;
}

vector<vector<float>> Network::Rotate(vector<vector<float>>* input)
{
	vector<vector<float>> rotated;
	for (int i = (int)input->size() - 1; i >= 0; i--)
	{
		vector<float> row;
		for (int j = (int)(*input)[i].size() - 1; j >= 0; j--)
		{
			row.push_back((*input)[i][j]);
		}
		rotated.push_back(row);
	}
	return rotated;
}

void Network::AddVectors(vector<vector<float>>& v1, vector<vector<float>>& v2)
{
	if (v1.size() == 0)
	{
		v1 = v2;
		return;
	}
	else
	{
		for (int i = 0; i < v1.size(); i++)
		{
			for (int j = 0; j < v1[0].size(); j++)
			{
				v1[i][j] += v2[i][j];
			}
		}
	}
}

void Network::MaxPooling2D(vector<vector<float>>* input, short int padnum, vector<vector<float>>* outputdest, vector<vector<float>>* chosendest)
{
	int columns = (int)input->size();
	int rows = (int)(*input)[0].size();
	short int rowpadding = padnum - rows % padnum;
	short int columnpadding = padnum - columns % padnum;

	vector<vector<float>> output;
	vector<vector<float>> chosenvalues(columns, vector<float>(rows));


	for (int y = 0; y <= columns + columnpadding; y += padnum)
	{
		vector<float> row;
		for (int x = 0; x <= rows + rowpadding; x += padnum)
		{
			short int chosenx = 0;
			short int choseny = 0;
			float maxval = 0;
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

void Network::UpdateKernel(vector<vector<float>>* v1, vector<vector<float>>* v2, vector<vector<float>>* momentumkernel, vector<vector<float>>* rmspkernel)
{
	for (int i = 0; i < v1->size(); i++)
	{
		for (int j = 0; j < (*v1)[0].size(); j++)
		{
			float change = (*v2)[i][j] / (float)batchsize;
			switch (Optimizer)
			{
			case No_Optimizer:
				(*v1)[i][j] += change * lr;
				break;
			case Momentum:
				(*momentumkernel)[i][j] = momentumbeta * (*momentumkernel)[i][j] + (1 - momentumbeta) * change;
				(*v1)[i][j] += (*momentumkernel)[i][j] * lr;
				break;
			case RMSProp:
				(*rmspkernel)[i][j] = rmspropbeta * (*rmspkernel)[i][j] + (1 - momentumbeta) * pow(change, 2);
				(*v1)[i][j] += change / sqrt((*rmspkernel)[i][j] + rmspropepsilon) * lr;
				break;
			case Adam:
				(*momentumkernel)[i][j] = momentumbeta * (*momentumkernel)[i][j] + (1 - momentumbeta) * change;
				(*rmspkernel)[i][j] = rmspropbeta * (*rmspkernel)[i][j] + (1 - momentumbeta) * pow(change, 2);
				(*v1)[i][j] += (*momentumkernel)[i][j] / sqrt((*rmspkernel)[i][j] + rmspropepsilon) * lr;
				break;
			}
			(*v2)[i][j] = 0;
		}
	}
	return;
}

vector<vector<float>> Network::Dilate2D(vector<vector<float>>* input, int dilation)
{
	vector<vector<float>> output;

	int rows = (int)input->size();
	int cols = (int)(*input)[0].size();

	int rowpos = 0;
	int colpos = 0;

	for (int j = 0; j < rows + (dilation - 1) * (rows - 1); j++)
	{
		vector<float> row;
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

vector<vector<float>> Network::Zero2DMatrix(int x, int y)
{
	vector<vector<float>> mat;
	for (int ay = 0; ay < y; ay++)
	{
		vector<float> zerorow(x);
		mat.push_back(zerorow);
	}
	return mat;
}

vector<vector<float>> Network::InitializeKernel(int kernelsize, int dilation)
{
	vector<vector<float>> kernel;

	for (int j = 0; j < kernelsize + (dilation - 1) * (kernelsize - 1); j++)
	{
		vector<float> row;
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


//PROPOGATION
void Network::ForwardPropogation(int samplenum, vector<vector<float>> sample, vector<float> actualvalue)
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
					vector<vector<float>> convolution;
					Convolve2D(layers[i - 1].values2D[samplenum][k], layers[i].kernels[j][k], layers[i].stride,convolution);
					AddVectors(layers[i].pre_activation_values2D[samplenum][j], convolution);
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
				layers[i - 1].values[samplenum] = vector<float>(layers[i - 1].values2D[samplenum].size() * layers[i - 1].values2D[samplenum][0].size() * layers[i - 1].values2D[samplenum][0][0].size());
				layers[i - 1].number = layers[i - 1].values2D[samplenum].size() * (int)layers[i - 1].values2D[samplenum][0].size() * layers[i - 1].values2D[samplenum][0][0].size();
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
				weights[i - 1] = (float**)malloc(sizeof(float*) * layers[i - 1].number);
				delta_weights[i-1]= (float**)malloc(sizeof(float*) * layers[i - 1].number);
				momentum1D[i - 1] = (float**)malloc(sizeof(float*) * layers[i - 1].number);
				rmsp1D[i - 1] = (float**)malloc(sizeof(float*) * layers[i - 1].number);

				for (int j = 0; j < layers[i - 1].number; j++)
				{
					weights[i - 1][j] = (float*)malloc(sizeof(float) * layers[i].number);
					delta_weights[i - 1][j] = (float*)malloc(sizeof(float) * layers[i].number);
					momentum1D[i - 1][j] = (float*)malloc(sizeof(float) * layers[i].number);
					rmsp1D[i - 1][j] = (float*)malloc(sizeof(float) * layers[i].number);

					for (int k = 0; k < layers[i].number; k++)
					{
						weights[i - 1][j][k] = WeightInitialization(layers[i - 1].number, layers[i].number);
						delta_weights[i - 1][j][k] = 0.0;
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
			float multrate = 1 / (float)(1 - layers[i].dropout);
			int totaloff = (int)(layers[i].dropout * layers[i].number);
			int off = 0;

			//Initialization
			for (int j = 0; j < layers[i].number; j++)
			{
				layers[i].values[samplenum][j] = layers[i - 1].values[samplenum][j];
				if (layers[i].values[samplenum][j] == 0)layers[i].values[samplenum][j] = 0.0001;
			}

			//Train only
			if (Model_Mode == train)
			{
				//Turn off neurons
				if (totaloff > 0)
				{
					bool flag = 0;
					while (!flag)
						for (int j = 0; j < layers[i].number; j++)
						{
							if (rand() / (float)RAND_MAX < layers[i].dropout && layers[i].values[samplenum][j] != 0)
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
		}

		else if (layers[i].type == Layer::Softmax)
		{
			//Calculate Pre Activation Values
			for (int j = 0; j < layers[i].number; j++)
			{
				float sum = 0;
				for (int k = 0; k < layers[i - 1].number; k++)
				{
					sum += weights[i - 1][k][j] * layers[i - 1].values[samplenum][k];
				}
				layers[i].pre_activation_values[samplenum][j] = sum;
			}

			//Calculate Softsum
			float softsum = 0;
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
				float sum = 0;
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


	if (Model_Mode == train)
	{
		//Call Backpropogation
		BackPropogation(samplenum);

		//Increment ThreadCounter
		lock_guard<mutex> counter_lock(*thread_counter_mutex);
		threadcounter++;;
	}

	return;
}

void Network::BackPropogation(int samplenum)
{
	//Initial Error
	short int final_layer = (int)layers.size() - 1;

	//Shift errors to values
	layers.back().values[samplenum] = derrors[samplenum];

	int convstart = 0;

	unique_lock<mutex> delta_lock(*delta_mutex);
	for (int i = final_layer; i > 0; i--)
	{
		//Check for Convolution Start
		if (layers[i].type == Layer::Conv || layers[i].type == Layer::Pool2D)
		{
			convstart = i;
			//Calculate derivate
			for (int j = 0; j < layers[i].number; j++)
			{
				float sum = 0;
				for (int k = 0; k < layers[i + 1].number; k++)
				{
					sum += layers[i + 1].values[samplenum][k] * weights[i][j][k];
				}
				layers[i].values[samplenum][j] = sum;
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
				if (layers[i].values[samplenum][j] == 0)
				{
					layers[i - 1].values[samplenum][j] = 0;
					continue;
				}

				float sum = 0;
				for (int k = 0; k < layers[i + 1].number; k++)
				{
					sum += layers[i + 1].values[samplenum][k] * weights[i][j][k];
				}

				layers[i].values[samplenum][j] = DActivation(layers[i].pre_activation_values[samplenum][j], i) * sum;
				layers[i - 1].values[samplenum][j] = layers[i].values[samplenum][j];

			}

			else
			{
				//Calculate Delta Values 
				if (i < final_layer)
					if (layers[i + 1].type != Layer::Dropout)
					{
						float sum = 0;
						for (int k = 0; k < layers[i + 1].number; k++)
						{
							sum += layers[i + 1].values[samplenum][k] * weights[i][j][k];
						}
						layers[i].values[samplenum][j] = DActivation(layers[i].pre_activation_values[samplenum][j], i) * sum;
					}

				//Delta Weights
				for (int k = 0; k < layers[i - 1].number; k++)
				{
					delta_weights[i - 1][k][j] += (layers[i].values[samplenum][j] * layers[i - 1].values[samplenum][k]) / (float)(batchsize);
				}

				//Bias Updates
				if (layers[i].type != Layer::Softmax)
				{
					delta_biases[i][j] += layers[i].values[samplenum][j] / (float)(batchsize);
				}

			}
		}
	}
	delta_lock.unlock();


	//STOP IF NO CONVOLUTIONS
	if (convstart == 0)
	{
		if (fullcarryvalues) //Backprop upto start layer
		{
			for (int j = 0; j < layers[0].number; j++)
			{
				if (layers[1].type != Layer::Dropout)
				{
					float sum = 0;
					for (int k = 0; k < layers[1].number; k++)
					{
						sum += layers[1].values[samplenum][k] * weights[0][j][k];
					}
					layers[0].values[samplenum][j] = sum;
				}
			}
		}

		return;
	}


	//UNFLATTEN
	long long int counter = 0;
	for (int i = 0; i < layers[convstart].values2D[samplenum].size(); i++)
	{
		//Create Matrix if not existing
		if (layers[convstart].values2Dderivative[samplenum][i].size() == 0)
		{
			for (int j = 0; j < layers[convstart].values2D[samplenum][i].size(); j++)
			{
				vector<float> row;
				for (int k = 0; k < layers[convstart].values2D[samplenum][i][j].size(); k++)
				{
					row.push_back(0);
				}
				layers[convstart].values2Dderivative[samplenum][i].push_back(row);
			}
		}

		//Add Values
		for (int j = 0; j < layers[convstart].values2D[samplenum][i].size(); j++)
		{
			vector<float> row;
			for (int k = 0; k < layers[convstart].values2D[samplenum][i][j].size(); k++)
			{
				if(layers[convstart].pre_activation_values2D[samplenum][i][j][k]>0)
					layers[convstart].values2Dderivative[samplenum][i][j][k]=layers[convstart].values[samplenum][counter];
				else
					layers[convstart].values2Dderivative[samplenum][i][j][k] = 0;

				counter++;
			}
		}

	}

	//2D Derivative
	unique_lock<mutex> conv_lock(*conv_mutex);
	for (int i = convstart; i > 0; i--)
	{
		//Convolution Case
		if (layers[i].type == Layer::Conv)
		{
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				for (int k = 0; k < layers[i - 1].kernelnumber; k++)
				{
					vector<vector<float>> deltay = layers[i].values2Dderivative[samplenum][j];
					vector<vector<float>> rotatedfilter = Rotate(&layers[i].kernels[j][k]);
					//Check for Strides
					if (layers[i].stride > 1)
					{
						deltay = Dilate2D(&deltay, layers[i].stride);
						rotatedfilter = Dilate2D(&rotatedfilter, layers[i].stride);
					}
					Convolve2D(layers[i - 1].values2D[samplenum][k], deltay, 1, layers[i].deltakernel[j][k]);
					vector<vector<float>> delta2D;
					FullConvolve2D(rotatedfilter,layers[i].values2Dderivative[samplenum][j], 1,delta2D);

					//Update Change
					AddVectors(layers[i - 1].values2Dderivative[samplenum][k], delta2D);
				}
			}
		}

		//Pooling Case
		else if (layers[i].type == Layer::Pool2D)
		{
			for (int h = 0; h < layers[i].values2D[samplenum].size(); h++)
			{
				if (layers[i - 1].values2Dderivative[samplenum][h].size() == 0)
				{
					for (int m = 0; m < layers[i - 1].values2D[samplenum][h].size(); m++)
					{
						vector<float> row(layers[i - 1].values2D[samplenum][h][m].size());
						layers[i - 1].values2Dderivative[samplenum][h].push_back(row);
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
								if (layers[i].pre_activation_values2D[samplenum][h][j + l][k + m]!=0)
									layers[i - 1].values2Dderivative[samplenum][h][j + l][k + m] = layers[i].values2Dderivative[samplenum][h][j][k];
							}
						}
					}
				}
			}
		}
	}
	conv_lock.unlock();

	CleanLayers(samplenum);
}

void Network::UpdateParameters()
{
	int convstart = 0;
	for (int i = (int)layers.size()-1; i > 0; i--)
	{
		//Check for Convolution Start
		if (layers[i].type == Layer::Conv || layers[i].type == Layer::Pool2D)
		{
			convstart = i;
			break;
		}
		if (layers[i].type == Layer::Dropout)
			continue;

		for (int j = 0; j < layers[i].number; j++)
		{
			for (int k = 0; k < layers[i - 1].number; k++)
			{
				float prevweight = weights[i - 1][k][j];

				//Weight Updates
				switch (Optimizer)
				{
				case No_Optimizer:
					weights[i - 1][k][j] += lr * delta_weights[i - 1][k][j];
					break;
				case Momentum:
					momentum1D[i - 1][k][j] = momentumbeta * momentum1D[i - 1][k][j] + (1 - momentumbeta) * (delta_weights[i - 1][k][j]);
					weights[i - 1][k][j] += lr * momentum1D[i - 1][k][j];
					break;
				case RMSProp:
					rmsp1D[i - 1][k][j] = rmspropbeta * rmsp1D[i - 1][k][j] + (1 - rmspropbeta) * pow(delta_weights[i - 1][k][j], 2);
					weights[i - 1][k][j] += lr * delta_weights[i - 1][k][j] / (sqrt(rmsp1D[i - 1][k][j]) + rmspropepsilon);
					break;
				case Adam:
					momentum1D[i - 1][k][j] = momentumbeta * momentum1D[i - 1][k][j] + (1 - momentumbeta) * (delta_weights[i - 1][k][j]);
					rmsp1D[i - 1][k][j] = rmspropbeta * rmsp1D[i - 1][k][j] + (1 - rmspropbeta) * pow(delta_weights[i - 1][k][j], 2);
					weights[i - 1][k][j] += lr * momentum1D[i - 1][k][j] / (sqrt(rmsp1D[i - 1][k][j]) + rmspropepsilon);
					break;
				}
				delta_weights[i - 1][k][j] = 0;

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
			{
				biases[i][j] += lr * delta_biases[i][j];
				delta_biases[i][j] = 0;
			}

		}
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
					UpdateKernel(&layers[i].kernels[j][k], &layers[i].deltakernel[j][k], &layers[i].momentum2D[j][k], &layers[i].rmsp2D[j][k]);
				}
			}
		}

	}
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

	return "";
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

	return "";
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

	return "";
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

	return "";
}

//Layer Functions
void Network::AddLayer(Layer l)
{
	if (l.type == Layer::Input2D)
		input1D = false;
	else if (l.type == Layer::Input)
		input1D = true;
	layers.push_back(l);
}

void Network::CleanLayers(int samplenum)
{
	for (int i = 0; i < layers.size(); i++)
	{
		if (layers[i].type == Layer::Conv||layers[i].type==Layer::Input2D)
		{
			for (int j = 0; j < layers[i].pre_activation_values2D[samplenum].size(); j++)
			{
				for (int k = 0; k < layers[i].pre_activation_values2D[samplenum][j].size(); k++)
				{
					for (int l = 0; l < layers[i].pre_activation_values2D[samplenum][j][k].size(); l++)
					{
						layers[i].pre_activation_values2D[samplenum][j][k][l] = 0;
						layers[i].values2Dderivative[samplenum][j][k][l] = 0;
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
	static float prevloss = 0;
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
			LR_Scheduler.lineardiff = (LR_Scheduler.final_lr - lr) / (float)LR_Scheduler.iterations;
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

		float diff = epochloss - prevloss;
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
	vector<vector<float>> temp;
	layers[0].kernelnumber = 1;

	//Parameter Initialisation
	weights = (float***)malloc(sizeof(float**) * layers.size() - 1);
	delta_weights = (float***)malloc(sizeof(float**) * layers.size() - 1);
	biases = (float**)malloc(sizeof(float*) * layers.size());
	delta_biases = (float**)malloc(sizeof(float*) * layers.size());
	momentum1D = (float***)malloc(sizeof(float**) * layers.size() - 1);
	rmsp1D = (float***)malloc(sizeof(float**) * layers.size() - 1);

	for (int x = 0; x < layers.size() - 1; x++)
	{
		//Check for Convolution Layer
		if (layers[x].type == Layer::Conv)
		{
			//Kernels
			for (int i = 0; i < layers[x].kernelnumber; i++)
			{
				vector<vector<vector<float>>> kernelset;
				layers[x].kernels.push_back(kernelset);
				layers[x].deltakernel.push_back(kernelset);
				layers[x].momentum2D.push_back(kernelset);
				layers[x].rmsp2D.push_back(kernelset);

				for (int j = 0; j < layers[x - 1].kernelnumber; j++)
				{
					vector<vector<float>> kernel = InitializeKernel(layers[x].kernelsize, layers[x].dilation);
					vector<vector<float>> deltaker;

					//Optimizer params
					vector<vector<float>> momentumkernel = Zero2DMatrix(layers[x].kernelsize + (layers[x].dilation - 1) * (layers[x].kernelsize - 1), layers[x].kernelsize + (layers[x].dilation - 1) * (layers[x].kernelsize - 1));

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

			}


			//Weights
			weights[x] = (float**)malloc(sizeof(float*) * layers[x].number);
			delta_weights[x] = (float**)malloc(sizeof(float*) * layers[x].number);
			biases[x] = (float*)malloc(sizeof(float) * layers[x].number);
			delta_biases[x] = (float*)malloc(sizeof(float) * layers[x].number);

			//Optimizer Params
			momentum1D[x] = (float**)malloc(sizeof(float*) * layers[x].number);
			rmsp1D[x] = (float**)malloc(sizeof(float*) * layers[x].number);

			for (int y = 0; y < layers[x].number; y++)
			{
				weights[x][y] = (float*)malloc(sizeof(float) * layers[x + 1].number);
				delta_weights[x][y] = (float*)malloc(sizeof(float) * layers[x + 1].number);
				momentum1D[x][y] = (float*)malloc(sizeof(float) * layers[x + 1].number);
				rmsp1D[x][y] = (float*)malloc(sizeof(float) * layers[x + 1].number);

				biases[x][y] = 0;
				delta_biases[x][y] = 0;
				for (int z = 0; z < layers[x + 1].number; z++)
				{
					weights[x][y][z] = WeightInitialization(layers[x].number, layers[x + 1].number);
					delta_weights[x][y][z] = 0.0;
					momentum1D[x][y][z] = 0.0;
					rmsp1D[x][y][z] = 0.0;
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
	biases[layers.size() - 1] = (float*)malloc(sizeof(float) * layers.back().number);
	delta_biases[layers.size() - 1] = (float*)malloc(sizeof(float) * layers.back().number);
	for (int j = 0; j < layers.back().number; j++)
	{
		biases[layers.size() - 1][j] = (float)rand() / (float)RAND_MAX;
		delta_biases[layers.size() - 1][j] = 0;
	}

}

void Network::InitializeValueMatrices(int batchsize)
{
	for (int t = 0; t < batchsize; t++)
	{
		//Row 0
		vector<vector<vector<float>>> temp2Dsetinit;

		vector<vector<float>>temp2D;
		temp2Dsetinit.push_back(temp2D);

		layers[0].values2D.push_back(temp2Dsetinit);
		layers[0].values2Dderivative.push_back(temp2Dsetinit);
		layers[0].pre_activation_values2D.push_back(temp2Dsetinit);

		for (int i = 0; i < layers.size(); i++)
		{
			vector<float> temp(layers[i].number);
			layers[i].values.push_back(temp);
			layers[i].pre_activation_values.push_back(temp);
			layers[i].softmaxsum.push_back(0);

			vector<vector<vector<float>>> temp2Dset;
			for (int j = 0; j < layers[i].kernelnumber; j++)
			{
				vector<vector<float>>temp2D;
				temp2Dset.push_back(temp2D);
			}
			layers[i].values2D.push_back(temp2Dset);
			layers[i].values2Dderivative.push_back(temp2Dset);
			layers[i].pre_activation_values2D.push_back(temp2Dset);
		}

		//Errors
		vector<float> temp(layers.back().number);
		errors.push_back(temp);
		derrors.push_back(temp);
	}
}

void Network::InitializePredictedMatrix(vector<vector<float>>* predicted)
{
	for (int j = 0; j < totalepochs; j++)
		for (int i = 0; i < totalinputsize; i++)
		{
			vector<float> temp(layers.back().number);
			predicted->push_back(temp);
		}
}

float Network::WeightInitialization(int fan_in, int fan_out)
{
	static default_random_engine generator;
	normal_distribution<float> glorotdist(0, 2 / (float)(fan_in + fan_out));
	normal_distribution<float> hedist(0, 2 / (float)(fan_in));
	float val;

	switch (WeightInitializer)
	{
	case Random:
		val = rand() / (float)RAND_MAX;
		return (rand() / (float)RAND_MAX > 0.5) ? val : -val;
	case Glorot:
		return glorotdist(generator);
	case He:
		return hedist(generator);
	}

	return -1;
}

//Compiling
void Network::Compile(string type)
{
	if (!type.compare("Stochastic"))
	{
		gradient_descent_type = Stochastic;
		batchsize = 1;
	}
	else if (!type.compare("Mini_Batch"))
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
float Network::DActivation(float x, int i)
{
	static float temp;
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

	return -1;
}

float Network::Activation(float x, int i)
{
	switch (layers[i].type)
	{
	case Layer::Sigmoid:
		return 1 / (float)(1 + exp(-x));
	case Layer::Linear:
		return x;
	case Layer::Relu:
		return (x > 0) ? x : 0;
	case Layer::LeakyRelu:
		return (x > 0) ? x : layers[i].parameters.LeakyReluAlpha * x;
	case Layer::Tanh:
		return tanh(x);
	case Layer::Softmax:
		return exp(x) / (float)layers[i].softmaxsum[0];
	}

	return -1;
}

//Errors
void Network::ErrorCalculation(int samplenum, vector<float>& actualvalue)
{
	static float avgerror = 0;
	static int counter = 0;
	static int totalcounter = 0;

	for (int i = 0; i < layers.back().number; i++)
	{
		float error = 0;
		float val = layers.back().values[samplenum][i];

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

		errors[samplenum][i] = error;
		derrors[samplenum][i] = DActivation(layers.back().pre_activation_values[samplenum][i], (int)layers.size() - 1) * DError(layers.back().values[samplenum][i], actualvalue[i], i);

		avgerror += errors[samplenum][i];
	}


	counter++;
	totalcounter++;
}

void Network::AccumulateErrors()
{
	float errorsum = 0;
	for (int i = 0; i < layers.back().number; i++)
	{
		for (int j = 1; j < errors.size(); j++)
			errors[0][i] += errors[j][i];

		switch (model_loss_type)
		{
		case Root_Mean_Squared:
			errors[0][i] = sqrt(errors[0][i] / (float)(2 * batchsize));
			errorsum += errors[0][i];
			break;
		case Mean_Squared:
		case Mean_Absolute:
		case Mean_Biased:
		case Binary_Cross_Entropy:
		case Categorical_Cross_Entropy:
			errorsum += errors[0][i];
			break;
		}
	}
	epochloss += errorsum / (float)batchsize;
}

float Network::DError(float predictedvalue, float actualvalue, int neuronnum)
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
		return (actualvalue - predictedvalue) / (float)(predictedvalue * (1 - predictedvalue));
	case Categorical_Cross_Entropy:
		return actualvalue / (float)predictedvalue;
	}

	return -1;
}

void Network::CleanErrors()
{
	for (int i = 0; i < layers.back().number; i++)
	{
		for (int j = 0; j < (int)errors.size(); j++)
		{
			errors[j][i] = 0;
			derrors[j][i] = 0;
		}
	}
}

//Params
void Network::LeakyReluParameters(float i, float a)
{
	if (i<1 || i>layers.size())
	{
		cout << "Layer Number Out Of Range" << endl;
		return;
	}
	layers[i - 1].parameters.LeakyReluAlpha = a;
}

void Network::ShowTrainingStats(vector<vector<float>>* inputs, vector<vector<float>>* actual, int i)
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
		cout << errors[0][j] << " ";
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
void Network::Predict(vector<vector<float>>* input, vector<vector<float>>* predicted)
{
	input1D = true;

	vector<vector<float>> tempinput = *input;
	vector<vector<vector<float>>> inputs = { tempinput };
	Predict(&inputs, predicted);
}

void Network::Predict(vector<vector<vector<float>>>* input, vector<vector<float>>* predicted)
{
	if (Model_Mode != eval)
		Model_Mode = eval;

	CleanErrors();

	int totalsize;
	if (input1D)
	{
		totalsize = (int)(*input)[0].size();
	}
	else
	{
		totalsize = (int)input->size();
	}

	
	for (int i = 0; i < totalsize; i++)
	{
		//Taking the sample
		vector<vector<float>> sample;

		//Check for 1D or 2D
		if (input1D)
		{
			sample = { (*input)[0][i] };
		}
		else
		{
			sample = (*input)[i];
		}

		vector<float> actual(layers.back().number);

		//Calculate Prop
		ForwardPropogation(0, sample, actual);

		//Get Prediction
		vector<float>prediction = layers.back().values[0];
		predicted->push_back(prediction);

		//Clean Errors
		CleanErrors();
	}
	
}


//Training
void Network::Train(vector<vector<float>>* inputs, vector<vector<float>>* actual, int epochs, string losstype)
{
	vector<vector<float>> tempinput = *inputs;
	vector<vector<vector<float>>> input = { tempinput };
	Train(&input, actual, epochs, losstype);
}

void Network::Train(vector<vector<vector<float>>>* inputs, vector<vector<float>>* actual, int epochs, string loss)
{
	if (Model_Mode != train)
		Model_Mode = train;

	totalinputsize = (int)inputs->size();
	if (input1D)
		totalinputsize = (int)(*inputs)[0].size();

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

		for (int l = 0; l < epochs; l++)
		{
			if (verbosetraining)
				cout << "\n\nEpoch: " << l + 1 << "\n";

			for (int j = 0; j < batchnum; j++)
			{
				for (int i = 0; i < batchsize; i++)
				{
					//Taking the sample
					vector<vector<float>> sample;

					//Check for 1D or 2D
					if (input1D)
					{
						sample = { (*inputs)[0][j * batchsize + i] };
					}
					else
					{
						sample = (*inputs)[j * batchsize + i];
					}

					//Actual Value
					vector<float> actualvalue = (*actual)[j * batchsize + i];

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
				while (threadcounter < batchsize) {
				};

				//Reset Counters
				threadcounter = 0;
				batchcounter++;

				//Backprop
				AccumulateErrors();

				UpdateParameters();

				if (cleanerrors)
					CleanErrors();
			}

			//Print Loss
			if (verbosetraining || verboseloss)
				cout << "\nLoss: " << epochloss/(float)batchnum;

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