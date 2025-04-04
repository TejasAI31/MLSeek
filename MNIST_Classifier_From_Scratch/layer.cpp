#include "layer.h"

Layer::Layer(int num, std::string neuronname)
{

	if (!neuronname.compare("Sigmoid"))
		type = Sigmoid;
	else if (!neuronname.compare("Linear"))
		type = Linear;
	else if (!neuronname.compare("Relu"))
		type = Relu;
	else if (!neuronname.compare("LRelu"))
		type = LeakyRelu;
	else if (!neuronname.compare("Tanh"))
		type = Tanh;
	else if (!neuronname.compare("Softmax"))
		type = Softmax;
	else if (!neuronname.compare("Input"))
		type = Input;
	else if (!neuronname.compare("Input2D"))
		type = Input2D;
	else if (!neuronname.compare("Pool2D"))
	{
		type = Pool2D;
		padding = num;
		return;
	}
	else
		type = Sigmoid;

	number = num;
	neurontype = neuronname;

}

Layer::Layer(int kernalnumber, int size, std::string layertype)
{
	type = Conv;
	neurontype = "Relu";
	kernelnumber = kernalnumber;
	kernelsize = size;
}

Layer::Layer(int kernalnumber, int size, int kerneldilation, std::string layertype)
{
	type = Conv;
	neurontype = "Relu";
	kernelnumber = kernalnumber;
	kernelsize = size;
	dilation = kerneldilation;
}

Layer::Layer(int kernalnumber, int size, int kerneldilation, int stridenum, std::string layertype)
{
	type = Conv;
	neurontype = "Relu";
	kernelnumber = kernalnumber;
	kernelsize = size;
	dilation = kerneldilation;
	stride = stridenum;
}

Layer::Layer(double drop, std::string layertype)
{
	if (!layertype.compare("Dropout"))type = Dropout;
	else
		cout << "No Such Layer" << endl;

	dropout = drop;
	if (drop == 0)
		dropout = 0.0;
}

//