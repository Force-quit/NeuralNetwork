//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>

#include "NeuralNetworkTrainer.h"
#include "DataReader.h"
#include "Matrix.h"
#include "vectorstream.h"

#if _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 4702)
#endif

#include "cmdParser.h"

#if _MSC_VER
#pragma warning(pop)
#endif

// Operators from "vectorstream.h"
using bpn::operator<<;
using bpn::operator>>;

//-------------------------------------------------------------------------

int main(int argc, char* argv[])
{
	//bpn::Network nin( std::cin );
	//std::cout << nin << std::endl;
	//exit(0);



	cli::Parser cmdParser(argc, argv);
	cmdParser.set_required<std::string>("f", "format", "Format of the training data file.\n"
		"     numberList, Comma separated list of numbers. The first ones are the\n"
		"                 values of the input nodes and the following ones are the\n"
		"                 expected values of the output nodes. The number of values\n"
		"                 must fit the number of intput/output nodes.\n"
		"     text,       Format is \"<text>\",<expectedOutput> where the expected\n"
		"                 output is given a comma separated list of integers. Size of\n"
		"                 the inputs may not equal the number of input nodes. If less\n"
		"                 then extra zeros are added. If more, extra characters are\n"
		"                 ignored.");
	cmdParser.set_required<std::string>("i", "import", "Import neural network from file before training. ( \"-\" stands for stdin)");
	cmdParser.set_optional<int32_t>("v", "verbose", 1, "Verbosity level.\n     Level 0: quiet mode.\n     Level 1: prints training evolution.\n     Level 2: prints NN at initialization and at the end.\n     Level 3: prints NN at every iteration of the learning phase.");

	if (!cmdParser.run())
	{
		std::cout << "Invalid command line arguments";
		return 1;
	}

	std::string const format = cmdParser.get<std::string>("f");
	std::string const importFile = cmdParser.get<std::string>("i");
	int32_t verbosity = cmdParser.get<int32_t>("v");

	bpn::InputDataFormat inputDataFormat;
	if (format.compare("binary") == 0)
		inputDataFormat = bpn::binary;
	if (format.compare("numberList") == 0)
		inputDataFormat = bpn::numberList;
	else
		throw std::runtime_error("Invalid format for input data. For more help use --help or -h.");

	std::fstream fs;
	fs.open(importFile, std::fstream::in);
	bpn::Network* nn = new bpn::Network(fs);


	if (verbosity >= 2)
	{
		std::cout << *nn << std::endl;
	}

	bpn::DataReader dataReader("-",
		nn->getNumInputs(),
		nn->getNumOutputs(),
		inputDataFormat,
		verbosity);



	while (dataReader.hasMoreData())
	{
		std::vector<double> inputData;
		dataReader.readOneInputData(inputData);
		nn->Evaluate(inputData);
		std::cout << nn->getUnClampedOutput() << std::endl;
	}


	delete nn;
	return 0;
}

