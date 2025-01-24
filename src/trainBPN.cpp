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
#include "StopWatcher.h"

#include "configFileParser.h"

// Operators from "vectorstream.h"
using bpn::operator<<;
using bpn::operator>>;

int main(int argc, char* argv[])
{
	StopWatcher::init("delete_this_to_stop.txt");

	std::string trainingDataPath;
	std::string format;
	std::string layers;
	std::string importFile;
	std::string exportFile;
	std::string activationFunction;
	uint64_t    maxEpoch;
	double      learningRate;
	double      momentum;
	bool        batchLearning;
	double      accuracy;
	std::string labels;
	int32_t     verbosity;

	cfp::Parser cfParser("config.txt");
	try {
		if (!cfParser.run())
		{
			exit(-1);
		}
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << e.what() << '\n';
		std::cerr << "See 'trainBPN --help' for help." << std::endl;
		exit(-1);
	}
	trainingDataPath = cfParser.get<std::string>("datafile");
	format = cfParser.get<std::string>("format", "numberList");
	layers = cfParser.get<std::string>("layers", "[]");
	importFile = cfParser.get<std::string>("import", "");
	exportFile = cfParser.get<std::string>("export", "");
	activationFunction = cfParser.get<std::string>("activation", "Sigmoid(1)");
	maxEpoch = cfParser.get<uint64_t>("maxEpoch", 100);
	learningRate = cfParser.get<double>("learningRate", 0.01);
	momentum = cfParser.get<double>("momentum", 0.9);
	batchLearning = cfParser.get<bool>("batchLearning", 0);
	accuracy = cfParser.get<double>("accuracy", 95);
	labels = cfParser.get<std::string>("labels", "");
	verbosity = cfParser.get<int32_t>("verbosity", 1);
	

	// If the user wants to create a new neural network then he must specify it's
	// shape (number of layers and their sizes). Otherwise, an existing neural
	// network must be imported. 
	// Verify that the user has specified at least one of the two.
	if (layers == "[]" && importFile == "")
	{
		std::cerr << "At least one parameter among -l or -i must be specified."
			<< " (See help for more impormations)" << std::endl;
		exit(1);
	}

	// Verify that the user has specified not provided both.
	if (layers != "[]" && importFile != "")
	{
		std::cerr << "Only one parameter among -l and -i may be specified. (See help for more impormations)" << std::endl;
		exit(1);
	}

	// Validation of the input format : eigher `binary` or `numberList`.
	bpn::DataReader::Format inputDataFormat;
	if (format.compare("binary") == 0)
		inputDataFormat = bpn::DataReader::Format::binary;
	else if (format.compare("numberList") == 0)
		inputDataFormat = bpn::DataReader::Format::numberList;
	else
		throw std::runtime_error("Invalid format for input data. For more help use --help or -h.");

	// Create neural network
	bpn::Network* nn = NULL;

	if (layers.compare("[]") != 0) // new network
	{
		// Read layers sizes. The first layer is the input layer, the last layer
		// is the output. All other layers are hidden.
		std::vector<int> layerSizes;
		std::stringstream ss(layers);
		ss >> layerSizes;

		nn = new bpn::Network(layerSizes, bpn::ActivationFunction::deserialize(activationFunction), labels);
	}
	else if (importFile.compare("") != 0) // existing network
	{
		if (importFile.compare("-") == 0) // read on stdin
		{
			nn = new bpn::Network(std::cin);
		}
		else
		{
			// import network from text file
			std::fstream fs;
			fs.open(importFile, std::fstream::in);
			nn = new bpn::Network(fs);
		}
	}

	assert(nn != NULL);

	if (verbosity >= 2)
	{
		std::cout << *nn << std::endl;
	}

	// 
	// Data from the data file is loaded into memory
	bpn::DataReader dataReader(trainingDataPath,
		nn->getNumInputs(),
		nn->getNumOutputs(),
		inputDataFormat,
		verbosity);

	std::cout << "Reading data from file `" << trainingDataPath << "`" << std::endl;
	bpn::TrainingData data;
	if (!dataReader.readTraningData(data))
	{
		std::cerr << "Data error" << std::endl;
		return 1;
	}
	if (verbosity >= 1)
	{
		int nbTraining = data.m_trainingSet.size();
		int nbGeneralization = data.m_generalizationSet.size();
		int nbValidation = data.m_validationSet.size();
		std::cout << "Training data read successfully:\n";
		std::cout << "==========================================================================\n"
			<< " Input data file: " << trainingDataPath << "\n"
			<< " Read complete: " << nbTraining + nbGeneralization + nbValidation << " inputs loaded"
			<< " (" << nbTraining << " for training, "
			<< nbGeneralization << " for generalization and "
			<< nbValidation << " for validation)\n"
			<< "=========================================================================="
			<< std::endl;
	}


	//
	// Create neural network trainer
	// 
	bpn::NetworkTrainer::Settings trainerSettings;
	trainerSettings.m_learningRate = learningRate;
	trainerSettings.m_momentum = momentum;
	trainerSettings.m_useBatchLearning = batchLearning;
	trainerSettings.m_maxEpochs = maxEpoch;
	trainerSettings.m_desiredAccuracy = accuracy;
	trainerSettings.m_verbosity = verbosity;

	bpn::NetworkTrainer trainer(trainerSettings, nn);

	//
	// All the real work is done here
	// 
	trainer.Train(data);
	// It's all over now

	if (verbosity >= 2)
	{
		std::cout << *nn << std::endl;
	}

	// 
	// If required, the network is exported
	if (exportFile.compare("") != 0)
	{
		if (exportFile.compare("-") == 0)
		{
			// export to stdout
			std::cout << nn->serialize() << std::endl;
		}
		else
		{
			// export to a text file
			std::fstream fs;
			fs.open(exportFile, std::fstream::out);
			fs << nn->serialize() << std::endl;
		}
	}

	// Useless but hey, best practices are best practices. 
	delete nn;
	return 0;
}

