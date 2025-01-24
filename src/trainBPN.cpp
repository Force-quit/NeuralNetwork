//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// Copyright (C) 2025 Émile Laforce
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "StopWatcher.h"
#include "ConfigFileParser.h"
#include <optional>
#include <print>

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>

#include "NeuralNetworkTrainer.h"
#include "DataReader.h"
#include "Matrix.h"
#include "vectorstream.h"

// Operators from "vectorstream.h"
using bpn::operator<<;
using bpn::operator>>;

int main()
{
	StopWatcher::init("delete_this_to_stop.txt");

	ConfigFileParser configParser("config.txt");

	if (std::optional<std::string> error(configParser.run()); error.has_value())
	{
		std::println(std::cerr, "Error: {}", error.value());
		return 1;
	}

	std::string trainingDataPath(configParser.get<std::string>("datafile"));
	std::string format(configParser.get<std::string>("format", "numberList"));
	std::string layers(configParser.get<std::string>("layers", "[]"));
	std::string exportFile(configParser.get<std::string>("export", ""));
	std::string activationFunction(configParser.get<std::string>("activation", "Sigmoid(1)"));
	std::string labels(configParser.get<std::string>("labels", ""));
	std::uint64_t maxEpoch(configParser.get<std::uint64_t>("maxEpoch", 100));
	double learningRate{ configParser.get<double>("learningRate", 0.01) };
	double momentum{ configParser.get<double>("momentum", 0.9) };
	bool batchLearning{ configParser.get<bool>("batchLearning", 0) };
	double accuracy{ configParser.get<double>("accuracy", 95) };
	std::uint16_t verbosity{ configParser.get<std::uint16_t>("verbosity", 1) };

	// Validation of the input format : eigher `binary` or `numberList`.
	bpn::DataReader::Format inputDataFormat;
	if (format.compare("binary") == 0)
	{
		inputDataFormat = bpn::DataReader::Format::binary;
	}
	else if (format.compare("numberList") == 0)
	{
		inputDataFormat = bpn::DataReader::Format::numberList;
	}

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