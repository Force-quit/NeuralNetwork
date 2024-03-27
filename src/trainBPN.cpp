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
#include "StopFileWatcher.h"

#if _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 4702)
#endif

#include "cmdParser.h"
#include "configFileParser.h"

#if _MSC_VER
#pragma warning(pop)
#endif

const char* DEFAULTCONFIGURATIONFILENAME = "config.txt";

// Operators from "vectorstream.h"
using bpn::operator<<;
using bpn::operator>>;

//-------------------------------------------------------------------------

void configureCmdParser(cli::Parser& cmdParser)
{
	//
	// Definition of the command line arguments
	//
	std::stringstream ss;
	ss << "usage: trainBPN [--help] [configurationFile] [<args>]\n\n"
		"  There are three ways to run this program:\n"
		"  1. Without CLI parameters\n"
		"     Default configuration file `" << DEFAULTCONFIGURATIONFILENAME << "` is used.\n"
		"  2. Exactly ONE parameter\n"
		"     This unique parameter must be the absolute or relative path to a configuraion file\n"
		"  3. Using CLI parameters and no configuration file\n"
		"\n";
	cmdParser.setHelpText(ss.str());
	cmdParser.set_optional<std::string>("h", "help", "", "Path to training data file.");
	cmdParser.set_optional<std::string>("d", "datafile", "", "Path to training data file.");
	cmdParser.set_optional<std::string>("f", "format", "numberList", "Format of the training data file.\n"
		"     numberList, Comma separated list of numbers. The first ones are the\n"
		"                 values of the input nodes and the following ones are the\n"
		"                 expected values of the output nodes. The number of values\n"
		"                 must fit the number of intput/output nodes.\n"
		"     binary,     First 4 bytes is the number of data, next 4 bytes is the\n"
		"                 number of input values in each data, next 4 bytes is the\n"
		"                 number of output values in each data. Then, each byte is\n"
		"                 one data value. Each value is converted to a float by\n"
		"                 dividing it by 255."
	);
	cmdParser.set_optional<std::string>("l", "layers", "[]", "Comma separated list of the layers sizes (e.g. 16,4,4,3 or 1,1,1).\n     First layer is the input neurons.\n     Last layer is the output layer.");
	cmdParser.set_optional<std::string>("i", "import", "", "Import neural network from file before training. ( \"-\" stands for stdin)");
	cmdParser.set_optional<std::string>("e", "export", "", "Export neural network to file after training. ( \"-\" stands for stdout)");

	cmdParser.set_optional<std::string>("s", "activation", "Sigmoid(1)", "The actiation function. Available options are:\n"
		"     Sigmoid(k), Logistic function with stepness ``k``.\n"
		"     ReLU,       Rectified linear unit.\n"
		"     LeakyReLY,  Leaky ReLU, like ReLU but with small gradiant (1/100)\n"
		"                 when the unit is not active.");
	cmdParser.set_optional<uint64_t>("m", "maxEpoch", 100, "Maximum num of iterations (-1 stands for 2^64-1).");
	cmdParser.set_optional<double>("r", "learningRate", 0.01, "Multiplicative coefficient on error gradient");
	cmdParser.set_optional<double>("mom", "momentum", 0.9, "Multiplicative coefficient applied on previous error delta when non using batch learning.");
	cmdParser.set_optional<bool>("b", "batchLearning", false, "Use batch learning (1 : yes, 0 : no).");
	cmdParser.set_optional<double>("a", "accuracy", 95, "Desired accuracy. Training stops when the desired accuracy is obtained.");
	cmdParser.set_optional<std::string>("L", "labels", "", "Labels for output nodes. Comma separated list of work without white spaces.\n"
		"   Only for new networks and is only used with the GUI visualization tool.");
	cmdParser.set_optional<int32_t>("v", "verbosity", 1, "Verbosity level.\n     Level 0: quiet mode.\n     Level 1: prints training evolution.\n     Level 2: prints NN at initialization and at the end.\n     Level 3: prints NN at every iteration of the learning phase.");
}


int main(int argc, char* argv[])
{
	StopFileWatcher::init("delete_this_to_stop.txt");

	//
	// Parameters
	//
	// They are either defined in the configuration file or by the command line
	// arguments.
	//
	// Note that command line arguments override what's written in the
	// configuration file.
	// 
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



	// There are three ways to run this program
	// 1. Without CLI arguments
	//    In this case, the default configuration file is used.
	// 2. Exactly ONE argument 
	//    This unique argument must be the configuraion file.
	// 3. Using CLI arguments and no configuration file.

	std::string configurationFile;
	bool useConfigurationFile = false;
	// If no arguments are given then read the default config file.
	if (argc == 1)
	{
		std::cout << "No arguments --> reading default configuration file `"
			<< DEFAULTCONFIGURATIONFILENAME << "`.\n" << std::endl;
		configurationFile = DEFAULTCONFIGURATIONFILENAME;
		useConfigurationFile = true;
	}
	else if (argc == 2 && argv[1][0] != '-')
	{
		configurationFile = argv[1];
		useConfigurationFile = true;
	}
	if (useConfigurationFile)
	{
		//
		// Read data from configuration file parser
		//
		//
		cfp::Parser cfParser(configurationFile);
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
	}
	else
	{
		cli::Parser cmdParser(argc, argv);
		configureCmdParser(cmdParser);
		if (!cmdParser.run())
		{
			std::cout << "Invalid command line arguments." << std::endl;;
			return 1;
		}

		//
		// Get values from the command line parser
		// 
		trainingDataPath = cmdParser.get<std::string>("d");
		format = cmdParser.get<std::string>("f");
		layers = cmdParser.get<std::string>("l");
		importFile = cmdParser.get<std::string>("i");
		exportFile = cmdParser.get<std::string>("e");
		activationFunction = cmdParser.get<std::string>("s");
		maxEpoch = cmdParser.get<uint64_t>("m");
		learningRate = cmdParser.get<double>("r");
		momentum = cmdParser.get<double>("mom");
		batchLearning = cmdParser.get<bool>("b");
		accuracy = cmdParser.get<double>("a");
		labels = cmdParser.get<std::string>("L");
		verbosity = cmdParser.get<int32_t>("v");
	}


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

