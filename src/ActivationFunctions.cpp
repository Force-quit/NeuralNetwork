//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2021  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------


#include "ActivationFunctions.h"

namespace bpn {

	ActivationFunction* ActivationFunction::deserialize(const std::string& s)
	{
		if (s.find("Sigmoid(") != std::string::npos)
		{
			double lambda = atof(s.substr(8, s.size() - 9).c_str());
			return new Sigmoid(lambda);
		}
		else if (s.find("ReLU") != std::string::npos)
		{
			return new ReLU();
		}
		else if (s.find("LeakyReLU") != std::string::npos)
		{
			return new LeakyReLU();
		}
		throw std::runtime_error("Unknown activation function");
	}

}
