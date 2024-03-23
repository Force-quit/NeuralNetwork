//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2021  Xavier Provençal
// Copyright (C) 2024 Émile Laforce
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "ActivationFunctions.h"
#include <ranges>
#include <stdexcept>

namespace bpn {

	ActivationFunction* ActivationFunction::deserialize(std::string_view s)
	{
		if (s.contains("Sigmoid("))
		{
			auto is_digit = [](char c) { return std::isdigit(c); };
			double lambda = *std::ranges::find_if(s, is_digit) - '0';
			return new Sigmoid(lambda);
		}
		else if (s.contains("ReLU"))
		{
			return new ReLU();
		}
		else if (s.contains("LeakyReLU"))
		{
			return new LeakyReLU();
		}

		throw std::runtime_error("Unknown activation function");
	}

}
