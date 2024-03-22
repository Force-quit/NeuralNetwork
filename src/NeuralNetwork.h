//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// Copyright (C) 2024  Émile Laforce
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer

#pragma once

#include "ActivationFunctions.h"
#include "Matrix.h"
#include "vectorstream.h"
#include <iostream>
#include <stdint.h>
#include <vector>

namespace bpn
{
	struct Neuron
	{
		Neuron() noexcept : activation{}, value{} {}
		Neuron(double a, double v) noexcept : activation{ a }, value{ v } {}
		double activation;
		double value; // = Sigma(activation)
		friend std::ostream& operator<<(std::ostream& os, const bpn::Neuron& n);
	};

	class Network
	{
		friend class NetworkTrainer;

		//-------------------------------------------------------------------------

		inline static int32_t ClampOutputValue(double x)
		{
			if (x < 0.1) return 0;
			else if (x > 0.9) return 1;
			else return -1;
		}

	public:

		Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma, const std::string labels);
		Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma);
		Network(std::istream& is);

		std::vector<int32_t> const& Evaluate(std::vector<double> const& input);

		void saveToFile(const char* filename) const;

		std::string serialize() const;
		void deserialize(std::istream& is);

		inline int32_t getNumInputs() const
		{
			return m_numInputs;
		}

		inline int32_t getNumOutputs() const
		{
			return m_numOutputs;
		}

		inline int32_t getNumLayers() const
		{
			return m_layers.size();
		}

		inline const std::vector<int>& getLayerSizes() const
		{
			return m_layerSizes;
		}

		inline double getValue(int layer, int n) const
		{
			return m_layers[layer][n].value;
		}

		inline const std::string activationFunctionName() const
		{
			return m_sigma->serialize();
		}

		inline const std::vector<int32_t>& getOutput() const
		{
			return m_clampedOutputs;
		}

		inline const std::vector<double> getUnClampedOutput() const
		{
			std::vector<double> t;
			for (int i = 0; i < m_numOutputs; ++i)
			{
				t.push_back(m_outputNeurons->at(i).value);
			}
			return t;
		}

	private:
		void loadFromFile(const char* filename);
		void InitializeNetwork();
		void InitializeWeights();

	private:

		int32_t                     m_numLayers;       // number of layers including input and output (min 3)
		int32_t                     m_numInputs;       // number of neurons on the input layer
		int32_t                     m_numOutputs;      // number of neurons on the output layer
		int32_t                     m_numOnLastHidden; // number of neurons on the last hidden layer
		std::vector<int>            m_layerSizes;      // m_layerSizes[i] is the number of neurons on the i-th layer.
		using Layer = std::vector<Neuron>;
		std::vector<Layer>          m_layers;           // m_layers[i] is the i-th layer
		Layer* m_inputNeurons;      // &m_layers[0]
		Layer* m_lastHiddenNeurons; // &m_layers[-2] (python notations)
		Layer* m_outputNeurons;     // &m_layers[-1]
		std::vector<int32_t>        m_clampedOutputs;
		// m_wrigntsByLayer[i] is the matrix of weights from layer i to layer i+1
		std::vector<Matrix>         m_weightsByLayer;
		const ActivationFunction* m_sigma;

		std::string                 m_labels;          // labels for the output nodes

	public:

		std::string selfDisplay() const;
		friend std::ostream& operator<<(std::ostream& os, const bpn::Network& n);
	};

}


