//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// Basic Gradient Descent NN Trainer with Momentum and Batch Learning

#pragma once

#include "NeuralNetwork.h"
#include <fstream>

namespace bpn
{
	struct TrainingEntry
	{
		std::vector<double>         m_inputs;
		std::vector<int32_t>        m_expectedOutputs;
	};

	typedef std::vector<TrainingEntry> TrainingSet;

	struct TrainingData
	{
		TrainingSet m_trainingSet;
		TrainingSet m_generalizationSet;
		TrainingSet m_validationSet;
	};

	//-------------------------------------------------------------------------

	class NetworkTrainer
	{
	public:

		struct Settings
		{
			// Learning params
			double      m_learningRate;
			double      m_momentum;
			bool        m_useBatchLearning;

			// Stopping conditions
			uint64_t    m_maxEpochs;
			double      m_desiredAccuracy;

			// Verbosity
			int32_t     m_verbosity;
		};

	public:

		NetworkTrainer(Settings const& settings, Network* pNetwork);

		void Train(TrainingData const& trainingData);

	private:

		inline double getOutputErrorGradient(double desiredValue, const Neuron& outputNeuron) const
		{
			// TODO : mean square error is hard coded here so we have 
			// a factor : desiredValue - outputNeuron.value;
			double derivative = m_pNetwork->m_sigma->evalDerivative(
				outputNeuron.activation, outputNeuron.value);
			return derivative * (desiredValue - outputNeuron.value);
			//return outputValue * ( 1.0 - outputValue ) * ( desiredValue - outputValue ); 
		}
		//double GetHiddenErrorGradient( int32_t hiddenIdx ) const;
		double getErrorGradient(int32_t layer, int32_t index) const;

		void RunEpoch(TrainingSet const& trainingSet);
		void Backpropagate(std::vector<int32_t> const& expectedOutputs);
		void UpdateWeights();

		void GetSetAccuracyAndMSE(TrainingSet const& trainingSet, double& accuracy, double& mse) const;

	private:

		Network* m_pNetwork;             // Network to train

		// Training settings
		double                            m_learningRate;         // Adjusts the step size of the weight update
		double                            m_momentum;             // Improves performance of stochastic 
		// learning (don't use for batch)

		double                            m_desiredAccuracy;      // Target accuracy for training
		uint64_t                          m_maxEpochs;            // Max number of training epochs
		bool                              m_useBatchLearning;     // Should we use batch learning

		// m_deltas[i] : deltas from layer i to i+1
		std::vector<Matrix>               m_deltas;
		// m_errorGradients[i] error gradients on layer i
		std::vector< std::vector<double> > m_errorGradients;

		uint64_t                          m_currentEpoch;             // Epoch counter
		double                            m_trainingSetAccuracy;
		double                            m_validationSetAccuracy;
		double                            m_generalizationSetAccuracy;
		double                            m_trainingSetMSE;
		double                            m_validationSetMSE;
		double                            m_generalizationSetMSE;
		int32_t                           m_verbosity;

	};
}
