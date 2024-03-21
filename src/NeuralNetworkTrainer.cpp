//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "NeuralNetworkTrainer.h"
#include "StopFileWatcher.h"
#include <string.h>
#include <assert.h>
#include <iostream>

//-------------------------------------------------------------------------

namespace bpn
{
	NetworkTrainer::NetworkTrainer(Settings const& settings, Network* pNetwork)
		: m_pNetwork(pNetwork)
		, m_learningRate(settings.m_learningRate)
		, m_momentum(settings.m_momentum)
		, m_desiredAccuracy(settings.m_desiredAccuracy)
		, m_maxEpochs(settings.m_maxEpochs)
		, m_useBatchLearning(settings.m_useBatchLearning)
		, m_currentEpoch(0)
		, m_trainingSetAccuracy(0)
		, m_validationSetAccuracy(0)
		, m_generalizationSetAccuracy(0)
		, m_trainingSetMSE(0)
		, m_validationSetMSE(0)
		, m_generalizationSetMSE(0)
		, m_verbosity(settings.m_verbosity)
	{
		assert(pNetwork != nullptr);
		for (int32_t i = 0; i < m_pNetwork->m_numLayers - 1; ++i)
		{
			// Generate the delta matrix from later i to layer i+1
			// add one to actualLayerSize for bias
			int32_t actualLayerSize = m_pNetwork->m_layerSizes[i];
			int32_t nextLayerSize = m_pNetwork->m_layerSizes[i + 1];
			m_deltas.push_back(Matrix(actualLayerSize + 1, nextLayerSize, 0.0));
		}

		// m_errorGradients[0] is not used... dummy value to fill the spot
		m_errorGradients.push_back(std::vector<double>());
		for (int32_t i = 1; i < m_pNetwork->m_numLayers; ++i)
		{
			int layerSize = m_pNetwork->m_layerSizes[i];
			if (i < m_pNetwork->m_numLayers - 1)
			{
				layerSize += 1; // add one for bias
			}
			m_errorGradients.push_back(std::vector<double>());
			m_errorGradients[i].resize(layerSize, 0.0);
		}

	}

	void NetworkTrainer::Train(TrainingData const& trainingData)
	{
		// Reset training state
		m_currentEpoch = 0;
		m_trainingSetAccuracy = 0;
		m_validationSetAccuracy = 0;
		m_generalizationSetAccuracy = 0;
		m_trainingSetMSE = 0;
		m_validationSetMSE = 0;
		m_generalizationSetMSE = 0;

		// Print header
		//-------------------------------------------------------------------------

		if (m_verbosity >= 1)
		{
			std::cout << "\n\n"
				<< "=========================================================================="
				<< std::endl
				<< " Learning Rate: " << m_learningRate
				<< ", Momentum: " << m_momentum
				<< ", Max Epochs: " << m_maxEpochs << std::endl
				<< " Target Accucaty: " << m_desiredAccuracy
				<< ", Layers Sizes: " << m_pNetwork->m_layerSizes << std::endl
				<< " Activation function: " << m_pNetwork->activationFunctionName() << std::endl
				<< "=========================================================================="
				<< std::endl << std::endl;
		}

		// Train network using training dataset for training and generalization dataset for testing
		//--------------------------------------------------------------------------------------------------------

		while ((!StopFileWatcher::doesTheStopFileTellsMeToStop()) &&
			((m_trainingSetAccuracy < m_desiredAccuracy
				|| m_generalizationSetAccuracy < m_desiredAccuracy)
				&& m_currentEpoch < m_maxEpochs)
			)
		{
			// Use training set to train network
			RunEpoch(trainingData.m_trainingSet);

			// Get generalization set accuracy and MSE
			GetSetAccuracyAndMSE(trainingData.m_generalizationSet,
				m_generalizationSetAccuracy,
				m_generalizationSetMSE);

			if (m_verbosity >= 1)
			{
				std::cout << std::fixed << std::setprecision(6)
					<< "Epoch: " << m_currentEpoch
					<< " Training Set Accuracy: " << m_trainingSetAccuracy
					<< "%, MSE: " << m_trainingSetMSE
					<< ". Generalization Set Accuracy:" << m_generalizationSetAccuracy
					<< "%, MSE: " << m_generalizationSetMSE << std::endl;
			}

			m_currentEpoch++;
		}

		// Get validation set accuracy and MSE
		GetSetAccuracyAndMSE(trainingData.m_validationSet, m_validationSetAccuracy, m_validationSetMSE);

		// Print validation accuracy and MSE
		if (m_verbosity >= 1)
		{
			std::cout << std::endl << "Training Complete!!! - > Elapsed Epochs: " << m_currentEpoch << std::endl;
			std::cout << " Validation Set Accuracy: " << m_validationSetAccuracy << std::endl;
			std::cout << " Validation Set MSE: " << m_validationSetMSE << std::endl << std::endl;
		}
	}

	double NetworkTrainer::getErrorGradient(int32_t layer, int32_t index) const
	{
		assert(layer >= 1); // no error on input
		assert(layer <= m_pNetwork->m_numLayers - 2); // output layer is computed differently

		// Get sum of ``layer[i] --> layer[i+1] weights`` * layer[i+1] error dradients
		double weightedSum = 0;
		int32_t numOnNextLayer = m_pNetwork->m_layerSizes[layer + 1];
		for (auto nextLayerIdx = 0; nextLayerIdx < numOnNextLayer; ++nextLayerIdx)
		{
			weightedSum += m_pNetwork->m_weightsByLayer[layer](index, nextLayerIdx)
				* m_errorGradients[layer + 1][nextLayerIdx];
		}

		// Return error gradient
		const Neuron& n = m_pNetwork->m_neurons[layer][index];
		double derivative = m_pNetwork->m_sigma->evalDerivative(n.activation, n.value);
		return derivative * weightedSum;
	}

	void NetworkTrainer::RunEpoch(TrainingSet const& trainingSet)
	{
		double incorrectEntries = 0;
		double MSE = 0;

		for (auto const& trainingEntry : trainingSet)
		{
			// Feed inputs through network and back propagate errors
			m_pNetwork->Evaluate(trainingEntry.m_inputs);

			Backpropagate(trainingEntry.m_expectedOutputs);

			// Check all outputs from neural network against desired values
			bool resultCorrect = true;
			for (int outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++)
			{
				if (m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx])
				{
					resultCorrect = false;
				}

				// Calculate MSE
				MSE += pow(((*m_pNetwork->m_outputNeurons)[outputIdx].value
					- trainingEntry.m_expectedOutputs[outputIdx]), 2);
			}

			if (!resultCorrect)
			{
				incorrectEntries++;
			}
		}

		// If using batch learning - update the weights
		if (m_useBatchLearning)
		{
			UpdateWeights();
		}

		// Update training accuracy and MSE
		m_trainingSetAccuracy = 100.0 - (incorrectEntries / trainingSet.size() * 100.0);
		m_trainingSetMSE = MSE / (m_pNetwork->m_numOutputs * trainingSet.size());

		if (m_verbosity >= 3)
		{
			std::cout << "----------------------------------------------------"
				<< *m_pNetwork
				<< std::endl;
		}
	}

	void NetworkTrainer::Backpropagate(std::vector<int32_t> const& expectedOutputs)
	{
		// Modify deltas between the last hidden layer and output layers
		//---------------------------------------------------------------------
		int32_t numLayers = m_pNetwork->m_numLayers;
		bpn::Network::Layer& lastHiddenNeurons = *m_pNetwork->m_lastHiddenNeurons;
		bpn::Network::Layer& outputNeurons = *m_pNetwork->m_outputNeurons;

		for (auto outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; ++outputIdx)
		{
			// Get error gradient for every output node
			m_errorGradients[numLayers - 1][outputIdx] = getOutputErrorGradient(
				(double)expectedOutputs[outputIdx],
				outputNeurons[outputIdx]);

			// For all nodes in the last hidden layer and bias neuron
			for (auto hiddenIdx = 0; hiddenIdx <= m_pNetwork->m_numOnLastHidden; ++hiddenIdx)
			{
				// Calculate change in weight
				if (m_useBatchLearning)
				{
					m_deltas[numLayers - 2](hiddenIdx, outputIdx) +=
						m_learningRate
						* lastHiddenNeurons[hiddenIdx].value
						* m_errorGradients[numLayers - 1][outputIdx];
				}
				else
				{
					m_deltas[numLayers - 2](hiddenIdx, outputIdx) =
						m_learningRate
						* lastHiddenNeurons[hiddenIdx].value
						* m_errorGradients[numLayers - 1][outputIdx]
						+ m_momentum * m_deltas[numLayers - 2](hiddenIdx, outputIdx);
				}
			}
		}

		//// Modify deltas between all other layers
		////--------------------------------------------------------------------
		// deltas[numLaters-2] have been computed, lets compute all others.
		for (int32_t layer = numLayers - 3; layer >= 0; --layer)
		{
			// ``next layer`` is (layer+1)-th layer
			// ``actual layer`` is layer-th layer
			for (auto nextIdx = 0; nextIdx < m_pNetwork->m_layerSizes[layer + 1]; nextIdx++)
			{
				// Get error gradient for every hidden node
				m_errorGradients[layer + 1][nextIdx] = getErrorGradient(layer + 1, nextIdx);

				// For all nodes in actual layer and bias neuron
				for (auto actualIdx = 0; actualIdx <= m_pNetwork->m_layerSizes[layer]; actualIdx++)
				{
					// Calculate change in weight 
					if (m_useBatchLearning)
					{
						m_deltas[layer](actualIdx, nextIdx) +=
							m_learningRate
							* m_pNetwork->m_neurons[layer][actualIdx].value
							* m_errorGradients[layer + 1][nextIdx];
					}
					else
					{
						m_deltas[layer](actualIdx, nextIdx) =
							m_learningRate
							* m_pNetwork->m_neurons[layer][actualIdx].value
							* m_errorGradients[layer + 1][nextIdx]
							+ m_momentum * m_deltas[layer](actualIdx, nextIdx);
					}
				}
			}

		}

		// If using stochastic learning update the weights immediately
		if (!m_useBatchLearning)
		{
			UpdateWeights();
		}

	}

	void NetworkTrainer::UpdateWeights()
	{
		for (int32_t layer = 0; layer < m_pNetwork->m_numLayers - 1; ++layer)
		{
			for (int32_t actualIdx = 0; actualIdx <= m_pNetwork->m_layerSizes[layer]; ++actualIdx)
			{
				for (int32_t nextIdx = 0; nextIdx < m_pNetwork->m_layerSizes[layer + 1]; ++nextIdx)
				{
					m_pNetwork->m_weightsByLayer[layer](actualIdx, nextIdx) +=
						m_deltas[layer](actualIdx, nextIdx);

					// Clear delta only if using batch (previous delta is needed for momentum
					if (m_useBatchLearning)
					{
						m_deltas[layer](actualIdx, nextIdx) = 0.0;
					}

				}
			}

		}
	}

	void NetworkTrainer::GetSetAccuracyAndMSE(TrainingSet const& trainingSet, double& accuracy, double& MSE) const
	{
		accuracy = 0;
		MSE = 0;

		double numIncorrectResults = 0;
		for (auto const& trainingEntry : trainingSet)
		{
			m_pNetwork->Evaluate(trainingEntry.m_inputs);

			// Check if the network outputs match the expected outputs
			bool correctResult = true;
			for (int32_t outputIdx = 0; outputIdx < m_pNetwork->m_numOutputs; outputIdx++)
			{
				if ((double)m_pNetwork->m_clampedOutputs[outputIdx] != trainingEntry.m_expectedOutputs[outputIdx])
				{
					correctResult = false;
				}

				MSE += pow(((*m_pNetwork->m_outputNeurons)[outputIdx].value - trainingEntry.m_expectedOutputs[outputIdx]), 2);
			}

			if (!correctResult)
			{
				numIncorrectResults++;
			}
		}

		accuracy = 100.0f - (numIncorrectResults / trainingSet.size() * 100.0);
		MSE = MSE / (m_pNetwork->m_numOutputs * trainingSet.size());
	}

}
