//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <random>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "NeuralNetwork.h"

//-------------------------------------------------------------------------

namespace BPN
{
  Network::Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma) 
    : m_layerSizes(layerSizes)
      , m_sigma(sigma)
    {
      assert(layerSizes.size() >= 3);
      m_numLayers       = m_layerSizes.size();
      m_numInputs       = m_layerSizes[0];
      m_numOutputs      = m_layerSizes[m_numLayers-1];
      m_numOnLastHidden = m_layerSizes[m_numLayers-2];
      InitializeNetwork();
      InitializeWeights();
    }


  Network::Network( std::istream& is )
    {
      deserialize( is );
    }


  void Network::InitializeNetwork()
    {
      // Create storage and initialize the neurons and the outputs
      //-------------------------------------------------------------------------

      for (auto layerSize : m_layerSizes) 
        {
          m_neurons.push_back(Layer(layerSize, Neuron(0,0)));
        }
      m_inputNeurons = &m_neurons[0];
      m_lastHiddenNeurons = &m_neurons[m_numLayers-2];
      m_outputNeurons = &m_neurons[m_numLayers-1];


      // Add bias values
      for (int i=0; i<m_numLayers-1; ++i)
        {
          m_neurons[i].push_back(Neuron(1.0, 1.0));
        }

      // Set the size of clamped output 
      m_clampedOutputs.resize( m_numOutputs, 0 );

      // Create storage and initialize the weights
      //-------------------------------------------------------------------------
      for (int i=0; i<m_numLayers-1; ++i)
        {
          // add one the the input size for th bias
          m_weightsByLayer.push_back(Matrix(m_layerSizes[i]+1, m_layerSizes[i+1], 0.0));
        }

    }

  void Network::InitializeWeights()
    {
      std::random_device rd;
      //std::mt19937 generator( rd() );
      std::mt19937 generator( 0 );

      double const distributionRangeHalfWidth = ( 2.4 / m_numInputs );
      double const standardDeviation = distributionRangeHalfWidth * 2 / 6;
      std::normal_distribution<> normalDistribution( 0, standardDeviation );

      // Set weights to normally distributed random values between [-2.4 / numInputs, 2.4 / numInputs]
      for ( int32_t i=0; i<m_numLayers-1; ++i) 
        {
          int32_t currentLayerSize = m_layerSizes[i];
          int32_t nextLayerSize = m_layerSizes[i+1];
          for (int32_t currentLayerIdx=0; currentLayerIdx<=currentLayerSize; ++currentLayerIdx)
            {
              for (int32_t nextLayerIdx=0; nextLayerIdx<nextLayerSize; ++nextLayerIdx)
                {
                  double const weight = normalDistribution( generator );
                  m_weightsByLayer[i](currentLayerIdx, nextLayerIdx) = weight;
                }
            }
        }
    }


  std::vector<int32_t> const& Network::Evaluate( std::vector<double> const& input )
    {
      assert( input.size() == (unsigned int) m_numInputs );
      for ( int i=0; i<m_numLayers-1; ++i )
        {
          assert( m_neurons[i].back().value == 1.0);
        }

      // Local variables
      Layer& inputNeurons  = *m_inputNeurons;

      // Set input values
      //-------------------------------------------------------------------------

      // Activation function is not applied on the value of input neurons
      for ( int i=0; i<m_numInputs; ++i ) 
        {
          inputNeurons[i] = Neuron(input[i], input[i]);
        }

      // Update neurons one layer at the time starting from the first hidden
      // la yer to the output layer.
      //-------------------------------------------------------------------------
      
      for ( int32_t i=1; i<m_numLayers; ++i )
        {

          for ( int32_t actualIdx=0; actualIdx < m_layerSizes[i]; ++actualIdx )
            {

              double activation = 0.0;

              // Get weighted sum of pattern and bias neuron
              for ( int32_t prevIdx = 0; prevIdx <= m_layerSizes[i-1]; ++prevIdx )
                {
                  activation += m_neurons[i-1][prevIdx].value * m_weightsByLayer[i-1](prevIdx, actualIdx);
                  //std::cout << "layer=" << i << ", actualIdx" << actualIdx << ", prevIdx=" << prevIdx << ", activation = " << m_neurons[i-1][prevIdx].value << " * " << m_weightsByLayer[i-1](prevIdx, actualIdx) << std::endl;
                }

              // Apply activation function
              m_neurons[i][actualIdx].activation = activation;
              m_neurons[i][actualIdx].value = m_sigma->evaluate( activation );

              // If this is the output layer (the last layer), then update
              // clamped outputs
              if (i == m_numLayers-1)
                {
                  m_clampedOutputs[actualIdx] = ClampOutputValue( activation );
                }
            }
        }

      return m_clampedOutputs;
    }

  std::string Network::selfDisplay() const
    {
      std::ostringstream ss;
      ss << "+----------------------------------------------------+\n"
         << "| Number of input  nodes: " << m_numInputs << '\n'
         << "| Number of output nodes: " << m_numOutputs << "\n"
         << "| Layer sizes (first in input, last is output) : " << m_layerSizes << '\n'
         << "|\n"
         << "| Weights : Input  (line)(last is bias) to Hidden #1 (column)\n"
         << m_weightsByLayer[0]
         << "|\n";
      for (int32_t i=1; i<m_numLayers-2; ++i) 
        {
          ss << "| Weights : Hidden #" << i << " (line)(last is bias) to Hidden #" << i+1 << " (column)\n" 
             << m_weightsByLayer[i]
             << "|\n";
        }
      ss << "| Weights : Hidden #" << m_numLayers-2 << " (line)(last is bias) to Output (column)\n"
         << m_weightsByLayer[m_numLayers-2]
         << "|\n"
         << "| --- Neurons ---\n"
         << "|\n"
         << "| Input layer       : " << *m_inputNeurons << "\n";
         for ( int32_t i=1; i<m_numLayers-1; ++i )
           {
             ss << "| Hidden layer #" << i << "   : " << m_neurons[i] << "\n";
           }
      ss << "| Output neurons    : " << *m_outputNeurons << "\n"
         << "| Clamp o/p neurons : " << m_clampedOutputs << "\n"
         << "+----------------------------------------------------+\n";
      return ss.str();
    }

  void Network::deserialize(std::istream& is)
    {
      std::string s;
      is >> s;
      if ( s.compare( "layerSizes" ) != 0 )
        {
          throw std::runtime_error("Invalid BPN serialization");
        }
      is >> m_layerSizes;

      is >> s;
      if ( s.compare( "activation" ) != 0 )
        {
          throw std::runtime_error("Invalid BPN serialization");
        }
      is >> s;
      m_sigma           = ActivationFunction::deserialize(s);

      m_numLayers       = m_layerSizes.size();
      m_numInputs       = m_layerSizes[0];
      m_numOutputs      = m_layerSizes[m_numLayers-1];
      m_numOnLastHidden = m_layerSizes[m_numLayers-2];

      InitializeNetwork();

      // Read weights
      is >> s;
      if ( s.compare( "weights" ) != 0 )
        {
          throw std::runtime_error("Invalid BPN serialization");
        }
      for( int32_t i=0; i<m_numLayers; ++i )
        {
          for (int32_t actualIdx = 0; actualIdx <= m_layerSizes[i]; ++actualIdx)
            {
              for (int32_t nextIdx = 0; nextIdx < m_layerSizes[i+1]; ++nextIdx)
                {
                  double d;
                  is >> d;
                  std::cout << "i=" << i << ", actualIdx=" << actualIdx << ", nextIdx=" << nextIdx << ", d=" << d << std::endl;
                  m_weightsByLayer[i](actualIdx, nextIdx) = d;
                }
            }
        }
    }

  std::string Network::serialize() const
    {
      std::stringstream ss;
      ss << "layerSizes "      << m_layerSizes << '\n';
      ss << "activation "      << m_sigma->serialize() << '\n';
      ss << "weights";
      for( int32_t i=0; i<m_numLayers; ++i )
        {
          for (int32_t actualIdx = 0; actualIdx <= m_layerSizes[i]; ++actualIdx)
            {
              for (int32_t nextIdx = 0; nextIdx < m_layerSizes[i+1]; ++nextIdx)
                {
                  ss << ' ' << m_weightsByLayer[i](actualIdx, nextIdx);
                }
            }
        }
      return ss.str();
    }

  

  void Network::saveToFile(const char* filename) const
    {
      (void) filename;
    }

  void Network::loadFromFile(const char* filename)
    {
      (void) filename;
    }


std::ostream& operator<<( std::ostream& os, const BPN::Network& n )
  {
    //return os;
    os << n.selfDisplay();
    return os;
  }

std::ostream& operator<<( std::ostream& os, const BPN::Neuron& n) 
  {
    os << "(" << n.activation << ", " << n.value << ")";
    return os;
  }

}
