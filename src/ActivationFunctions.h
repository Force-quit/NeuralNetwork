//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2021 Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------


#pragma once

#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits>


namespace bpn
{
	enum class ActivationFunctionType
	{
		Sigmoid, ReLU
	};

	class Sigmoid;
	class ReLU;

	class ActivationFunction
	{
	public:
		/**
		 * Let f be the activation function, returns f(x)
		 */
		virtual double evaluate(double x) const
		{
			(void)x;
			return 0.0;
		}

		/**
		 * Let f be the activation function, returns f'(x)
		 *
		 * Second parameter is the pre-computed f(x). In some cases, like for
		 * the sigmoid activation function, it is faster to compute f'(x) from
		 * f(x) then from x.
		 */
		virtual double evalDerivative(double x, double fx = 0.0) const
		{
			(void)x;
			(void)fx;
			return 0.0;
		}

		/**
		 * Representation of the function as text.
		 */
		virtual std::string serialize() const
		{
			return NULL;
		}

		static ActivationFunction* deserialize(const std::string& s);
	};

	class Sigmoid : public ActivationFunction
	{
		/**
		 *                   1
		 * f(x) =    -----------------
		 *           1 + exp(-lambda*x)
		 *
		 * f'(x) = lambda * f(x) * (1-f(x))
		 */
	public:

		Sigmoid() : lambda(1.0)
		{ }

		Sigmoid(double lambda) : lambda(lambda)
		{ }

		inline double evaluate(double x) const
		{
			return 1.0 / (1.0 + std::exp(-lambda * x));
		}

		inline double evalDerivative(double x, double fx = 0.0) const
		{
			(void)x; // avoid compilation warning for unused variable
			return lambda * fx * (1.0 - fx);
		}

		inline std::string serialize() const
		{
			std::stringstream ss;
			ss << "Sigmoid("
				<< std::setprecision(std::numeric_limits<long double>::digits10 + 1)
				<< lambda << ")";
			return ss.str();
		}

		const double lambda;
	};

	class ReLU : public ActivationFunction
	{
		/**
		 *
		 * f(x) =  max(x,0);
		 *
		 * f'(x) = (x > 0) ? 1 : 0;
		 */
	public:

		ReLU() { }

		inline double evaluate(double x) const
		{
			//return std::log(1.0 + std::exp(x));
			return (x > 0) ? x : 0;
		}

		inline double evalDerivative(double x, double fx) const
		{
			(void)x; (void)fx;
			return (x > 0) ? 1 : 0;
		}

		inline std::string serialize() const
		{
			return "ReLU";
		}
	};

	class LeakyReLU : public ActivationFunction
	{
		/**
		 * Like ReLU but in case of a negative input x, then the ouput is x/100
		 * instead of 0.
		 *
		 * The idea is that a negative input is almost 0 but it still have a
		 * non-nul derivative.
		 */
	public:

		LeakyReLU() { }

		inline double evaluate(double x) const
		{
			return (x > 0) ? x : 0.01 * x;
		}

		inline double evalDerivative(double x, double fx) const
		{
			(void)x; (void)fx;
			return (x > 0) ? 1 : 0.01;
		}

		inline std::string serialize() const
		{
			return "LeakyReLU";
		}
	};


}

