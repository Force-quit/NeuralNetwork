//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2021 Xavier Provençal
// Copyright (C) 2024 Émile Laforce
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <format>

namespace bpn
{
	class ActivationFunction
	{
	public:
		virtual ~ActivationFunction() = default;

		/**
		 * Let f be the activation function, returns f(x)
		 */
		virtual double evaluate(double x) const = 0;

		/**
		 * Let f be the activation function, returns f'(x)
		 *
		 * Second parameter is the pre-computed f(x). In some cases, like for
		 * the sigmoid activation function, it is faster to compute f'(x) from
		 * f(x) then from x.
		 */
		virtual double evalDerivative(double x, double fx = 0.0) const = 0;

		/**
		 * Representation of the function as text.
		 */
		virtual std::string serialize() const = 0;

		static ActivationFunction* deserialize(std::string_view s);
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

		Sigmoid() : lambda{ 1 }
		{ }

		Sigmoid(double lambda) : lambda{ lambda }
		{ }

		double evaluate(double x) const override
		{
			return 1.0 / (1.0 + std::exp(-lambda * x));
		}

		double evalDerivative(double x, double fx = 0.0) const override
		{
			return lambda * fx * (1.0 - fx);
		}

		std::string serialize() const override
		{
			return std::format("Sigmoid({:.6f})", lambda);
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

		double evaluate(double x) const override
		{
			//return std::log(1.0 + std::exp(x));
			return (x > 0) ? x : 0;
		}

		double evalDerivative(double x, double fx) const override
		{
			return (x > 0) ? 1 : 0;
		}

		std::string serialize() const override
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
		double evaluate(double x) const override
		{
			return (x > 0) ? x : 0.01 * x;
		}

		double evalDerivative(double x, double fx) const override
		{
			return (x > 0) ? 1 : 0.01;
		}

		std::string serialize() const override
		{
			return "LeakyReLU";
		}
	};
}