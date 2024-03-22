//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// Copyright (C) 2024  Émile Laforce
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <iostream>
#include <memory>

namespace bpn
{
	class Matrix
	{
	public:
		constexpr Matrix(int nRows, int nCols, double value = 0.0) noexcept
			: nRows{ nRows }, nCols{ nCols }, data(std::make_unique<double[]>(nRows* nCols))
		{
			assert(nRows > 0 && nCols > 0 && "Matrix constructor has 0 size");
			for (int i{}; i < nRows * nCols; ++i)
			{
				data[i] = value;
			}
		}

		constexpr Matrix(const Matrix& other) noexcept
			: nRows{ other.nRows }, nCols{ other.nCols }, data(std::make_unique<double[]>(nRows* nCols))
		{
			std::copy(data.get(), data.get() + nRows * nCols, other.data.get());
		}

		// TODO Multidimensional subscript operator when MSVC supports it
		[[nodiscard]] constexpr double& operator()(int r, int c)
		{
			assert(r > 0 && r < nRows && c > 0 && c < nCols && "Matrix subscript out of bounds");
			return data[r * nCols + c];
		}

		// TODO Multidimensional subscript operator when MSVC supports it
		[[nodiscard]] constexpr double operator()(int r, int c) const
		{
			assert(r > 0 && r < nRows && c > 0 && c < nCols && "Matrix subscript out of bounds");
			return data[r * nCols + c];
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

	private:
		const int nRows;
		const int nCols;
		const std::unique_ptr<double[]> data;
	};
}
