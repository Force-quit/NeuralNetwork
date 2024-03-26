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
#include <vector>

namespace bpn
{
	class Matrix
	{
	public:
		constexpr Matrix(int nRows, int nCols, double value = 0.0) noexcept
			: nRows{ nRows }, nCols{ nCols }, data(nRows * nCols, value)
		{
			assert(nRows > 0 && nCols > 0 && "Matrix constructor has 0 size");
		}

		// TODO Multidimensional subscript operator when MSVC supports it
		[[nodiscard]] constexpr double& operator()(int r, int c)
		{
			assert(r >= 0 && r < nRows && c >= 0 && c < nCols && "Matrix subscript out of bounds");
			return data[r * nCols + c];
		}

		// TODO Multidimensional subscript operator when MSVC supports it
		[[nodiscard]] constexpr double operator()(int r, int c) const
		{
			assert(r >= 0 && r < nRows && c >= 0 && c < nCols && "Matrix subscript out of bounds");
			return data[r * nCols + c];
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

	private:
		const int nRows;
		const int nCols;
		std::vector<double> data;
	};
}