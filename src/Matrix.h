//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
//
#pragma once

#include <vector>
#include <iostream>
#include <string.h>

namespace bpn
{

	class Matrix
	{
	public:

		Matrix() : nRows(0), nCols(0), m(NULL) {}

		Matrix(int nRows, int nCols, double value = 0.0) : m(NULL)
		{
			init(nRows, nCols, value);
		}

		Matrix(const Matrix& other) : nRows(other.nRows), nCols(other.nCols)
		{
			this->m = new double[nRows * nCols];
			memcpy(this->m, other.m, nRows * nCols * sizeof(double));
		}


		void init(int nRows, int nCols, double value = 0.0)
		{
			if (nRows == 0 || nCols == 0)
			{
				throw std::runtime_error("Matrix constructor has 0 size");
			}
			if (m != NULL)
			{
				throw std::runtime_error("Can't init a matrix that already has been initialized");
			}
			this->nRows = nRows;
			this->nCols = nCols;
			this->m = new double[nRows * nCols];
			for (int i = 0; i < nRows * nCols; ++i)
			{
				m[i] = value;
			}
		}


		~Matrix()
		{
			delete[] m;
		}


		inline int coordsToIndex(int r, int c) const
		{
			return r * nCols + c;
		}

		double& operator()(int r, int c)
		{
			if (r < 0 || r >= nRows || c < 0 || c >= nCols)
			{
				throw std::runtime_error("Matrix subscript out of bounds");
			}
			return m[coordsToIndex(r, c)];
		}

		double operator()(int r, int c) const
		{
			if (r < 0 || r >= nRows || c < 0 || c >= nCols)
			{
				throw std::runtime_error("Matrix subscript out of bounds");
			}
			return m[coordsToIndex(r, c)];
		}

		friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

	private:
		int nRows;
		int nCols;
		double* m;
	};
}
