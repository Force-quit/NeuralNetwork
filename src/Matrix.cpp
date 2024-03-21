//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2021  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
//
#include <iostream>
#include <sstream>
#include "Matrix.h"
#include <iomanip>
#include <limits>

namespace bpn
{

	std::ostream& operator<<(std::ostream& os, const Matrix& m)
	{
		os << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
		for (int i = 0; i < m.nRows; ++i)
		{
			os << m(i, 0);
			for (int j = 1; j < m.nCols; ++j)
			{
				os << ",\t" << m(i, j);
			}
			os << "\n";
		}
		return os;
	}

}
