//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// Copyright (C) 2017  Bobby Anguelov
// Copyright (C) 2018  Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include "DataReader.h"
#include <assert.h>
#include <iosfwd>
#include <algorithm>
#include <iostream>
#include <random>
#include <memory>

//-------------------------------------------------------------------------


//std::vector<std::string> split(const std::string& t, const std::string& m)
//{
//  std::vector<std::string> splitted;
//  std::size_t first = 0;
//  std::size_t last = t.find( m );
//  while ( last != std::string::npos )
//    {
//      splitted.push_back( t.substr( first, last-first ) );
//      first = last + m.size();
//      last = t.find( m, first );
//    }
//  splitted.push_back( t.substr( first, t.size() - first ) );
//  return splitted;
//}

void insertListInStream(std::stringstream& ss, const std::string& text, const std::string& delimitor)
{
	std::size_t first = 0;
	std::size_t last = text.find(delimitor);
	while (last != std::string::npos)
	{
		ss << text.substr(first, last - first) << " ";
		first = last + delimitor.size();
		last = text.find(delimitor, first);
	}
	ss << text.substr(first, text.size() - first);
}

void textToListOfDouble(std::vector<double>& l, const std::string& s)
{
	for (uint32_t i = 0; i < s.size(); ++i)
	{
		l.push_back(((double)s[i]) / 256.0);
	}
}


namespace bpn
{
	DataReader::DataReader(std::string const& filename,
		int32_t numInputs,
		int32_t numOutputs,
		Format format,
		int verbosity)
		: m_filename(filename),
		m_numInputs(numInputs),
		m_numOutputs(numOutputs),
		m_dataFormat(format),
		m_verbosity(verbosity)
	{
		assert(m_numInputs > 0 && m_numOutputs > 0);
		if (m_filename.compare("-") == 0)
		{
			m_dataStream = &std::cin;
		}
		else
		{
			auto fileStream(std::make_unique<std::ifstream>());
			if (m_dataFormat == Format::binary)
			{
				fileStream->open(m_filename, std::ios::in | std::ios::binary);
			}
			else
			{
				fileStream->open(m_filename, std::ios::in);
			}
			if (!fileStream->is_open())
				throw std::runtime_error("Unable to read from input file");
			m_dataStream = fileStream.release();
		}
	}

	bool DataReader::readOneInputData(std::vector<double>& inputValues)
	{
		inputValues.clear();
		std::string line;
		std::getline(*m_dataStream, line);
		if (m_dataFormat == bpn::DataReader::Format::numberList)
		{
			std::stringstream ss;
			insertListInStream(ss, line, ",");
			for (int i = 0; i < m_numInputs; ++i)
			{
				double d;
				ss >> d;
				inputValues.push_back(d);
			}
		}

		if (m_verbosity >= 2)
		{
			std::cout << "  Input : " << inputValues << std::endl;
		}
		return true;
	}

	bool DataReader::readTraningData(TrainingData& data)
	{
		std::vector<TrainingEntry> entries;
		std::string line;

		if (m_dataFormat == Format::binary)
		{
			int nbData, nbInputValues, nbOutputValues;
			m_dataStream->read((char*)&nbData, sizeof(int));
			m_dataStream->read((char*)&nbInputValues, sizeof(int));
			m_dataStream->read((char*)&nbOutputValues, sizeof(int));

			for (int i = 0; i < nbData; ++i)
			{
				entries.push_back(TrainingEntry());
				TrainingEntry& entry = entries.back();

				for (int i = 0; i < m_numInputs; ++i)
				{
					double d;
					int x = 0;
					m_dataStream->read((char*)&x, sizeof(char));
					d = (1.0 * x) / 255;
					entry.m_inputs.push_back(d);
				}

				for (int i = 0; i < m_numOutputs; ++i)
				{
					int x = 0;
					m_dataStream->read((char*)&x, sizeof(char));
					entry.m_expectedOutputs.push_back(x);
				}
				if (m_verbosity >= 2)
				{
					std::cout << "  Input : " << entry.m_inputs << std::endl;
					std::cout << "  Output : " << entry.m_expectedOutputs << "\n" << std::endl;

				}

				assert(entry.m_inputs.size() == (uint32_t)m_numInputs);
				assert(entry.m_expectedOutputs.size() == (uint32_t)m_numOutputs);

			}
		}
		else if (m_dataFormat == bpn::DataReader::Format::numberList)
		{
			while (!m_dataStream->eof())
			{
				std::getline(*m_dataStream, line);

				// line that starts with # are comments and thus ignored.
				if (line[0] == '#' || line.size() == 0)
					continue;

				entries.push_back(TrainingEntry());
				TrainingEntry& entry = entries.back();

				std::stringstream ss;
				insertListInStream(ss, line, ",");
				for (int i = 0; i < m_numInputs; ++i)
				{
					double d;
					ss >> d;
					entry.m_inputs.push_back(d);
				}
				for (int i = 0; i < m_numOutputs; ++i)
				{
					int32_t x;
					ss >> x;
					entry.m_expectedOutputs.push_back(x);
				}
				if (m_verbosity >= 2)
				{
					std::cout << "  Input : " << entry.m_inputs << std::endl;
					std::cout << "  Output : " << entry.m_expectedOutputs << "\n" << std::endl;

				}

				assert(entry.m_inputs.size() == (uint32_t)m_numInputs);
				assert(entry.m_expectedOutputs.size() == (uint32_t)m_numOutputs);
			}

		}
		if (!entries.empty())
		{
			CreateTrainingData(data, entries);
		}
		return true;
	}




	void DataReader::CreateTrainingData(TrainingData& data, std::vector<TrainingEntry>& entries)
	{
		assert(!entries.empty());

		std::random_device rd;
		std::mt19937 g(rd());
		std::shuffle(entries.begin(), entries.end(), g);

		// Training set
		int32_t const numEntries = (int32_t)entries.size();
		int32_t const numTrainingEntries = (int32_t)(0.8 * numEntries);
		int32_t const numGeneralizationEntries = (int32_t)(ceil(0.1 * numEntries));

		int32_t entryIdx = 0;
		for (; entryIdx < numTrainingEntries; entryIdx++)
		{
			data.m_trainingSet.push_back(entries[entryIdx]);
		}

		// Generalization set
		for (; entryIdx < numTrainingEntries + numGeneralizationEntries; entryIdx++)
		{
			data.m_generalizationSet.push_back(entries[entryIdx]);
		}

		// Validation set
		for (; entryIdx < numEntries; entryIdx++)
		{
			data.m_validationSet.push_back(entries[entryIdx]);
		}
	}
}

