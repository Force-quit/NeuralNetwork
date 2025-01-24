// Configuration file parser
// Copyright (C) 2021 Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//

// Parses a configuratoin file.
// `#` indicates comments except if preceded by `\`.
//

#pragma once
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include "cmdParser.h"

namespace cfp 
{
	template<typename T>
	T convert(const std::string& s)
	{
		std::stringstream ss(s);
		T t;
		ss >> t;
		return t;
	}

	class Parser 
	{
	private:
		std::string filename;
		std::map<std::string, std::string> entries;

	public:

		Parser(const std::string& filename) : filename(filename)
		{
		}


		bool run()
		{
			std::ifstream file(filename);
			if (!file.good())
			{
				std::stringstream ss;
				ss << "Unable to read configuration file `" << filename << "`.\n";
				throw std::runtime_error(ss.str());
			}

			std::string line;
			int lineCount = 0;
			while (std::getline(file, line))
			{
				lineCount++;
				std::stringstream ss(removeComments(line));

				// Get key, if empty jump to next line
				std::string key;
				ss >> key;
				if (key.length() == 0)
					continue;

				std::string equalSign;
				ss >> equalSign;
				if (equalSign != "=") {
					std::cerr << "Error reading configuration file `" << filename << "` at line " << lineCount << ".\n";
					std::cerr << "Second token should be `=` but got `" << equalSign << "`." << std::endl;
					return false;
				}

				std::stringstream value;
				std::string tmp;
				ss >> tmp;
				value << tmp;
				while (ss >> tmp)
				{
					value << " " << tmp;
				}
				entries[key] = value.str();
			}
			return true;
		}

		template<typename T>
		T get(const std::string& name) const {
			std::map<std::string, std::string>::const_iterator it = entries.find(name);
			if (it == entries.end()) {
				std::stringstream ss;
				ss << "Configuration file `" << filename << "` does not specify the required feild `" << name << "`\n";
				throw std::runtime_error(ss.str());
			}
			return convert<T>(it->second);
		}

		template<typename T>
		T get(const std::string& name, const T& defaultValue) const {
			std::map<std::string, std::string>::const_iterator it = entries.find(name);
			if (it == entries.end()) {
				return defaultValue;
			}
			return convert<T>(it->second);
		}

	private:
		std::string removeComments(const std::string& line, size_t startAt = 0)
		{
			size_t pos = line.find("#", startAt);
			if (pos == std::string::npos) {
				return line;
			}
			if (pos > 0 && line.at(pos - 1) == '\\') {
				return removeComments(line, pos + 1);
			}
			return line.substr(0, pos);
		}

	};

	template<>
	bool convert<bool>(const std::string& s) 
	{
		if (s == "1" || s == "true" || s == "True" || s == "TRUE")
		{
			return true;
		}
		if (s == "0" || s == "false" || s == "False" || s == "FALSE")
		{
			return false;
		}
		std::stringstream ss;
		ss << "Configuration file, bad value for boolean field : `" << s << "`\n";
		throw std::runtime_error(ss.str());
	}
}