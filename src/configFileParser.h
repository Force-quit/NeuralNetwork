#pragma once

// Configuration file parser
// Copyright (C) 2021 Xavier Provençal
// Copyright (C) 2025 Émile Laforce
// 
// MIT license: https://opensource.org/licenses/MIT

#include <string>
#include <string_view>
#include <optional>
#include <format>
#include <ranges>
#include <vector>
#include <map>

#include <iostream>
#include <fstream>
#include <sstream>

class ConfigFileParser 
{
public:
	ConfigFileParser(std::string_view filename)
		: filename(filename)
	{

	}

	[[nodiscard]] std::optional<std::string> run()
	{
		std::ifstream file(filename);
		if (!file.good())
		{
			return std::format("Error opening configuration file `{}`", filename);
		}

		std::string line;
		std::uint16_t lineCount{};
		while (std::getline(file, line))
		{
			++lineCount;

			if (line.starts_with('#') || line.empty())
			{
				continue;
			}

			auto splitValues = line | std::ranges::views::split('=');
			std::vector<std::string> keyAndValue = splitValues | std::ranges::to<std::vector<std::string>>();

			if (keyAndValue.size() != 2)
			{
				return std::format("Configuration file `{}` line {} is not a key-value pair", filename, lineCount);
			}

			entries[keyAndValue[0]] = keyAndValue[1];
		}

		return std::nullopt;
	}

	template<typename T>
	T get(const std::string& name) const 
	{
		std::map<std::string, std::string>::const_iterator it = entries.find(name);
		if (it == entries.end()) {
			std::stringstream ss;
			ss << "Configuration file `" << filename << "` does not specify the required feild `" << name << "`\n";
			throw std::runtime_error(ss.str());
		}
		return convert<T>(it->second);
	}

private:
	template<typename T>
	static T convert(const std::string& s)
	{
		std::stringstream ss(s);
		T t;
		ss >> t;
		return t;
	}

	template<>
	static bool convert<bool>(const std::string& s) 
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

	std::string filename;
	std::map<std::string, std::string> entries;
};