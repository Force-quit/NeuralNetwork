#pragma once

#include <fstream>
#include <filesystem>

class StopFileWatcher
{
private:
	inline static std::filesystem::path stopFilePath;

public:

	static void init(std::filesystem::path filePath)
	{
		stopFilePath = filePath;
		std::ofstream f(stopFilePath);
		f << "Delete this file to stop the training";
	}

	[[nodiscard]] static bool stopRequested()
	{
		return !std::filesystem::exists(stopFilePath);
	}
};


