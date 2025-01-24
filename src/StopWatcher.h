#pragma once

#include <filesystem>

class StopWatcher
{
private:
	inline static std::filesystem::path stopFilePath;

public:

	static void init(const std::filesystem::path& filePath);

	static void stop()
	{
		std::filesystem::remove(stopFilePath);
	}

	[[nodiscard]] static bool stopRequested()
	{
		return !std::filesystem::exists(stopFilePath);
	}
};