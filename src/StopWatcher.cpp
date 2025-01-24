#include "StopWatcher.h"
#include <fstream>

#ifdef _WIN32
#include <Windows.h>
static BOOL WINAPI CtrlHandler(DWORD fdwCtrlType)
{
	if (fdwCtrlType == CTRL_C_EVENT)
	{
		StopWatcher::stop();
		std::exit(0);
		return TRUE;
	}

	return FALSE;
}
#else
#include <csignal>
static void interruptHandler(int signal)
{
	StopWatcher::stop();
	std::exit(0);
}
#endif

void StopWatcher::init(const std::filesystem::path& filePath)
{
	stopFilePath = filePath;
	std::ofstream f(stopFilePath);
	f << "Delete this file to stop the training";

#ifdef _WIN32
	SetConsoleCtrlHandler(&CtrlHandler, TRUE);
#else
	std::signal(SIGINT, &interruptHandler);
#endif
}