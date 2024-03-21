#pragma once

#include <fstream>
#include <iostream>

class StopFileWatcher
{
private:
	static const char* filename;

public:

	static void init(const char* _filename)
	{
		filename = _filename;
		std::ofstream f(filename, std::ofstream::out);
		f << "0" << std::endl;
		f.close();
	}

	static bool doesTheStopFileTellsMeToStop()
	{
		int x;
		std::ifstream f(filename, std::ifstream::in);
		f >> x;
		return (x == 1);
	}
};


