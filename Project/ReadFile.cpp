#include "ReadFile.h"

ReadFile::ReadFile(std::string filename)
{
	_data = new std::vector<float>;
}

ReadFile::~ReadFile()
{
	if (_data != nullptr)
		delete _data;
}
