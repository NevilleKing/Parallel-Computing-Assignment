#include "ReadFile.h"

ReadFile::ReadFile(std::string filename)
{
	Load(filename);
}

ReadFile::~ReadFile()
{
	if (_data != nullptr)
		delete _data;
}

void ReadFile::Load(std::string filename)
{
	// create the data vector
	_data = new std::vector<float>;

	std::ifstream thefile;
	thefile.open(filename, std::ios::in);

	if (!thefile.is_open())
	{
		std::cout << "Error opening " << filename.c_str() << std::endl;
		throw "Open File Error";
		return;
	}

	std::string line;
	while (std::getline(thefile, line))
	{
		std::cout << line << std::endl;
	}

	thefile.close();
}
