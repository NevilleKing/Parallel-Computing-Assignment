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
		_data->push_back(ParseLine(line));
	}

	thefile.close();
}

float ReadFile::ParseLine(std::string line)
{
	int numSpaces = 0;
	int index = 0;
	for (; index < line.length(); index++)
	{
		if (line[index] == ' ')
			numSpaces++;
		if (numSpaces == 5)
			break;
	}

	index++;

	return std::strtof(line.substr(index).c_str(), 0);
}
