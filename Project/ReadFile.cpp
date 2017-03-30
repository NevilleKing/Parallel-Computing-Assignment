#include "ReadFile.h"

namespace parallel_assignment
{

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
		Load(filename, 0);
	}

	void ReadFile::Load(std::string filename, size_t localSize)
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

		// cache the data size so it doesn't affect mean calculations
		_dataSize = _data->size();


		// pad the vector if needed
		if (localSize != 0)
		{
			_paddingSize = parallel_assignment::PadVector<float>(*_data, localSize);
		}
	}

	std::vector<float>& ReadFile::GetData()
	{
		return *_data;
	}

	int ReadFile::GetDataSize()
	{
		return _dataSize;
	}

	int ReadFile::GetPaddingSize()
	{
		return _paddingSize;
	}

	int ReadFile::GetTotalSize()
	{
		return _dataSize + _paddingSize;
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

}
