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
		_data = new std::vector<int>;

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
			size_t padding_size = _data->size() % localSize;

			//if the input vector is not a multiple of the local_size
			//insert additional neutral elements (0 for addition) so that the total will not be affected
			if (padding_size) {
				//create an extra vector with neutral values
				std::vector<int> A_ext(localSize - padding_size, 0);
				//append that extra vector to our input
				_data->insert(_data->end(), A_ext.begin(), A_ext.end());

				_paddingSize = A_ext.size();
			}
		}
	}

	std::vector<int>& ReadFile::GetData()
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

		return std::strtof(line.substr(index).c_str(), 0) * 100;
	}

}
