#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

namespace parallel_assignment
{

	class ReadFile
	{
	public:
		ReadFile(std::string filename);
		ReadFile() {};
		~ReadFile();

		void Load(std::string filename);
		void Load(std::string filename, size_t localSize);

		std::vector<float>& GetData();
		int GetDataSize();
		int GetPaddingSize();
		int GetTotalSize();
	private:
		std::vector<float>* _data;

		float ParseLine(std::string line);

		int _dataSize;
		int _paddingSize = 0;
	};

}
