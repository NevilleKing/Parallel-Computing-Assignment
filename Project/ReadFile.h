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

		std::vector<int>& GetData();
		int GetDataSize();
	private:
		std::vector<int>* _data;

		float ParseLine(std::string line);

		int _dataSize;
	};

}
