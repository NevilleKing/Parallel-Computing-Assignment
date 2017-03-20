#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

class ReadFile
{
public:
	ReadFile(std::string filename);
	ReadFile() {};
	~ReadFile();

	void Load(std::string filename);
private:
	std::vector<float>* _data;
};
