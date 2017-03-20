#pragma once

#include <iostream>
#include <vector>

class ReadFile
{
public:
	ReadFile(std::string filename);
	~ReadFile();
private:
	std::vector<float>* _data;
};
