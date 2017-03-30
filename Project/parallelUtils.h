#pragma once

#include <iostream>
#include <vector>

namespace parallel_assignment
{
	template <typename T>
	int PadVector(std::vector<T>& data, int localSize)
	{
		size_t padding_size = data.size() % localSize;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(localSize - padding_size, 0);
			//append that extra vector to our input
			data.insert(data.end(), A_ext.begin(), A_ext.end());

			return A_ext.size();
		}

		return 0;
	}
};
