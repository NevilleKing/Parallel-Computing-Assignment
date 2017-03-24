#pragma once

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace parallel_assignment
{
	class Kernel
	{
	public:
		Kernel(std::string kernel_name, int local_size, cl::Context& context, cl::CommandQueue& queue, cl::Program& program);
		~Kernel() {};

		int AddBuffer(const std::vector<int>& input, bool readOnly = true);
		int AddBuffer(int numElements);
		void AddLocalArg();

		void Execute();

		void ReadBuffer(int buffer_id, std::vector<int>& output_vector);

		int GetTime();

	private:
		cl::Context* _context;
		cl::CommandQueue* _queue;
		cl::Program* _program;

		cl::Kernel _kernel;

		std::vector<std::pair<cl::Buffer, int>> _buffers;

		int _currentArgument = 0;

		int _local_size;
		int _input_elements;

		int _profile_time = -1;
	};

}
