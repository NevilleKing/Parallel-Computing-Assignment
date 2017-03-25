#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <memory>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace parallel_assignment
{
	// buffer object for passing between different instances of the kernel class
	class Buffer
	{
	public:
		Buffer(cl::Buffer* buffer, int buffer_size)
		{
			buff = buffer;
			size = buffer_size;

		}
		~Buffer() 
		{
			if (buff != nullptr)
				delete buff;
		}
		cl::Buffer* buff;
		int size;
	private:
		Buffer(const Buffer& obj);
	};

	class Kernel
	{
	public:
		Kernel(std::string kernel_name, int local_size, cl::Context& context, cl::CommandQueue& queue, cl::Program& program);
		~Kernel() {};

		int AddBuffer(const std::vector<int>& input, bool readOnly = true);
		int AddBuffer(int numElements);
		int AddBufferLong(int numElements);
		int AddBufferFromBuffer(const std::shared_ptr<Buffer> prevBuffer);
		void AddArg(int arg);
		void AddLocalArg();
		void AddLocalArgLong();

		const std::shared_ptr<Buffer> GetRawBuffer(int buffer_id);

		void Execute();

		void ReadBuffer(int buffer_id, std::vector<int>& output_vector);
		void ReadBufferLong(int buffer_id, std::vector<long>& output_vector);

		int GetTime();

	private:
		cl::Context* _context;
		cl::CommandQueue* _queue;
		cl::Program* _program;

		cl::Kernel _kernel;

		std::vector<std::shared_ptr<parallel_assignment::Buffer>> _buffers;

		int _currentArgument = 0;

		int _local_size;
		int _input_elements;

		int _profile_time = -1;
	};

}
