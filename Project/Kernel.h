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

		int AddBuffer(const std::vector<float>& input, bool readOnly = true);

		template<typename F>
		int AddBuffer(int numElements)
		{
			int size = numElements * sizeof(F);

			cl::Buffer* buff = new cl::Buffer(*_context, CL_MEM_READ_WRITE, size);

			_queue->enqueueFillBuffer(*buff, 0, 0, size);

			_kernel.setArg(_currentArgument++, *buff);

			_buffers.push_back(std::shared_ptr<Buffer>(new parallel_assignment::Buffer(buff, size)));

			return _buffers.size() - 1;
		}

		int AddBufferFromBuffer(const std::shared_ptr<Buffer> prevBuffer);
		void AddArg(float arg);

		template<typename G>
		void AddLocalArg()
		{
			_kernel.setArg(_currentArgument++, cl::Local(_local_size * sizeof(G)));
		}

		const std::shared_ptr<Buffer> GetRawBuffer(int buffer_id);

		void Execute();

		template<typename T>
		void ReadBuffer(int buffer_id, std::vector<T>& output_vector)
		{
			_queue->enqueueReadBuffer(*_buffers[buffer_id].get()->buff, CL_TRUE, 0, _buffers[buffer_id].get()->size, &output_vector[0]);
		}

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
