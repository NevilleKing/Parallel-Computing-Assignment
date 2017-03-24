#include "Kernel.h"

namespace parallel_assignment
{
	Kernel::Kernel(std::string kernel_name, int local_size, cl::Context& context, cl::CommandQueue& queue, cl::Program& program)
	{
		_context = &context;
		_queue = &queue;
		_program = &program;
		_local_size = local_size;

		// create the kernel
		_kernel = cl::Kernel(program, kernel_name.c_str());
	}

	int Kernel::AddBuffer(const std::vector<int>& input, bool readOnly)
	{
		int size = input.size() * sizeof(int);
		int readWrite = (readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE);
		cl::Buffer buff(*_context, readWrite, size);

		_queue->enqueueWriteBuffer(buff, CL_TRUE, 0, size, &input[0]);

		_kernel.setArg(_currentArgument++, buff);
		
		_buffers.push_back(std::make_pair(buff, size));

		// first passed in buffer is assumed to be input array
		if (_buffers.size() == 1)
			_input_elements = input.size();

		return _buffers.size() - 1;
	}

	int Kernel::AddBuffer(int numElements)
	{
		int size = numElements * sizeof(int);

		cl::Buffer buff(*_context, CL_MEM_READ_WRITE, size);

		_queue->enqueueFillBuffer(buff, 0, 0, size);

		_kernel.setArg(_currentArgument++, buff);

		_buffers.push_back(std::make_pair(buff, size));

		return _buffers.size() - 1;
	}

	void Kernel::AddLocalArg()
	{
		_kernel.setArg(_currentArgument++, cl::Local(_local_size * sizeof(int)));
	}

	void Kernel::Execute()
	{
		cl::Event prof_event;
		_queue->enqueueNDRangeKernel(_kernel, cl::NullRange, cl::NDRange(_input_elements), cl::NDRange(_local_size), NULL, &prof_event);
		prof_event.wait(); // make sure we have a time back
		// calculate the time from the profile event
		_profile_time = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	}

	void Kernel::ReadBuffer(int buffer_id, std::vector<int>& output_vector)
	{
		_queue->enqueueReadBuffer(_buffers[buffer_id].first, CL_TRUE, 0, _buffers[buffer_id].second, &output_vector[0]);
	}

	int Kernel::GetTime()
	{
		return _profile_time;
	}
}
