#pragma once

namespace parallel_assignment
{

	template <class T>
	class Kernel
	{
	public:
		Kernel();
		~Kernel();


		void AddBuffer();

		void Execute();

		void ReadBuffer();
	};

}