TODO
GraphCL
[![Build status](https://ci.appveyor.com/api/projects/status/8cg021yesld25ykm?svg=true)](https://ci.appveyor.com/project/morkonrad/graphcl/runtime)

What is this ? 
--------------


**Features:**
1. Task graph API+Runtime
2. Parallel+asynchronous tasks/kernels execution on CPU+GPU
3. Variable workload splitting, partial offload to GPU
4. Support for APUs and CPUs+dGPUs

Requierments ?
---------------
1. C++17 compiler
2. CMake 3.x
3. OpenCL 1.x headers and lib with support for CPU and/or GPU
4. For unit-tests CTest

How to build ?
---------------
(example for some Windows platform toolchain)
  1. git clone GraphCL /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 15 2017 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2017 is a minimal tested version. For Linux it's tested with GCC 9.3 and Clang 9.0. 
In general, it needs a compiler that supports C++17. 
After succesfull build you can call unit tests to check if they pass:  
 1. cd /clDriver
 2. ctest 
  
How to use it ?
----------------
After successful build and tests, the GraphCL should be ready to go. 
Check sample usage/application below.

Example:
----------------
The following code executes simple task graph. Tasks B,C are executed asynchronously and in parallel on CPU and GPU:
```cpp

#include "clVirtualDevice.h"
#include "clCommon.h"
#include "utils.h"
#include <cassert>
#include <cstdlib>

constexpr auto tasks = R"(
						
						__kernel void kA (global int* A)                        
						{
							const int tid = get_global_id(0);                                                       
							A[tid] = 10;
						}
						
						__kernel void kB (const global int* A, global int* B)                        
						{
							const int tid = get_global_id(0);                                                       
							B[tid] = A[tid]+1;
						}
						
						__kernel void kC (const global int* A, global int* C)                        
						{
							const int tid = get_global_id(0);                                                       
							C[tid] = A[tid]+2;
						}
						
						__kernel void kF (const global int* restrict B, const global int* C, global int* D)                        
						{
							const int tid = get_global_id(0); 
							D[tid] = B[tid]+C[tid];
						}
						)";

int main()
{
  //Simple task_graph consist of 4 tasks	
	/*
	<BEGIN>
	 [A]
	/   \
       [B]  [C]
	\   /
	 [D]
	<END>
	*/
	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(A) = 12 >> C=A+2
	//D(B,C) = 23 >> D=B+C	

	int err = 0;
	
	std::cout << "---------------------------------------------" << std::endl;
	std::cout << "Start DAG with 4 tasks, 2 parallel tasks  ..." << std::endl;
	std::cout << " [A] cpu" << std::endl;
	std::cout << " | |" << std::endl;
	std::cout << "[B][C] gpu_1 || gpu_2" << std::endl;
	std::cout << " | | " << std::endl;
	std::cout << " [D] cpu" << std::endl;
	std::cout << "<END>" << std::endl;
	std::cout << "---------------------------------------------" << std::endl;
	
	auto taskA = device.create_task(tasks, "kA");	
	auto taskB = device.create_task(tasks, "kB");
	auto taskC = device.create_task(tasks, "kC");	
	auto taskD = device.create_task(tasks, "kF");	

	const std::array<size_t, 3> ndr = { items,1,1 };
	const std::array<size_t, 3> wgs = { 16,1,1 };

	size_t begin_byte = 0;
	size_t end_byte = items*sizeof(int);

	map_device_info cpu = { CL_DEVICE_TYPE_CPU,0 };
	map_device_info gpu_a = { CL_DEVICE_TYPE_GPU,0 };
	map_device_info gpu_b = { CL_DEVICE_TYPE_GPU,1 };

	std::vector<int> zeros(items, 0);

	for (int it = 0; it < iterations; it++)
	{
		std::cout << "Iteration:\t" << it + 1;

		//A = 10 
		//B(A) = 11 >> B=A+1
		//C(A) = 12 >> C=A+2
		//D(B,C) = 23 >> D=B+C
		auto mA = device.alloc(zeros, true, cpu);
		auto mB = device.alloc(zeros, true, cpu);
		auto mC = device.alloc(zeros, true, cpu);
		auto mD = device.alloc(zeros, true, cpu);
		
		offload_info exec_a = { { 1.0f,cpu } };
		offload_info exec_b = { { 1.0f,gpu_a } };
		offload_info exec_c = { { 1.0f,gpu_b } };
		offload_info exec_d = { { 1.0f,cpu } };//23

		err = device.execute_async(taskA, exec_a, ndr, wgs, mA);//10
		on_coopcl_error(err);
		
		taskB->add_dependence(taskA.get());		
		err = device.execute_async(taskB, exec_b, ndr, wgs, mA, mB);//11
		on_coopcl_error(err);

		taskC->add_dependence(taskA.get());
		err = device.execute_async(taskC, exec_c, ndr, wgs, mA, mC);//12
		on_coopcl_error(err);

		taskD->add_dependence(taskB.get());
		taskD->add_dependence(taskC.get());
		err = device.execute_async(taskD, exec_d, ndr, wgs, mB, mC, mD);//23
		on_coopcl_error(err);
		
		err = taskD->wait();
		on_coopcl_error(err);

		auto begin_ptr = mD->data_in_buffer_device<int>(cpu);		

		for (int i = 0;i < items;i++)
		{
			const auto val = begin_ptr[i];
			if (val != 23)
			{
				std::vector<int> tmpd(items);
				std::memcpy(tmpd.data(), begin_ptr, items * sizeof(int));

				std::cerr <<" Some error at pos i = " << i <<"\t{"<< val<<"!="<<23<<"}"<<std::endl;
				return -1;
			}
		}
		
		taskA->wait_clear_events();
		taskB->wait_clear_events();
		taskC->wait_clear_events();
		taskD->wait_clear_events();

		std::cout << ":\t DONE" <<std::endl;
	}
	std::cout << "Passed,ok!" << std::endl;
	return 0;
}
```

Current state
----------------

GraphpCL is still work in progress. Nevertheless, the whole-concept was already experimentally tested and works. The driver can successfully execute different kernel-graphs on Intel, NVIDIA and AMD platforms. For bug-report check readme of separate modules.

**Tested systems:**

| HW-Vendor             | CPU       | GPU         | OS                      | Driver version    |
| -----------           | --------- | ----------- | ----------------------- | -------------------- |
| Intel+Nvidia CPU+GPU  | i7-4930k  | GTX-780Ti   | Win10-21H1/Ubuntu20.04  | INT-CPU 18.1.0.0920 + NV-GPU 471.11  win_x64, INT-CPU 18.1.0.0920 + NV-GPU 470.57.02 unix_x64  |
| Intel+AMD 2-CPU+2-GPU | Xeon-6134G| R9-290      | Win10-21H1              | INT-CPU 18.1.0.0920 + AMD-GPU 3075.13 win_x64 |
|                       |           | WX-7100     |                         |                   |
| Intel+Nvidia CPU+3-GPU| i9-7980XE | GTX-1080Ti  | Win10-21H1 / Ubuntu20.04| INT-CPU 18.1.0.0920 + NV-GPU 456.71 win_x64, INT-CPU 18.1.0.0920 + NV-GPU 470.57.02 unix_x64   |
|                       |           | GTX-1080    |                         |                   |
|                       |           | GTX-TitanX  |                         |                   |



References
------------
