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
						
						kernel void kD (const global int* A, global int* D)                        
						{
							const int tid = get_global_id(0);                                                       
							D[tid] = A[tid]+3;
						}
						
						kernel void kE (const global int* B,const global int* C,const global int* D,global int* E)                        
						{
							const int tid = get_global_id(0);                                                       
							E[tid] = B[tid]+C[tid]+D[tid];
						}

						kernel void kF (const global int* restrict B, const global int* C, global int* D)                        
						{
							const int tid = get_global_id(0); 
							D[tid] = B[tid]+C[tid];
						}
						)";


template<typename T>
static auto compare = [](const std::vector<T>& c1,const std::vector<T>& c2,const T val)->bool
{
	if(c1.size()!=c2.size())return false;

	bool ok=true;
	for(size_t i=0;i<c1.size();i++)
	{
		if(c1[i]!=c2[i] || c1[i]!=val || c2[i]!=val)
		{
			std::cerr <<"Wrong value: {"<<c1[i]<<","<<c2[i]<<","<< val << "} pos: " << i << std::endl;
			ok&=false;
		}
	}

	return ok;
};

static int test_dag_seq2(const size_t items, virtual_device& device)
{

	//Simple task_graph consist of 3 tasks
	// 1 independent task A
	// 2 dependent tasks B(A),C(B)	
	/*
	<BEGIN>
	[A]
	 | 
	[B]
	 |
	[C]
	<END>
	*/

	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(B) = 13 >> C=B+2		
	int err = 0;
	
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Start DAG with 3 tasks, sequential ..." << std::endl;
	std::cout << "--------------------------------------" << std::endl;	

	
	auto task_A = device.create_task(tasks, "kA");	
	auto task_B = device.create_task(tasks, "kB");
	auto task_C = device.create_task(tasks, "kC");	

	map_device_info dev_cpu = { CL_DEVICE_TYPE_CPU,0 };
	map_device_info dev_gpu = { CL_DEVICE_TYPE_GPU,0 };
	
	std::vector<int> zeros(items, 0);

	for (int it = 0; it < 10; it++)
	{
		std::cout << "Execute\t";

		auto bA = device.alloc(zeros, true);
		auto bB = device.alloc(zeros, true);
		auto bC = device.alloc(zeros);

		const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU,0);
		const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU,0);

		const std::array<size_t, 3> gs = { items,1,1 };
		const std::array<size_t, 3> ls = { 16,1,1 };

		const offload_info offload_t1 = { {1.0f,cpu} };
		//Set all buffer values = 10
		err = device.execute_async(task_A, offload_t1, gs,ls, bA);
		on_coopcl_error(err);
		
		task_B->add_dependence(task_A.get());

		// each device executes the part of workload 
		const float offload = 0.25f;
		const offload_info offload_t2 = { {offload, cpu}, {offload, gpu} };
		err = device.execute_async(task_B, offload_t2, gs, ls, bA, bB);
		on_coopcl_error(err);
		

		//TODO: impl. add this
		// 
		// Gather chunks of memory to get coherent results
		//------------------
		// use offload to calculate offset
		/*auto data_split = static_cast<size_t>(std::floor(static_cast<float>(items) * offload / static_cast<float>(ls[0]))) * ls[0];
		const size_t offset = data_split * bB->item_size();
		
		auto wait_task = task_B->get_event_wait_kernel();
		clAppEvent wait_merge;
		err = bB->merge_async({ wait_task }, wait_merge, offset, bB->size(), dev_cpu , dev_gpu);
		on_coopcl_error(err);*/

		
		task_C->add_dependence(task_B.get());
		const offload_info offload_t3 = { { 1.0f,gpu } };
		err = device.execute_async(task_C, offload_t3, gs, ls, bB, bC);
		on_coopcl_error(err);
		
		err = task_C->wait();
		on_coopcl_error(err);

		std::cout << " Validate\t";
		auto begin_ptr = bC->data_in_buffer_device<int>(gpu);
		
		for (size_t i = 0; i < bC->items(); i++)
		{
			//const auto val = bC->at<int>(i);
			const auto val = begin_ptr[i];
			if (val != 13) {
				
				std::vector<int> tmp(items);
				memcpy(tmp.data(), begin_ptr, items * sizeof(int));

				std::cerr << " Some error at pos i = " << i << "\t{" << val << "!=" << 13 << "}" << std::endl;
				return -1;
			}
		}
		
		task_A->wait_clear_events();
		task_B->wait_clear_events();
		task_C->wait_clear_events();

		std::cout << "DONE!\n";
	}
	
	std::cout << "Passed ok, exit!" << std::endl;
	return 0;

}


static int test_dag_seq1(const size_t items, virtual_device& device)
{

	//Simple task_graph consist of 2 tasks
	/*
	* 
	a,b,c are buffers
	<BEGIN>
	a b  "a,b are inputs"
	| |
	[A]
	  |
	b c "c is output of taskA and than input of taskB"
	| |
	[B]
	 |
	 a "a is now output of taskB"

	<END>
	*/

	//task_A => a+b=c 
	//task_B => c+b=a
	
	int err = 0;

	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Start DAG with 2 tasks, sequential ..." << std::endl;
	std::cout << "--------------------------------------" << std::endl;


	auto task_A = device.create_task(tasks, "kF");
	auto task_B = device.create_task(tasks, "kF");

	const auto dev_cpu = std::make_pair(CL_DEVICE_TYPE_CPU,0);
	const auto dev_gpu = std::make_pair(CL_DEVICE_TYPE_GPU,0);
	
	
	std::vector<int> ones(items, 1);

	for (int it = 0; it < 10; it++)
	{
		std::cout << "Execute\t";

		auto bA = device.alloc(ones, true);
		auto bB = device.alloc(ones, true);
		auto bC = device.alloc<int>(items, nullptr, false);

		task_B->add_dependence(task_A.get());

		const offload_info offload_t1 = { {1.0f,dev_gpu} };	
		//a+b=c
		err = device.execute_async(task_A, offload_t1, { items,1,1 }, { 16,1,1 }, bA, bB, bC);
		on_coopcl_error(err);

		const offload_info offload_t2 = { {1.0f,dev_cpu} };
		//c+b=a
		err = device.execute_async(task_B, offload_t2, { items,1,1 }, { 16,1,1 }, bC, bB, bA);
		on_coopcl_error(err);

		err = task_B->wait();
		on_coopcl_error(err);

		std::cout << " Validate\t";
		auto begin_ptr = bA->data_in_buffer_device<int>(dev_cpu);

		for (size_t i = 0; i < bA->items(); i++)
		{
			const auto val = begin_ptr[i];
			const auto expected_val = 3;
			if (val != expected_val) {

				std::vector<int> tmp(items);
				memcpy(tmp.data(), begin_ptr, items * sizeof(int));

				std::cerr << " Some error at pos i = " << i << "\t{" << val << "!=" << expected_val << "}" << std::endl;
				return -1;
			}
		}
		
		task_A->wait_clear_events();
		task_B->wait_clear_events();

		std::cout << "DONE!\n";
	}

	std::cout << "Passed ok, exit!" << std::endl;
	return 0;

}

static int test_dag_par_3GPUs_CPU(const size_t items, virtual_device& device)
{
	//Simple task_graph consist of 5 tasks
	// 3 dependent, data parallel tasks B,C,D
	// 1 independent task A	
	// 1 dependent task E
	/*
	<BEGIN>
	   [A]
	  / | \
	[B][C][D]
	  \ | /
	   [E]
	<END>
	*/
	//A = 10 
	//B(A) = 11 >> B=A+1
	//C(A) = 12 >> C=A+2
	//D(A) = 13 >> D=A+3
	//E(B,C,D) = B+C+D = 11+12+13 = 36
	int err = 0;
	
	std::cout << "--------------------------------------------" << std::endl;
	std::cout << "Start DAG with 5 tasks, 3 parallel tasks ..." << std::endl;
	std::cout << "--------------------------------------------" << std::endl;
	
	auto task_A = device.create_task( tasks, "kA");
	auto task_B = device.create_task( tasks, "kB");
	auto task_C = device.create_task( tasks, "kC");
	auto task_D = device.create_task( tasks, "kD");
	auto task_E = device.create_task(tasks, "kE");
	
	size_t begin_byte = 0;
	size_t end_byte = items * sizeof(int);

	map_device_info cpu = { CL_DEVICE_TYPE_CPU,0 };
	map_device_info gpu_a = { CL_DEVICE_TYPE_GPU,0 };
	map_device_info gpu_b = { CL_DEVICE_TYPE_GPU,1 };
	map_device_info gpu_c = { CL_DEVICE_TYPE_GPU,2 };

	std::vector<int> zeros(items, 0);

	for (int it = 0; it < 10; it++)
	{
		std::cout << "Execute\t";


		auto bA = device.alloc(zeros, true, cpu);
		auto bB = device.alloc(zeros, true, cpu);
		auto bC = device.alloc(zeros, true, cpu);
		auto bD = device.alloc(zeros, true, cpu);
		auto bE = device.alloc(zeros, true, cpu);
		
		offload_info exec_a = { { 1.0f,cpu } };
		err = device.execute_async(task_A, exec_a, { items,1,1 }, { 16,1,1 }, bA);
		on_coopcl_error(err);

		//----------------------------------------
		//Execute tasks B on gpu_a
		offload_info exec_b = { { 1.0f,gpu_a } };
		err = device.execute_async(task_B, exec_b, { items,1,1 }, { 16,1,1 }, bA, bB);
		on_coopcl_error(err);


		//----------------------------------------
		//Execute tasks C on gpu_b
		offload_info exec_c = { { 1.0f,gpu_b } };
		err = device.execute_async(task_C, exec_c, { items,1,1 }, { 16,1,1 }, bA, bC);
		on_coopcl_error(err);

		
		//----------------------------------------
		//Execute tasks D on gpu_c
		offload_info exec_d = { { 1.0f,gpu_c } };
		err = device.execute_async(task_D, exec_d, { items,1,1 }, { 16,1,1 }, bA, bD);
		on_coopcl_error(err);

		//----------------------------------------
		//Execute tasks E on cpu
		offload_info exec_e = { { 1.0f,cpu } };
		err = device.execute_async(task_E, exec_e, { items,1,1 }, { 16,1,1 }, bB, bC, bD, bE);
		on_coopcl_error(err);

		err = task_E->wait();
		on_coopcl_error(err);
		
		std::cout << "Validate\t";
		//auto begin_ptr = static_cast<const int*>(bE->host_begin());
		auto begin_ptr = bE->data_in_buffer_device<int>(cpu);

		for (size_t i = 0; i < bE->items(); i++)
		{
			//const auto val = bE->at<int>(i);
			const auto val = begin_ptr[i];
			if (val != 36) {
				std::cerr << " Some error at pos i = " << i << "\t{" << val << "!=" << 36 << "}" << std::endl;
				return -1;
			}
		}

		task_A->wait_clear_events();
		task_B->wait_clear_events();
		task_C->wait_clear_events();
		task_D->wait_clear_events();
		task_E->wait_clear_events();

		std::cout << "DONE!\n";
	}
	std::cout << "Passed ok, exit!" << std::endl;
	return 0;
}

static int test_dag_par_2GPUs_CPU(const size_t items, virtual_device& device)
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

	for (int it = 0; it < 10; it++)
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
		err = device.execute_async(taskA, exec_a, ndr, wgs, mA);
		on_coopcl_error(err);

		
		offload_info exec_b = { { 1.0f,gpu_a } };
		err = device.execute_async(taskB, exec_b, ndr, wgs, mA, mB);
		on_coopcl_error(err);

		
		offload_info exec_c = { { 1.0f,gpu_b } };
		err = device.execute_async(taskC, exec_c, ndr, wgs, mA, mC);
		on_coopcl_error(err);

		
		offload_info exec_d = { { 1.0f,cpu } };
		err = device.execute_async(taskD, exec_d, ndr, wgs, mB, mC, mD);
		on_coopcl_error(err);
		
		err = taskD->wait();
		on_coopcl_error(err);

		//------------------------------------------------
		//DBG 
		/*std::vector<int> tmpb_cpu(items);
		auto mbh = mB->host_double_buffer_memory(gpu_a);
		std::memcpy(tmpb_cpu.data(), mbh, items * sizeof(int));

		std::vector<int> tmpc_cpu(items);
		auto mch = mC->host_double_buffer_memory(gpu_b);
		std::memcpy(tmpc_cpu.data(), mch, items * sizeof(int));

		std::vector<int> tmpb(items);		
		auto mb = mB->data_in_buffer<int>(gpu_a);
		std::memcpy(tmpb.data(), mb, items * sizeof(int));

		std::vector<int> tmpc(items);
		auto mc = mC->data_in_buffer<int>(gpu_b);
		std::memcpy(tmpc.data(), mc, items * sizeof(int));*/

		auto begin_ptr = mD->data_in_buffer_device<int>(cpu);
		//auto begin_ptr = static_cast<const int*>(mD->host_begin());

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

int main()
{
	int ok = 0;

	std::string status;
	virtual_device device(status);

	ok = test_dag_seq1(128, device);
	if (ok != 0)return ok;

	/*ok = test_dag_seq2(128, device);
	if (ok != 0)return ok;

	if (device.cnt_devices() > 2)
	{
		ok = test_dag_par_2GPUs_CPU(128 * 1e4, device);
		if (ok != 0)return ok;
	}

	if (device.cnt_devices() > 3)
	{
		ok = test_dag_par_3GPUs_CPU(1024 * 1e3, device);
		if (ok != 0)return ok;
	}*/
	return ok;
}
