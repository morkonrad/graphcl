#include "mkmd.h"


#include <random>
#include <execution>
#include <stdexcept>
#include <iomanip>
#include <thread>
#include <chrono>


static std::string tasks = "";

void kernels::init_kernels_mkmd()
{
	if (tasks.empty())
	{
		tasks.append("#define T float");
		tasks.append("\n");
		tasks.append(kernels::task_ADD_SUB);
		tasks.append("\n");
		tasks.append(kernels::task_ADD);
		tasks.append("\n");
		tasks.append(kernels::task_MERGE);
		tasks.append("\n");
		tasks.append(kernels::task_MM);
		tasks.append("\n");
		tasks.append(kernels::task_MT);
		//std::cout << tasks << std::endl;
	}
}

constexpr size_t BLOCK_DIM_global = 16;

template<typename T>
static void generate_rand_real(std::vector<T>& container, const T start, const T end)
{
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(static_cast<T>(start), static_cast<T>(end));

	std::for_each(std::execution::par_unseq, container.begin(), container.end(), [&gen, &dis](T& val) {
		val = static_cast<float>(dis(gen));
	});
}

int mkmd::Algebraic_Bernoulli_ABE(mkmd_input matrix_data)
{
	auto& device = *matrix_data.device();
	int err = 0;
	err = device.finish();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;
	const auto items = matrix_data.items();

	std::vector<float> random_values_a(items);
	std::vector<float> random_values_b(items);
	std::vector<float> random_values_c(items);

	generate_rand_real(random_values_a, 0.01f, 1.0f);
	generate_rand_real(random_values_b, 0.1f, 2.0f);
	generate_rand_real(random_values_c, 1.1f, 2.0f);

	//allocate memory
	auto matrix_a = device.alloc<float>(random_values_a, true);//read_only
	if (!matrix_a)return COOPCL_BAD_ALLOC;

	auto matrix_x = device.alloc<float>(random_values_b, true);//read_only
	if (!matrix_x)return COOPCL_BAD_ALLOC;

	auto matrix_b = device.alloc<float>(random_values_c, true);//read_only
	if (!matrix_b)return COOPCL_BAD_ALLOC;

	//------------------------------------------ READ_WRITE

	auto matrix_ax = device.alloc<float>(items);
	if (!matrix_ax)return COOPCL_BAD_ALLOC;

	auto matrix_at = device.alloc<float>(items);
	if (!matrix_at)return COOPCL_BAD_ALLOC;

	auto matrix_xat = device.alloc<float>(items);
	if (!matrix_xat)return COOPCL_BAD_ALLOC;

	auto matrix_bt = device.alloc<float>(items);
	if (!matrix_bt)return COOPCL_BAD_ALLOC;

	auto matrix_xb = device.alloc<float>(items);
	if (!matrix_xb)return COOPCL_BAD_ALLOC;

	auto matrix_btx = device.alloc<float>(items);
	if (!matrix_btx)return COOPCL_BAD_ALLOC;

	auto matrix_xbbtx = device.alloc<float>(items);
	if (!matrix_xbbtx)return COOPCL_BAD_ALLOC;

	auto matrix_results = device.alloc<float>(items);
	if (!matrix_results)return COOPCL_BAD_ALLOC;

	//Create tasks

	std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DBLOCK_DIM=";
	jit_flags.append(std::to_string(BLOCK_DIM_global));

	auto task_mat_mul_a = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_b = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_c = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_c)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_d = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_d)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_e = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_e)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose_a = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose_b = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_add_sub = device.create_task(tasks, "mat_add_sub", jit_flags);
	if (!task_mat_add_sub)throw std::runtime_error("Error JTI, FIMXE!!!");

	std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1 };
	std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

	const auto local_mem = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
	const int offset = 0;

	//DEFAULT
	const auto gid = matrix_data.gpu_id();

	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, gid);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gid);

	offload_info exec_mt_a = { {1.0f,gpu } };//4
	offload_info exec_mt_b = { {1.0f,gpu } };//5
	offload_info exec_mm_a = { {1.0f,gpu } };//2
	offload_info exec_mm_b = { {1.0f,gpu } };//3
	offload_info exec_mm_c = { {1.0f,gpu } };//6
	offload_info exec_mm_d = { {1.0f,gpu } };//7
	offload_info exec_mm_e = { {1.0f,gpu } };//8
	offload_info exec_add_sub = { {1.0f,gpu } };//9


	float app_duration_avg = 0;
	for (int i = 0; i < matrix_data.iterations(); i++)
	{

		std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
		const auto begin = std::chrono::system_clock::now();

		cl::LocalSpaceArg lmem = cl::Local(local_mem);

		const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
		const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
		const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
		const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

		if (matrix_data.schedule() == SCHEDULE_DEFAULT)
		{
			//A*X
			err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
				matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			//XB
			err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
				matrix_xb, matrix_x, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			//B^T*X
			err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
				matrix_btx, matrix_x, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			//A->A^T
			err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
				matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
			if (err != CL_SUCCESS)return err;

			//B->B^T
			err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
				matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
			if (err != CL_SUCCESS)return err;

			//X*A^T
			err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
				matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			//XB*XB^T
			err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
				matrix_xbbtx, matrix_xb, matrix_btx, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			//ADD = AX+AX^T+XB*XB^T
			err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
				matrix_results, matrix_ax, matrix_xat, matrix_xbbtx, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;
		}
		else
		{
			/* TASK-KERNEL-GRAPH
			*
			*
			*
			<START>
			|
			|						      B->B^T
			|						    (MT_a)_T3
			|				 			    |
			|		 		   X*B   	    |			   A->A^T
			|		         (MM_b)_T2	(MM_c)_T5 B^T*X	 (MT_b)_T4
			|		     	    |			|	     	  /
			|			A*X	    |			|		     /
			|		 T1_(MM_a)  (MM_e)_T7 X*B*B^T*X	 (MM_d)_T6 A^T*X
			|			     \			|			 /
							  \			|			/
							   \		|		   /
							   (ADD_SUB)_T8	//ABE = A^T*X + X*A - X*B * B^T*X
			|
			<END>
			*/
			task_mat_mul_c->add_dependence(task_mat_transpose_a.get());
			task_mat_mul_d->add_dependence(task_mat_transpose_b.get());

			task_mat_mul_e->add_dependence(task_mat_mul_b.get());
			task_mat_mul_e->add_dependence(task_mat_mul_c.get());

			task_mat_add_sub->add_dependence(task_mat_mul_a.get());
			task_mat_add_sub->add_dependence(task_mat_mul_e.get());
			task_mat_add_sub->add_dependence(task_mat_mul_d.get());

			if (matrix_data.schedule() == SCHEDULE_COARSE)
			{
				//mapping
				exec_mt_a = { {1.0f,gpu0 } };//4
				exec_mm_c = { {1.0f,gpu0 } };//6
				exec_mm_d = { {1.0f,gpu0 } };//7

				exec_mm_b = { {1.0f,gpu1} };//3
				exec_mm_a = { {1.0f,gpu1} };//2
				exec_mm_e = { {1.0f,gpu1} };//8
				exec_add_sub = { {1.0f,gpu1 } };//9

				exec_mt_b = { {1.0f,cpu } };//5

				//schedule-enqueue order

				//CPU
				err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				//GPU_0
				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_btx, matrix_x, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU 1
				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_xb, matrix_x, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_xbbtx, matrix_xb, matrix_btx, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
					matrix_results, matrix_ax, matrix_xat, matrix_xbbtx, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

			}
			else if (matrix_data.schedule() == SCHEDULE_FINE)
			{
				//mapping
				exec_mt_a = { {1.0f,gpu0 } };//4
				exec_mm_c = { {1.0f,gpu0 } };//6
				exec_mm_d = { {1.0f,gpu0 } };//7

				exec_mm_b = { {1.0f,gpu1 } };//3
				exec_mm_a = { {1.0f,gpu1 } };//2

				if (device.cnt_devices() == 3)
					exec_mm_e = { {0.5f,gpu0 },{0.5f,gpu1 } };//8
				else if (device.cnt_devices() == 4)
					exec_mm_e = { {0.4f,gpu0 },{0.4f,gpu1 }, {0.2f,gpu2 } };//8

				exec_add_sub = { {1.0f,gpu1 } };//9

				exec_mt_b = { {1.0f,cpu } };//5

				//schedule-enqueue order

				//CPU
				err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				//GPU_0
				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_btx, matrix_x, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU 1
				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_xb, matrix_x, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_xbbtx, matrix_xb, matrix_btx, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
					matrix_results, matrix_ax, matrix_xat, matrix_xbbtx, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

			}
		}

		err = device.flush();
		if (err != CL_SUCCESS)return err;

		err = task_mat_add_sub->wait();
		if (err != CL_SUCCESS)return err;

		task_mat_mul_a->wait_clear_events();
		task_mat_mul_b->wait_clear_events();
		task_mat_mul_c->wait_clear_events();
		task_mat_mul_d->wait_clear_events();
		task_mat_mul_e->wait_clear_events();

		task_mat_transpose_a->wait_clear_events();
		task_mat_transpose_b->wait_clear_events();
		task_mat_add_sub->wait_clear_events();

		const auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - begin;
		auto app_duration = diff.count() * 1e3f;//msec
		app_duration_avg += app_duration;
	}
	app_duration_avg = app_duration_avg / matrix_data.iterations();
	std::cout << "ABE duration:\t" << app_duration_avg << " ms" << std::endl;
	std::cout << "---------------------------------------" << std::endl;

	return err;
}

int mkmd::Generalized_Algebraic_Bernoulli_GABE(mkmd_input matrix_data)
{
	auto& device = *matrix_data.device();
	int err = 0;
	err = device.finish();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;
	const auto items = matrix_data.items();

	std::vector<float> random_values_a(items);
	std::vector<float> random_values_b(items);
	std::vector<float> random_values_c(items);
	std::vector<float> random_values_d(items);

	generate_rand_real(random_values_a, 0.01f, 1.0f);
	generate_rand_real(random_values_b, 0.1f, 2.0f);
	generate_rand_real(random_values_c, 1.1f, 2.0f);
	generate_rand_real(random_values_d, 2.1f, 2.5f);

	// GABE = A^T * X*E + E^T * X*A - E^T* X*G * X*E

	//allocate memory
	auto matrix_a = device.alloc<float>(random_values_a, true);//read_only
	if (!matrix_a)return COOPCL_BAD_ALLOC;

	auto matrix_x = device.alloc<float>(random_values_b, true);//read_only
	if (!matrix_x)return COOPCL_BAD_ALLOC;

	auto matrix_g = device.alloc<float>(random_values_c, true);//read_only
	if (!matrix_g)return COOPCL_BAD_ALLOC;

	auto matrix_e = device.alloc<float>(random_values_d, true);//read_only
	if (!matrix_e)return COOPCL_BAD_ALLOC;

	//------------------------------------------ READ_WRITE

	auto matrix_ax = device.alloc<float>(items);
	if (!matrix_ax)return COOPCL_BAD_ALLOC;

	auto matrix_at = device.alloc<float>(items);
	if (!matrix_at)return COOPCL_BAD_ALLOC;

	auto matrix_xe = device.alloc<float>(items);
	if (!matrix_xe)return COOPCL_BAD_ALLOC;

	auto matrix_et = device.alloc<float>(items);
	if (!matrix_et)return COOPCL_BAD_ALLOC;

	auto matrix_xg = device.alloc<float>(items);
	if (!matrix_xg)return COOPCL_BAD_ALLOC;

	auto matrix_etxa = device.alloc<float>(items);
	if (!matrix_etxa)return COOPCL_BAD_ALLOC;

	auto matrix_etxg = device.alloc<float>(items);
	if (!matrix_etxg)return COOPCL_BAD_ALLOC;

	auto matrix_atxe = device.alloc<float>(items);
	if (!matrix_atxe)return COOPCL_BAD_ALLOC;

	auto matrix_etxgxe = device.alloc<float>(items);
	if (!matrix_etxgxe)return COOPCL_BAD_ALLOC;

	auto matrix_results = device.alloc<float>(items);
	if (!matrix_results)return COOPCL_BAD_ALLOC;

	//Create tasks

	std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DBLOCK_DIM=";
	jit_flags.append(std::to_string(BLOCK_DIM_global));

	auto task_mat_mul_a = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_b = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_c = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_c)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_d = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_d)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_e = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_e)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_f = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_f)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_g = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_g)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_h = device.create_task(tasks, "mat_mul_no_opt", jit_flags);
	if (!task_mat_mul_h)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose_a = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose_b = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_add_sub = device.create_task(tasks, "mat_add_sub", jit_flags);
	if (!task_mat_add_sub)throw std::runtime_error("Error JTI, FIMXE!!!");



	std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1 };
	std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

	const auto local_mem = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
	const int offset = 0;
	cl::LocalSpaceArg lmem = cl::Local(local_mem);
	const auto gid = matrix_data.gpu_id();

	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gid);

	offload_info exec_mt_a = { {1.0f,gpu } };
	offload_info exec_mt_b = { {1.0f,gpu } };
	offload_info exec_mm_a = { {1.0f,gpu } };
	offload_info exec_mm_b = { {1.0f,gpu } };
	offload_info exec_mm_c = { {1.0f,gpu } };
	offload_info exec_mm_d = { {1.0f,gpu } };
	offload_info exec_mm_e = { {1.0f,gpu } };
	offload_info exec_mm_f = { {1.0f,gpu } };
	offload_info exec_mm_g = { {1.0f,gpu } };
	offload_info exec_mm_h = { {1.0f,gpu } };
	offload_info exec_add_sub = { {1.0f,gpu } };

	float app_duration_avg = 0;
	for (int i = 0; i < matrix_data.iterations(); i++)
	{
		std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
		const auto begin = std::chrono::system_clock::now();

		const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
		const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
		const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
		const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

		if (matrix_data.schedule() == SCHEDULE_DEFAULT)
		{
			err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
				matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
				matrix_et, matrix_e, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
				matrix_xe, matrix_e, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
				matrix_ax, matrix_x, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
				matrix_xg, matrix_x, matrix_g, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
				matrix_xe, matrix_x, matrix_e, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
				matrix_atxe, matrix_at, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_f, exec_mm_f, global_sizes, local_sizes,
				matrix_etxa, matrix_et, matrix_ax, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_g, exec_mm_e, global_sizes, local_sizes,
				matrix_etxg, matrix_et, matrix_xg, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_mul_h, exec_mm_h, global_sizes, local_sizes,
				matrix_etxgxe, matrix_etxg, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;

			err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
				matrix_results, matrix_atxe, matrix_etxa, matrix_etxgxe, matrix_data._matrix_width, matrix_data._matrix_height);
			if (err != CL_SUCCESS)return err;
		}
		else
		{
			/* TASK-KERNEL-GRAPH
			  *
			  *
			  *
			  <START>
			  |
			  |				X*E		  A->A^T		   A*X    	E->E^T		X*G			X*E
			  |				 |			|				|	  (MT_b)_T4   (MM_c)_T5      |
			  |				 |			|				|		/   \			|		 |
			  |		 		 |	   		|				|	   /	 \			|		 |
			  |		  	 (MM_a)_T1	 (MT_a)_T2		(MM_b)_T3 /  	  (MM_g)_T9 E^T*X*G (MM_d)_T6
			  |		   	 	 |	        |               |	 /					|		 |
			  |				(MM_e)_T7 A^T*X*E     (MM_f)_T8 E^T*X*A	  (MM_h)_T10 E^T*X*G*X*E
			  |						   \				|				 /
			  |						    \				|				/
			  |							(ADD_SUB)_T11 // GABE = A^T * X*E + E^T * X*A - E^T* X*G * X*E
			  |
			  <END>
			  */

			task_mat_mul_e->add_dependence(task_mat_transpose_a.get());
			task_mat_mul_e->add_dependence(task_mat_mul_a.get());

			task_mat_mul_f->add_dependence(task_mat_mul_b.get());
			task_mat_mul_f->add_dependence(task_mat_transpose_b.get());

			task_mat_mul_g->add_dependence(task_mat_transpose_b.get());
			task_mat_mul_g->add_dependence(task_mat_mul_c.get());

			task_mat_mul_h->add_dependence(task_mat_mul_d.get());
			task_mat_mul_h->add_dependence(task_mat_mul_g.get());

			task_mat_add_sub->add_dependence(task_mat_mul_e.get());
			task_mat_add_sub->add_dependence(task_mat_mul_f.get());
			task_mat_add_sub->add_dependence(task_mat_mul_h.get());

			if (matrix_data.schedule() == SCHEDULE_COARSE)
			{
				//mapping
				exec_mm_c = { {1.0f,gpu0 } };//6
				exec_mm_b = { {1.0f,gpu0 } };//4
				exec_mm_g = { {1.0f,gpu0 } };//10
				exec_mm_f = { {1.0f,gpu0 } };//9

				exec_mt_b = { {1.0f,gpu1 } };//5
				exec_mm_a = { {1.0f,gpu1 } };//2
				exec_mm_d = { {1.0f,gpu1 } };//7
				exec_mm_e = { {1.0f,gpu1 } };//8
				exec_mm_h = { {1.0f,gpu1 } };//11
				exec_add_sub = { {1.0f,gpu1 } };//12

				exec_mt_a = { {1.0f,cpu } };//3


				//schedule-enqueue order
				//CPU
				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				//GPU_0
				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_xg, matrix_x, matrix_g, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_ax, matrix_x, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU_1 (need enque before MM_G)
				err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
					matrix_et, matrix_e, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				//GPU0
				err = device.execute_async(task_mat_mul_g, exec_mm_g, global_sizes, local_sizes,
					matrix_etxg, matrix_et, matrix_xg, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_f, exec_mm_f, global_sizes, local_sizes,
					matrix_etxa, matrix_et, matrix_ax, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU1
				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_xe, matrix_e, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_xe, matrix_x, matrix_e, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_atxe, matrix_at, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_h, exec_mm_h, global_sizes, local_sizes,
					matrix_etxgxe, matrix_etxg, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
					matrix_results, matrix_atxe, matrix_etxa, matrix_etxgxe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;



			}
			else if (matrix_data.schedule() == SCHEDULE_FINE)
			{
				//mapping
				exec_mm_c = { {1.0f,gpu0 } };//6
				exec_mm_b = { {1.0f,gpu0 } };//4
				exec_mm_g = { {1.0f,gpu0 } };//10
				exec_mm_f = { {1.0f,gpu0 } };//9

				exec_mt_b = { {1.0f,gpu1 } };//5
				exec_mm_a = { {1.0f,gpu1 } };//2
				exec_mm_d = { {1.0f,gpu1 } };//7
				exec_mm_e = { {1.0f,gpu1 } };//8
				exec_mm_h = { {1.0f,gpu1 } };//11
				exec_add_sub = { {1.0f,gpu1} };//12

				exec_mt_a = { {1.0f,cpu } };//3


				//schedule-enqueue order
				//CPU
				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				//GPU_0
				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_xg, matrix_x, matrix_g, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_ax, matrix_x, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU_1
				err = device.execute_async(task_mat_transpose_b, exec_mt_b, global_sizes, local_sizes,
					matrix_et, matrix_e, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_xe, matrix_e, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_xe, matrix_x, matrix_e, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU0
				err = device.execute_async(task_mat_mul_g, exec_mm_g, global_sizes, local_sizes,
					matrix_etxg, matrix_et, matrix_xg, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_f, exec_mm_f, global_sizes, local_sizes,
					matrix_etxa, matrix_et, matrix_ax, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				//GPU1
				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_atxe, matrix_at, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_h, exec_mm_h, global_sizes, local_sizes,
					matrix_etxgxe, matrix_etxg, matrix_xe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add_sub, exec_add_sub, global_sizes, local_sizes,
					matrix_results, matrix_atxe, matrix_etxa, matrix_etxgxe, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;
			}
		}

		err = device.flush();
		if (err != CL_SUCCESS)return err;

		err = task_mat_add_sub->wait();
		if (err != CL_SUCCESS)return err;


		task_mat_mul_a->wait_clear_events();
		task_mat_mul_b->wait_clear_events();
		task_mat_mul_c->wait_clear_events();
		task_mat_mul_d->wait_clear_events();
		task_mat_mul_e->wait_clear_events();
		task_mat_mul_f->wait_clear_events();
		task_mat_mul_g->wait_clear_events();
		task_mat_mul_h->wait_clear_events();
		task_mat_transpose_a->wait_clear_events();
		task_mat_transpose_b->wait_clear_events();
		task_mat_add_sub->wait_clear_events();



		const auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - begin;
		auto app_duration = diff.count() * 1e3f;//msec
		app_duration_avg += app_duration;
	}
	app_duration_avg = app_duration_avg / matrix_data.iterations();
	std::cout << "GABE duration:\t" << app_duration_avg << " ms" << std::endl;
	std::cout << "---------------------------------------" << std::endl;

	/*
	auto duration_ms_task_add = task_mat_add_sub->duration(ms_scale_factor);
	std::cout << "Task_ADD:\t" << duration_ms_task_add << std::endl;
	//ADD: (1Kx1K) CPU 1.6, GPU_0 0.15, GPU_1 0.1
	*/

	return err;
}

int mkmd::SVD(mkmd_input matrix_data)
{
	auto& device = *matrix_data.device();

	int err = 0;
	err = device.finish();
	if(on_coopcl_error(err)!=CL_SUCCESS)return err;

	const auto items = matrix_data.items();

	std::vector<float> random_values_a(items);
	std::vector<float> random_values_b(items);
	std::vector<float> random_values_c(items);

	generate_rand_real(random_values_a, 0.01f, 1.0f);
	generate_rand_real(random_values_b, 0.5f, 1.0f);
	generate_rand_real(random_values_c, 1.0f, 2.0f);

	//allocate memory
	auto matrix_u = device.alloc<float>(random_values_a,true);//read_only
	if (!matrix_u)return COOPCL_BAD_ALLOC;

	auto matrix_e = device.alloc<float>(random_values_b,true);//read_only
	if (!matrix_e)return COOPCL_BAD_ALLOC;

	auto matrix_v = device.alloc<float>(random_values_c, true); //read_only
	if (!matrix_v)return COOPCL_BAD_ALLOC;

	//------------------------------------------ READ_WRITE

	auto matrix_ue = device.alloc<float>(items);
	if (!matrix_ue)return COOPCL_BAD_ALLOC;

	auto matrix_vt = device.alloc<float>(items);
	if (!matrix_vt)return COOPCL_BAD_ALLOC;

	auto matrix_svd = device.alloc<float>(items);
	if (!matrix_svd)return COOPCL_BAD_ALLOC;

	//Create tasks
	std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DT=float -DBLOCK_DIM=";
	jit_flags.append(std::to_string(BLOCK_DIM_global));

	auto task_mat_mul_a = device.create_task(tasks, "matrixMul", jit_flags);
	if (!task_mat_mul_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_b = device.create_task(tasks, "matrixMul", jit_flags);
	if (!task_mat_mul_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose)throw std::runtime_error("Error JTI, FIMXE!!!");

	/* TASK-KERNEL-GRAPH
	*
	* SVD = U*E*V^T
	*
	<START>
	|
	|	U*E		  V->V^T
	|  T1_(MM_a)  (MT)_T2
	|		\    /
	|		 \  /
	|		 (MM_b)_T3	SVD=U*E * V^T
	|
	|
	|
	<END> TIME
	*/
	

	std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1};
	std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };
	
	const auto gid = matrix_data.gpu_id();
	
	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gid);
	
	offload_info exec_mm_a = { {1.0f,cpu } };
	offload_info exec_mt = { {1.0f,gpu } };
	offload_info exec_mm_b = { {1.0f,gpu } };

	const auto local_mem_mt = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
	cl::LocalSpaceArg lmem_mt = cl::Local(local_mem_mt);
	const int offset = 0;

	float app_duration_avg = 0;

	const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
	const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
	const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

	//NOTE: with this init. the code expects a square matrices A,B,C !
				//const int uiWA = matrix_data.width();
				//const int uiWB = uiWA;
				//const int trueLocalSize1 = matrix_data.height();
	const auto local_mem_mm = BLOCK_DIM_global * BLOCK_DIM_global * sizeof(float);
	cl::LocalSpaceArg lmem_mm = cl::Local(local_mem_mm);


	for (int i = 0; i < matrix_data.iterations(); i++)
	{
		std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r"<< std::flush;
		const auto begin = std::chrono::system_clock::now();

		if (matrix_data.schedule() == SCHEDULE_DEFAULT)
		{
			exec_mm_a = { {1.0f,gpu } };
			exec_mt = { {1.0f,gpu } };
			exec_mm_b = { {1.0f,gpu } };			
		}
		else
		{
			task_mat_mul_b->add_dependence(task_mat_mul_a.get());
			task_mat_mul_b->add_dependence(task_mat_transpose.get());

			if (matrix_data.schedule() == SCHEDULE_COARSE)
			{
				exec_mm_a = { {1.0f,gpu0 } };
				exec_mt = { {1.0f,cpu } };
				exec_mm_b = { {1.0f,gpu0 } };
			}
			else if (matrix_data.schedule() == SCHEDULE_FINE)
			{
				if (matrix_data.device()->cnt_gpus()==2) //2 GPUS
				{					
					exec_mm_a = { {0.5f,gpu0 } , {0.5f,gpu1 } };
					exec_mt = { {1.0f,cpu } };
					exec_mm_b = { {0.5f,gpu0 } , {0.5f,gpu1 } };
				}
				else if (matrix_data.device()->cnt_gpus()==3) 
				{
					exec_mm_a = { {0.52f,gpu0 },{0.3f,gpu1 },{0.18f,gpu2 } };
					exec_mt = { {1.0f,cpu } };					
					exec_mm_b = { {0.52f,gpu0 },{0.3f,gpu1 },{0.18f,gpu2 } };
				}
				else
				{
					exec_mm_a = { {1.0f,gpu0 } };
					exec_mt = { {1.0f,cpu } };
					exec_mm_b = { {1.0f,gpu0 } };
				}				
			}
		}

		//enqueue order MM->MT->MM
		err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
			matrix_ue, matrix_u, matrix_e, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
		if (err != CL_SUCCESS)return err;

		err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
			matrix_vt, matrix_v, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
		if (err != CL_SUCCESS)return err;

		err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
			matrix_svd, matrix_ue, matrix_vt, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
		if (err != CL_SUCCESS)return err;

		err = device.flush();
		if (err != CL_SUCCESS)return err;

		err = task_mat_mul_b->wait();
		if (err != CL_SUCCESS)return err;
		

		const auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - begin;
		auto app_duration = diff.count() * 1e3f;//msec
		app_duration_avg += app_duration;

		task_mat_mul_a->wait_clear_events();
		task_mat_mul_b->wait_clear_events();
		task_mat_transpose->wait_clear_events();
	}

	app_duration_avg = app_duration_avg / matrix_data.iterations();
	std::cout << "SVD duration:\t" << app_duration_avg << " ms"<<std::endl;;
	std::cout << "---------------------------------------"<<std::endl;

	return err;
}

int mkmd::Continuous_Lyapunov(mkmd_input matrix_data)
{
	auto& device = *matrix_data.device();
	int err = 0;
	err = device.finish();
	if(on_coopcl_error(err)!=CL_SUCCESS)return err;
	const auto items = matrix_data.items();

	std::vector<float> random_values(items);
	generate_rand_real(random_values, 0.01f, 1.0f);

	//allocate memory
	auto matrix_a = device.alloc<float>(random_values, true);//read_only
	if (!matrix_a)return COOPCL_BAD_ALLOC;

	auto matrix_x = device.alloc<float>(random_values, true);//read_only
	if (!matrix_x)return COOPCL_BAD_ALLOC;

	auto matrix_q = device.alloc<float>(random_values, true);//read_only
	if (!matrix_q)return COOPCL_BAD_ALLOC;

	//------------------------------------------ READ_WRITE

	auto matrix_ax = device.alloc<float>(items);
	if (!matrix_ax)return COOPCL_BAD_ALLOC;

	auto matrix_at = device.alloc<float>(items);
	if (!matrix_at)return COOPCL_BAD_ALLOC;

	auto matrix_xat = device.alloc<float>(items);
	if (!matrix_xat)return COOPCL_BAD_ALLOC;

	auto matrix_results = device.alloc<float>(items);
	if (!matrix_results)return COOPCL_BAD_ALLOC;

	//Create tasks

	std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DT=float -DBLOCK_DIM=";
	jit_flags.append(std::to_string(BLOCK_DIM_global));

	//There are two version of mat_multiply kernel with a different interface and performance
	const auto matrix_mul_name = "matrixMul";

	auto task_mat_mul_a = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_b = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_transpose = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_add = device.create_task(tasks, "mat_add", jit_flags);
	if (!task_mat_add)throw std::runtime_error("Error JTI, FIMXE!!!");	

	std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1};
	std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

	const auto gid = matrix_data.gpu_id();
	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, gid);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gid);

	offload_info exec_mm_a	= { {1.0f,gpu } };
	offload_info exec_mt	= { {1.0f,gpu } };
	offload_info exec_mm_b	= { {1.0f,gpu } };
	offload_info exec_add	= { {1.0f,gpu } };

	const auto local_mem_mt = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
	cl::LocalSpaceArg lmem_mt = cl::Local(local_mem_mt);

	//NOTE: with this init. the code expects a square matrices A,B,C !
				//const int uiWA = matrix_data.width();
				//const int uiWB = uiWA;
				//const int trueLocalSize1 = matrix_data.height();
	const auto local_mem_mm = BLOCK_DIM_global * BLOCK_DIM_global * sizeof(float);
	cl::LocalSpaceArg lmem_mm = cl::Local(local_mem_mm);
	
	const int offset = 0;
	float app_duration_avg = 0;
	for (int i = 0; i < matrix_data.iterations(); i++)
	{
		std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
		const auto begin = std::chrono::system_clock::now();

		

		const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
		const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
		const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
		const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);

		if (matrix_data.schedule() == SCHEDULE_DEFAULT)
		{
			if (matrix_mul_name == "mat_mul_no_opt")
			{
				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
					matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;
			}
			else if (matrix_mul_name == "matrixMul")
			{				
				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_ax, matrix_a, matrix_x, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
					matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_xat, matrix_x, matrix_at, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
					matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;
			}

		}
		else
		{
			/* TASK-KERNEL-GRAPH
			*
			* con_lyap = A*X + X*A^T + Q
			*
			<START>
			|
			|	A*X		  A->A^T
			|  T1_(MM_a)  (MT)_T2
			|		\		 |
			|		 \		 |
			|		  \		(MM_b)_T3 X*A^T
			|		   \     |
						\	 |
			|			 (ADD)_T4 con_lyap = A*X + X*A^T + Q
			|
			<END> TIME
			*/
			task_mat_mul_b->add_dependence(task_mat_transpose.get());
			task_mat_add->add_dependence(task_mat_mul_a.get());
			task_mat_add->add_dependence(task_mat_mul_b.get());

			if (matrix_data.schedule() == SCHEDULE_COARSE)
			{
				//mapping
				exec_mm_a = { {1.0f,gpu1 } };
				exec_mt = { {1.0f,gpu0 } };
				exec_mm_b = { {1.0f,gpu0 } };
				exec_add = { {1.0f,gpu0 } };

				if (matrix_mul_name == "mat_mul_no_opt")
				{
					//enqueue-schedule_order
					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
						matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
						matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;
				}
				else if (matrix_mul_name == "matrixMul")
				{

					//enqueue-schedule_order
					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_ax, matrix_a, matrix_x, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
						matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_xat, matrix_x, matrix_at, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
						matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;
				}
			}
			else if (matrix_data.schedule() == SCHEDULE_FINE)
			{
				//mapping
				exec_mm_a = { {1.0f,gpu1 } };
				exec_mt = { {1.0f,gpu0 } };
				exec_mm_b = { {1.0f,gpu0 } };
				exec_add = { {0.5f,gpu0 },{0.5f,gpu1 } };
				if (matrix_mul_name == "mat_mul_no_opt")
				{
					//enqueue-schedule_order
					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_ax, matrix_a, matrix_x, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
						matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_xat, matrix_x, matrix_at, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
						matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;
				}
				else if (matrix_mul_name == "matrixMul")
				{
					//enqueue-schedule_order
					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_ax, matrix_a, matrix_x, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose, exec_mt, global_sizes, local_sizes,
						matrix_at, matrix_a, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_xat, matrix_x, matrix_at, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_add, exec_add, global_sizes, local_sizes,
						matrix_results, matrix_ax, matrix_xat, matrix_q, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;
				}
			}
		}

		err = device.flush();
		if (err != CL_SUCCESS)return err;

		err = task_mat_add->wait();
		if (err != CL_SUCCESS)return err;

		task_mat_mul_a->wait_clear_events();
		task_mat_mul_b->wait_clear_events();
		task_mat_transpose->wait_clear_events();
		task_mat_add->wait_clear_events();


		const auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - begin;
		auto app_duration = diff.count() * 1e3f;//msec
		app_duration_avg += app_duration;
	}
	app_duration_avg = app_duration_avg / matrix_data.iterations();
	std::cout << "CLYAP duration:\t" << app_duration_avg << " ms"<<std::endl;
	std::cout << "---------------------------------------"<<std::endl;

	return err;
}

int mkmd::Matrix_equation(mkmd_input matrix_data)
{
	auto& device = *matrix_data.device();
	int err = 0;
	err = device.finish();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	const auto items = matrix_data.items();

	std::vector<float> random_values_a(items);
	std::vector<float> random_values_b(items);
	std::vector<float> random_values_c(items);

	generate_rand_real(random_values_a, 0.01f, 1.0f);
	generate_rand_real(random_values_b, 0.1f, 2.0f);
	generate_rand_real(random_values_c, 1.1f, 2.0f);

	//allocate memory
	auto matrix_a = device.alloc<float>(random_values_a, true);//read_only
	if (!matrix_a)return COOPCL_BAD_ALLOC;

	auto matrix_b = device.alloc<float>(random_values_b, true);//read_only
	if (!matrix_b)return COOPCL_BAD_ALLOC;

	auto matrix_c = device.alloc<float>(random_values_c, true);//read_only
	if (!matrix_c)return COOPCL_BAD_ALLOC;

	//------------------------------------------ READ_WRITE

	auto matrix_aa = device.alloc<float>(items);
	if (!matrix_aa)return COOPCL_BAD_ALLOC;

	auto matrix_bt = device.alloc<float>(items);
	if (!matrix_bt)return COOPCL_BAD_ALLOC;

	auto matrix_cb = device.alloc<float>(items);
	if (!matrix_cb)return COOPCL_BAD_ALLOC;

	auto matrix_bbt = device.alloc<float>(items);
	if (!matrix_bbt)return COOPCL_BAD_ALLOC;

	auto matrix_bbtcb = device.alloc<float>(items);
	if (!matrix_bbtcb)return COOPCL_BAD_ALLOC;

	auto matrix_results = device.alloc<float>(items);
	if (!matrix_results)return COOPCL_BAD_ALLOC;

	//Create tasks
	std::string jit_flags = " -cl-unsafe-math-optimizations -cl-fast-relaxed-math -DT=float -DBLOCK_DIM=";
	jit_flags.append(std::to_string(BLOCK_DIM_global));
	
	//There are two version on mat_multiply kernel with a different interface and performance
	const auto matrix_mul_name = "matrixMul";

	auto task_mat_transpose_a = device.create_task(tasks, "mat_transpose", jit_flags);
	if (!task_mat_transpose_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_a = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_a)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_b = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_b)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_c = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_c)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_d = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_d)throw std::runtime_error("Error JTI, FIMXE!!!");

	auto task_mat_mul_e = device.create_task(tasks, matrix_mul_name, jit_flags);
	if (!task_mat_mul_e)throw std::runtime_error("Error JTI, FIMXE!!!");

	std::array<size_t, 3> global_sizes = { static_cast<size_t>(matrix_data._matrix_width), static_cast<size_t>(matrix_data._matrix_height),1 };
	std::array<size_t, 3> local_sizes = { BLOCK_DIM_global,BLOCK_DIM_global,1 };

	const auto local_mem_mt = (BLOCK_DIM_global + 1) * BLOCK_DIM_global * sizeof(float);
	const int offset = 0;

	///default schedule 
	const auto gid = matrix_data.gpu_id();

	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, gid);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gid);

	offload_info exec_mt_a = { {1.0f,gpu } };
	offload_info exec_mm_a = { {1.0f,gpu } };
	offload_info exec_mm_b = { {1.0f,gpu } };
	offload_info exec_mm_c = { {1.0f,gpu } };
	offload_info exec_mm_d = { {1.0f,gpu } };
	offload_info exec_mm_e = { {1.0f,gpu } };

	cl::LocalSpaceArg lmem_mt = cl::Local(local_mem_mt);

	//NOTE: with this init. the code expects a square matrices A,B,C !
				//const int uiWA = matrix_data.width();
				//const int uiWB = uiWA;
				//const int trueLocalSize1 = matrix_data.height();
	const auto local_mem_mm = BLOCK_DIM_global * BLOCK_DIM_global * sizeof(float);
	cl::LocalSpaceArg lmem_mm = cl::Local(local_mem_mm);

	float app_duration_avg = 0;
	for (int i = 0; i < matrix_data.iterations(); i++)
	{
		std::cout << "Iteration:\t[" << i << "/" << matrix_data.iterations() << "]\r" << std::flush;
		const auto begin = std::chrono::system_clock::now();

		const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
		const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
		const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);
		const auto gpu2 = std::make_pair(CL_DEVICE_TYPE_GPU, 2);
		
		if (matrix_data.schedule() == SCHEDULE_DEFAULT)
		{
			//There are two version on mat_multiply kernel with a different interface and performance
			if (matrix_mul_name == "mat_mul_no_opt")
			{
				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_aa, matrix_a, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_cb, matrix_c, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_bbt, matrix_b, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_bbtcb, matrix_bbt, matrix_cb, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_results, matrix_aa, matrix_bbtcb, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

			}
			else if (matrix_mul_name == "matrixMul")
			{
				err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
					matrix_aa, matrix_a, matrix_a, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
					matrix_cb, matrix_c, matrix_b, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
					matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
					matrix_bbt, matrix_b, matrix_bt, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
					matrix_bbtcb, matrix_bbt, matrix_cb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;

				err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
					matrix_results, matrix_aa, matrix_bbtcb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
				if (err != CL_SUCCESS)return err;
			}
		}
		else
		{
			/* TASK-KERNEL-GRAPH
			*
			*
			*
			<START>
			|
			|	A*A			C*B		  B->B^T
			|  T1_(MM_a)  (MM_b)_T2	 (MT_a)_T3
			|		\		 |			|
			|		 \		 |	   		|
			|		  \		 |		(MM_c)_T4 B^T*B
			|		   \	 |			|
			|			\	 |			|
			|			 \ (MM_d)_T5 C*B* B*B^T
			|			  \			|
			|			   \		|
			|				\		|
			|				 \		|
			|				  \	    |
			|                 (MM_e)_T6		//Result = A^2* B*B^T* C*B
			|
			<END>
			*/

			task_mat_mul_c->add_dependence(task_mat_transpose_a.get());

			task_mat_mul_d->add_dependence(task_mat_mul_b.get());
			task_mat_mul_d->add_dependence(task_mat_mul_c.get());

			task_mat_mul_e->add_dependence(task_mat_mul_a.get());
			task_mat_mul_e->add_dependence(task_mat_mul_d.get());

			if (matrix_data.schedule() == SCHEDULE_COARSE)
			{
				if (device.cnt_gpus() > 1)
				{
					//mapping
					exec_mt_a = { {1.0f,cpu} };//4            
					exec_mm_b = { {1.0f,gpu1 } };//3
					exec_mm_a = { {1.0f,gpu1 } };//2
					exec_mm_c = { {1.0f,gpu0 } };//5
					exec_mm_d = { {1.0f,gpu0 } };//6
					exec_mm_e = { {1.0f,gpu0 } };//7
				}
				else
				{
					exec_mt_a = { {1.0f,cpu} };//4            
					exec_mm_b = { {1.0f,gpu0 } };//3
					exec_mm_a = { {1.0f,gpu0 } };//2
					exec_mm_c = { {1.0f,gpu0 } };//5
					exec_mm_d = { {1.0f,gpu0 } };//6
					exec_mm_e = { {1.0f,gpu0 } };//7
				}

				//Schedule-mapping GPU_0 TASK[3,2] 
				//Schedule-mapping GPU_1 TASK[4,5,6,7]

				//There are two version on mat_multiply kernel with a different interface and performance
				if (matrix_mul_name == "mat_mul_no_opt")
				{
					//enqueue_order
					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_cb, matrix_c, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_aa, matrix_a, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
						matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					//matrix_bt->set_as_input(true);

					err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
						matrix_bbt, matrix_b, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_bt->set_as_input(false);
					//matrix_cb->set_as_input(true);

					err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
						matrix_bbtcb, matrix_bbt, matrix_cb, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_cb->set_as_input(false);
					//matrix_aa->set_as_input(true);

					err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
						matrix_results, matrix_aa, matrix_bbtcb, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_aa->set_as_input(false);
				}
				else if (matrix_mul_name == "matrixMul")
				{
					//enqueue_order
					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_cb, matrix_c, matrix_b, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_aa, matrix_a, matrix_a, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
						matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					//matrix_bt->set_as_input(true);

					err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
						matrix_bbt, matrix_b, matrix_bt, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_bt->set_as_input(false);
					//matrix_cb->set_as_input(true);

					err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
						matrix_bbtcb, matrix_bbt, matrix_cb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_cb->set_as_input(false);
					//matrix_aa->set_as_input(true);

					err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
						matrix_results, matrix_aa, matrix_bbtcb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//reset flag iterative exec
					//matrix_aa->set_as_input(false);

				}
			}
			else if (matrix_data.schedule() == SCHEDULE_FINE)
			{
				//mapping
				if (device.cnt_gpus() == 2)
				{
					exec_mt_a = { {1.0f,cpu } };//4
					exec_mm_b = { {0.5f,gpu0 },{0.5f,gpu1 } };//3
					exec_mm_a = { {0.5f,gpu0 },{0.5f,gpu1 } };//2      					
					exec_mm_c = { {0.5f,gpu0 },{0.5f,gpu1 } };//5
					exec_mm_d = { {0.5f,gpu0 },{0.5f,gpu1 } };//6
					exec_mm_e = { {0.5f,gpu0},{0.5f,gpu1 } };//7
				}
				else if (device.cnt_gpus() == 3)
				{
					exec_mt_a = { {1.0f,cpu } };//4
					exec_mm_b = { {0.5f,gpu0 },{0.3f,gpu1 },{0.2f,gpu2 } };//3
					exec_mm_a = { {0.5f,gpu0 },{0.3f,gpu1 },{0.2f,gpu2 } };//2          					
					exec_mm_c = { {0.5f,gpu0 },{0.3f,gpu1 },{0.2f,gpu2 } };//5
					exec_mm_d = { {0.5f,gpu0 },{0.3f,gpu1 },{0.2f,gpu2 } };//6
					exec_mm_e = { {0.5f,gpu0 },{0.3f,gpu1 },{0.2f,gpu2 } };//7
				}
				else
				{
					
					exec_mt_a = { {1.0f,cpu} };//4            
					exec_mm_b = { {1.0f,gpu0 } };//3
					exec_mm_a = { {1.0f,gpu0 } };//2					
					exec_mm_c = { {1.0f,gpu0 } };//5
					exec_mm_d = { {1.0f,gpu0 } };//6
					exec_mm_e = { {1.0f,gpu0 } };//7
				}
				
				//There are two version on mat_multiply kernel with a different interface and performance
				if (matrix_mul_name == "mat_mul_no_opt")
				{
					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_cb, matrix_c, matrix_b, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_aa, matrix_a, matrix_a, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
						matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					//matrix_bt->set_as_input(true);

					err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
						matrix_bbt, matrix_b, matrix_bt, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//matrix_bt->set_as_input(false);
					//matrix_cb->set_as_input(true);

					err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
						matrix_bbtcb, matrix_bbt, matrix_cb, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//matrix_cb->set_as_input(false);
					//manual copy for multi-device executions

					//matrix_bbtcb->set_as_input(true);
					//task_mat_mul_e->set_device_mapping({ gpu1 });
					//task_mat_mul_e->transfer_device_memory(matrix_bbtcb);

					//reset flags
					//matrix_bbtcb->set_as_input(false);
					//matrix_aa->set_as_input(true);

					err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
						matrix_results, matrix_aa, matrix_bbtcb, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					//matrix_aa->set_as_input(false);
				}
				else if (matrix_mul_name == "matrixMul")
				{
					err = device.execute_async(task_mat_mul_b, exec_mm_b, global_sizes, local_sizes,
						matrix_cb, matrix_c, matrix_b, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_a, exec_mm_a, global_sizes, local_sizes,
						matrix_aa, matrix_a, matrix_a, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_transpose_a, exec_mt_a, global_sizes, local_sizes,
						matrix_bt, matrix_b, offset, matrix_data._matrix_width, matrix_data._matrix_height, lmem_mt);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_c, exec_mm_c, global_sizes, local_sizes,
						matrix_bbt, matrix_b, matrix_bt, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_d, exec_mm_d, global_sizes, local_sizes,
						matrix_bbtcb, matrix_bbt, matrix_cb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					err = device.execute_async(task_mat_mul_e, exec_mm_e, global_sizes, local_sizes,
						matrix_results, matrix_aa, matrix_bbtcb, lmem_mm, lmem_mm, matrix_data._matrix_width, matrix_data._matrix_width, matrix_data._matrix_height);
					if (err != CL_SUCCESS)return err;

					
				}
			}
		}
		err = device.flush();
		if (err != CL_SUCCESS)return err;

		err = task_mat_mul_e->wait();
		if (err != CL_SUCCESS)return err;

		task_mat_mul_a->wait_clear_events();
		task_mat_mul_b->wait_clear_events();
		task_mat_mul_c->wait_clear_events();
		task_mat_mul_d->wait_clear_events();
		task_mat_mul_e->wait_clear_events();
		task_mat_transpose_a->wait_clear_events();

		const auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - begin;
		auto app_duration = diff.count() * 1e3f;//msec
		app_duration_avg += app_duration;
	}
	app_duration_avg = app_duration_avg / matrix_data.iterations();
	std::cout << "MEQ duration:\t" << std::setw(3) << app_duration_avg << " ms" << std::endl;
	std::cout << "---------------------------------------" << std::endl;
	return err;
}
