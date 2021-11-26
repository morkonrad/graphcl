#include "clDevice.h"

#include <iostream>
#include <chrono>

static int create_program(
	cl::Program& program,
	const cl::Context& context,
	const cl::Device& device,
	const std::string task_body,
	const std::string jit_flags)
{
	int err = 0;

    //const auto start = std::chrono::system_clock::now();
	program = cl::Program(context, task_body, false, &err);
	if (on_coopcl_error(err) != 0) return err;
	std::vector<cl::Device> devices = { device };
	
	if(jit_flags.empty())
		err = program.build(devices,nullptr,nullptr,nullptr);
	else 
		err = program.build(devices, jit_flags.c_str(), nullptr, nullptr);

	if (err != 0)
	{
		std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &err) << std::endl;
		if (on_coopcl_error(err) != 0) return err;
	}
    //const auto end = std::chrono::system_clock::now();
    //const std::chrono::duration<double> diff = end - start;
	//auto dev_name = device.getInfo<CL_DEVICE_NAME>(&err);
	//std::cout << "JIT build time:\t" << dev_name<<"\t"<<diff.count()*1e3f <<" ms"<<std::endl;
	
	return err;
}

int clDevice::create(
	const cl::Context& ctx,
	const cl::Device& device,
	const std::uint8_t did)
{
	auto err = create_command_queues(ctx, device, did);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;	

	_max_wgs = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;	

	_name = device.getInfo<CL_DEVICE_NAME>(&err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	_ptr_ctx = &ctx;
	_ptr_device = &device;

	return err;
}

int clDevice::create_async(
	const cl::Context& ctx,
	const cl::Device& device,
	const std::uint8_t did)
{
	return create(ctx,device,did);
}

clDevice::clDevice(int& err,	
	const cl::Context& ctx,
	const cl::Device& device,
	const std::uint8_t did)
{
	err = create(ctx, device, did);
	if (err != CL_SUCCESS)std::exit(err);
}

clDevice::clDevice(
	int& err,	
	const cl::Context& ctx,
	const cl::Device& device,
	const std::uint8_t did,
	const std::string& task_body,
	const std::string& task_exec_name,
	const std::string jit_flags)
{
	err = create(ctx, device, did);
	if (err != CL_SUCCESS)std::exit(err);
	
	err = build_kernel(task_exec_name, task_body, jit_flags);
	if (err != CL_SUCCESS)std::exit(err);	
}

int clDevice::build_kernel(
	const std::string& task_name,
	const std::string& task_body,
	const std::string jit_flags)
{
	int err = 0;
	if (_program_kernel_execution() == nullptr) 
	{
		err = create_program(_program_kernel_execution, *_ptr_ctx, *_ptr_device, task_body, jit_flags);
		if (on_coopcl_error(err) != CL_SUCCESS)return err;
	}

	//// COMMENT out because of common_context and events !
	
//	else
//	{
//		auto knames = _program_kernel_execution.getInfo<CL_PROGRAM_KERNEL_NAMES>(&err);
//		if (on_coopcl_error(err) != CL_SUCCESS)return err;

//		if (knames.find(task_name) == std::string::npos)
//		{
//			err = create_program(_program_kernel_execution, *_ptr_ctx, *_ptr_device, task_body, jit_flags);
//			if (on_coopcl_error(err) != CL_SUCCESS)return err;
//		}
//	}

	_task_exec = cl::Kernel(_program_kernel_execution, task_name.c_str(), &err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	//copy and store in map
	_map_kernels[task_name] = _task_exec;

	//CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
	_wave_size = _task_exec.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(*_ptr_device, &err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	return err;
}

int clDevice::create_command_queues(
	const cl::Context& ctx,
	const cl::Device& device,
	const std::uint8_t did)
{
	int err = 0;	

	_device_id = did;
	_device_type = device.getInfo<CL_DEVICE_TYPE>(&err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	_compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	//clamp
	const auto max_command_queues = 1;// _compute_units > 4 ? 4 : _compute_units;
	
	cl_command_queue_properties props[3] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	_cq_io = cl::CommandQueue(clCreateCommandQueueWithProperties( ctx(), device() , &props[0],  &err));

	for (size_t i = 0; i < max_command_queues; i++)
	{
		_cqs_kernels.push_back(cl::CommandQueue(clCreateCommandQueueWithProperties( ctx(), device() , &props[0],  &err)));
		if (on_coopcl_error(err) != CL_SUCCESS)return err;
	}

	return err;
}

int clDevice::set_arg(const size_t id, const cl::Buffer* arg)
{
	return _task_exec.setArg<cl::Buffer>(id, *arg);
}

int clDevice::flush()const
{
	int err{ 0 };
	
	for (auto& cq : _cqs_kernels) {
		err = cq.flush();
		if (on_coopcl_error(err) != CL_SUCCESS)return err;
	}
	
	err = _cq_io.flush();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	return err;
}

int clDevice::finish() const
{
	int err{ 0 };

	err = flush();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	_cq_io.finish();
	if (on_coopcl_error(err) != CL_SUCCESS)return err;

	for (auto& cq : _cqs_kernels) {
		err = cq.finish();
		if (on_coopcl_error(err) != CL_SUCCESS)return err;
	}

	return err;
}

int clDevice::enqueue_async_ndrange_block(	
	cl::Event& wait_event_kernel,
	const std::vector<cl::Event>& wait_list,
	const std::array<size_t, 3>& global_sizes,
	const std::array<size_t, 3>& local_sizes,
	const std::array<size_t, 3>& offsets,
	const std::array<size_t, 3>& global_chunk_sizes,
	const std::uint8_t queue_id,
	const bool use_cout) const
{
	int err = 0;

	/*const auto sum_wi_block = global_chunk[0] + global_chunk[1] + global_chunk[2];
	const auto sum_wi_all = global_sizes[0] + global_sizes[1] + global_sizes[2];
	if (sum_wi_block > sum_wi_all)
		return -999;

	auto last_id_current = 0;
	if (split_nD)
		last_id_current = offsets[1] + global_chunk[1];
	else
		last_id_current = offsets[0] + global_chunk[0];

	const auto last_id = split_nD ? global_sizes[1] : global_sizes[0];
	if(last_id_current > last_id)
		return -999;*/

	err = enqueue_async_exec(wait_event_kernel, wait_list, global_chunk_sizes, local_sizes, offsets, queue_id);
	if (err != CL_SUCCESS)return err;

	if (use_cout) 
	{
		std::cout << "---------- enqueue async block ----------\n";
		std::cout << "Device:\t[" << _device_type << "," << (int)_device_id << "] " << _name << "\n";
		std::cout << "Global NDRange block:\t{"
			<< global_chunk_sizes.at(0) << ","
			<< global_chunk_sizes.at(1) << ","
			<< global_chunk_sizes.at(2) << "}\n";
	}

	return err;

}

int clDevice::enqueue_async_exec(
	cl::Event& wait_event,
	const std::vector<cl::Event>& wait_list,
	const std::array<size_t, 3>& global_sizes,
	const std::array<size_t, 3>& local_sizes,
	const std::array<size_t, 3>& offsets,
	const std::uint8_t queue_id )const
{
	const cl::NDRange offset_ndr{ offsets[0],offsets[1],offsets[2] };
	const cl::NDRange global_sizes_ndr{ global_sizes[0],global_sizes[1],global_sizes[2] };
	const cl::NDRange local_sizes_ndr{ local_sizes[0],local_sizes[1],local_sizes[2] };
	
	int err = 0;
	std::uint8_t qid = 0;

	qid = queue_id;
	
	//clamp
	if (queue_id >= _cqs_kernels.size())
		qid = 0;
	
	err = _cqs_kernels[qid].enqueueNDRangeKernel(_task_exec, offset_ndr, global_sizes_ndr,
			local_sizes_ndr, &wait_list, &wait_event);
	if (on_coopcl_error(err) != 0)return err;
	
	//err = _cqs_kernels[qid].flush();
	//if (on_coopcl_error(err) != 0)return err;

	//std::cout << "qid:\t" << (int)qid << std::endl;

	return err;
}
