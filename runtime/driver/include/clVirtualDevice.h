#pragma once
#include "clTask.h"


class virtual_device
{
	//all vendor software_platforms
	std::vector<cl::Platform> _cl_platforms;

	//pairs context,device
	std::vector<std::pair<cl::Context, cl::Device>> _cl_devices_context;

	std::map<const cl::Context*, clDevice> _map_devices;

	/**
	 * @brief intialize
	 * @param status returns non-empty string if any error else empty all ok
	 */
	void intialize(std::string& status, const cl_device_type device_type);

	void calculate_statistics(const std::vector<float>& data,
		const std::string msg, std::ostream* ptr_stream_out)const;

	void calculate_workload_distribution_statistics(
		const std::vector<workload_distribution_single_device>& data,
		const std::string msg,std::ostream* ptr_stream_out)const;

	template <typename... KernelArgs>
	int execute_async(
		std::unique_ptr<clTask>& task,
		const offload_info& offload_devices,
		const std::map<const cl::Context*, ndr_division>& ndr_splits,
		const std::array<size_t, 3>& global_size,
		const std::array<size_t, 3>& local_size,
		KernelArgs&... kargs)
	{
		int err = 0;

		//get kernel objects
		const auto task_name = task->name();
		for (auto& [ctx, cldev] : _map_devices)
			cldev.select_kernel_to_execute(task_name);

		//set kernel args 
		//set clMemory IN or OUT flag dependent on the kernel arguments
		//if any global pointer is marked with const than runtime detects
		//that this argument is input-read-only else output
		std::uint8_t id = 0;
		err = task->set_task_args(id, kargs...);
		if (on_coopcl_error(err) != CL_SUCCESS)
			return err;

		//Enqueue H2D(inputs) and D2H(outputs) transfer
		const bool use_cout_transfers = 0;
		if (use_cout_transfers)
			std::cout <<"check transfer_kernel_inputs_async ..."<<std::endl;

		id = 0;		
		auto wait_list_predecessors = task->get_predecessors_wait_list();
		std::vector<std::unique_ptr<clAppEvent>> wait_transfer_inputs;
		err = task->transfer_kernel_inputs_async(wait_list_predecessors, wait_transfer_inputs, offload_devices, id, kargs...);
		if (on_coopcl_error(err) != CL_SUCCESS)
			return err;

		if (use_cout_transfers)
			std::cout << "enqueue kernel ..." << std::endl;

		const bool dbg_use_cout_kernel_ndr = 0;
		if (wait_transfer_inputs.size() == 0)
		{
			//Enqueue kernel execution        
			err = task->async_execute(
				wait_list_predecessors,
				global_size,
				local_size,
				ndr_splits, dbg_use_cout_kernel_ndr);
		}
		else 
		{
			//Enqueue kernel execution        
			err = task->async_execute(
				wait_transfer_inputs,
				global_size,
				local_size,
				ndr_splits, dbg_use_cout_kernel_ndr);
		}

		if (on_coopcl_error(err) != CL_SUCCESS)
			return err;
		
		id = 0;
		err = task->set_task_outputs(offload_devices, id, kargs...);
		return err;
	}

public:

	virtual_device(std::string& status,
				   const cl_device_type devtype = CL_DEVICE_TYPE_ALL) {
		intialize(status, devtype);
	}

	~virtual_device()
	{
		//std::cout<<"Try destroy virtual_device driver ...\n"<<std::endl;

		for (auto& [ptr_ctx,cldevice]: _map_devices)
			cldevice.finish();

		_map_devices.clear();
		_cl_platforms.clear();
		_cl_devices_context.clear();

		//std::cout<<"Destroyed virtual_device OK, EXIT!\n"<<std::endl;
	}

	const std::string get_header_log()const {
		return std::string("Execution_time,Offload,global\n");
	}

	const auto& sub_devices()const { return _map_devices; }

	std::uint8_t cnt_devices()const;
	std::uint8_t cnt_gpus()const;
	std::uint8_t cnt_cpus()const;
	std::uint8_t cnt_accelerators()const; 

	static size_t count_platform_devices(const std::uint8_t device_type, int& err)
	{
		std::vector<cl::Platform> cl_platforms;
		err = cl::Platform::get(&cl_platforms);
		if (err != CL_SUCCESS) return 0;

		for (auto& p : cl_platforms)
		{
			cl::Context ctx;
			std::vector<cl::Device> devices;

			err = p.getDevices(device_type, &devices);
			if (err == CL_DEVICE_NOT_FOUND)continue;
			if (err != CL_SUCCESS) return 0;

			if (!devices.empty())
				return devices.size();

		}
		return 0;
	}

	static void find_device_type_id(
			const float offload_current,
			const float offload_next,
			map_device_info& src_dev_type_id,
			map_device_info& dst_dev_type_id )
	{
		if (offload_current == 0.0f && offload_next>0.0f)
		{
			src_dev_type_id = { CL_DEVICE_TYPE_CPU,0 };
			dst_dev_type_id = { CL_DEVICE_TYPE_GPU,0 };
		}
		else if (offload_current > 0.0f && offload_next<=1.0f)
		{
			src_dev_type_id = { CL_DEVICE_TYPE_GPU,0 };
			dst_dev_type_id = { CL_DEVICE_TYPE_CPU,0 };
		}
	}

	template <typename T>
	std::unique_ptr<clMemory>
	alloc(const size_t items, void* src = nullptr, const bool read_only = false, const map_device_info copy_to_device_type = { CL_DEVICE_TYPE_ALL,0 })
	{
		int err = 0;
		auto memory = std::make_unique<clMemory>();
	  
		for (auto& [context, device] : _map_devices)
		{
			auto pair_dev_queue = std::make_tuple<const cl_device_id, const cl::CommandQueue*, const cl::Context*>
				(device.device_ptr(), device.cq_io_ptr(), device.ctx());

			err = memory->allocate(pair_dev_queue, items, sizeof(T) * items, read_only, CL_DEVICE_TYPE_ALL,device.device_id());
			if (err != CL_SUCCESS)return nullptr;

		}
		//copy to allocated memory for specified device via copy_to_device_type
        err = memory->copy_application_memory(src,copy_to_device_type.first, copy_to_device_type.second);
		if (err != CL_SUCCESS)return nullptr;

		return memory;
	}

	template <typename T>
	std::unique_ptr<clMemory>
		alloc(const std::vector<T>& input, const bool read_only = false, const map_device_info copy_to_device_type = { CL_DEVICE_TYPE_ALL,0 })
	{
		int err = 0;

		auto memory = std::make_unique<clMemory>();

		for (auto& [context, device] : _map_devices)
		{
			auto pair_dev_queue = std::make_tuple<const cl_device_id, const cl::CommandQueue*, const cl::Context*>
				(device.device_ptr(), device.cq_io_ptr(), device.ctx());

			//allocate memory for all devices
			err = memory->allocate(pair_dev_queue, input.size(), sizeof(T) * input.size(), read_only, CL_DEVICE_TYPE_ALL, device.device_id());
			if (err != CL_SUCCESS)return nullptr;
		}

		//copy to allocated memory for specified device via copy_to_device_type
        err = memory->copy_application_memory(input.data(), copy_to_device_type.first, copy_to_device_type.second);
		if (err != CL_SUCCESS)return nullptr;

		return memory;
	}

	
	
	std::unique_ptr<clTask>
	create_task(const std::string& task_body, const std::string& task_name,
				const std::string jit_flags = "");

	template <typename... KernelArgs>
	int execute_async(
			std::unique_ptr<clTask>& task,
			const offload_info& offload_devices,
			const std::array<size_t, 3>& global_size,
			const std::array<size_t, 3>& local_size,
			KernelArgs&... kargs)
	{
		//partition NDRange in sub_ranges for all offload_devices
		const std::array<size_t, 3>& offsets = { 0,0,0 };
		const auto ndr_divisions = TaskArgs::calculate_ndr_division(
					global_size, local_size, _map_devices, offload_devices, offsets);

		return execute_async(task, offload_devices, ndr_divisions, global_size, local_size, kargs...);

	}



	int flush()const
	{
		int err=0;
		// flush devices queues
		for (auto& [ctx, cldev] : _map_devices)
		{
			err = cldev.flush();
			if (on_coopcl_error(err) != CL_SUCCESS)
				return err;
		}
		return err;
	}

	int finish()const
	{
		int err=0;
		// finish devices queues
		for (auto& [ctx, cldev] : _map_devices)
		{
			err = cldev.finish();
			if (on_coopcl_error(err) != CL_SUCCESS)
				return err;
		}
		return err;
	}

//	template <typename... KernelArgs>
//	std::string benchmark_execute_task(
//		std::unique_ptr<clTask>& task,
//		const offload_info& ndr_splits,
//		const std::array<size_t, 3>& global_size,
//		const std::array<size_t, 3>& local_size,
//		std::ostream* stream_out,
//		const size_t iterations,
//		const std::string msg,
//		KernelArgs&... kargs)
//	{
//		int err = 0;
//		std::string out{ "" };
//		float acc = 0;

//		std::vector<float> exec_times;
//		std::vector<workload_distribution_single_device> workload_distributions;


//		for (int i = 0; i < iterations; i++)
//		{
//			std::cout << "Iteration: " << i + 1 << " ...\n";
//			const auto begin = std::chrono::system_clock::now();

//			err = execute_async(task, ndr_splits, global_size, local_size, kargs...);
//            if (on_coopcl_error(err) != 0) return "FIXME!";
//			err = task->wait();
//            if (on_coopcl_error(err) != 0) return "FIXME!";

//			const auto end = std::chrono::system_clock::now();
//			const std::chrono::duration<double> diff = end - begin;
//			if (i > 0) exec_times.push_back(diff.count() * 1e3f);//time in ms
//		}

//        //calculate_statistics_execution_times<float>(exec_times, stream_out, msg);
//        //calculate_workload_distribution_statistics(workload_distributions, msg, stream_out);

//		return out;
//	}
};


