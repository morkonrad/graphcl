#pragma once
#include "clDevice.h"
#include "clCommon.h"
#include "clEvent.h"

#include <map>
#include <stdexcept>
#include <iostream>

class clTask
{
	struct arg_info 
	{
		size_t _ADDRESS_QUALIFIER;		
		cl_kernel_arg_type_qualifier _TYPE_QUALIFIER;		
		std::string _TYPE_NAME;
	};

	std::map<const cl::Context*, clDevice*> _map_devices;
	
	std::string _name{ "" };
	
	std::unique_ptr<clAppEvent> _event_exec_kernel;
	std::vector<const clAppEvent*> _events_to_wait_for_memory_copy;
	std::vector<const clTask*> _dependent_tasks;

	std::uint8_t _queue_id = 0;

	std::vector<arg_info> _kernel_arguments;

	int identify_kernel_input_output_memory(const cl::Kernel& kernel);

	//---------------------------------------
	// generic arg
	//---------------------------------------
	template <typename T>
	int set_task_arg(std::uint8_t& id, const T& arg)
	{
		int err = 0;
		for (auto& [ctx, ptr_device] : _map_devices)
		{
			err = ptr_device->set_arg(id, arg);
			if (err != CL_SUCCESS)
				return err;
		}
		return err;
	}

	int set_task_arg(std::uint8_t& id, cl::LocalSpaceArg& arg);
	int set_task_arg(std::uint8_t& id, std::unique_ptr<clMemory>& arg);	
	int set_task_args(std::uint8_t& id) { return 0; }

	//---------------------------------------
	// generic arg
	//---------------------------------------
	template <typename T>
	int set_task_output(const offload_info& offload_devices, std::uint8_t& id, const T& arg) {return 0;}
	int set_task_output(const offload_info& offload_devices, std::uint8_t& id, std::unique_ptr<clMemory>& arg);
	int set_task_outputs(const offload_info& offload_devices, std::uint8_t& id) { return 0; }

	//-----------------------------
	
	template <typename T>
	int transfer_memory_async(const std::vector<const clAppEvent*>& wait_events_in,
		clAppEvent& wait_event_out, const offload_info& offload_devices, const T& arg) {return 0;}

	int transfer_memory_async(const std::vector<const clAppEvent*>& wait_events_in,
		clAppEvent& wait_event_out, const offload_info& offload_devices, std::unique_ptr<clMemory>& arg);
	
	int transfer_kernel_inputs_async(const std::vector<const clAppEvent*>& wait_events_in,
		std::vector<std::unique_ptr<clAppEvent>>& wait_events_out, const offload_info& offload_devices, std::uint8_t& id) {
		return 0;
	}
	
public:
	clTask() = default;

	~clTask()
	{        
		if(_event_exec_kernel)
			_event_exec_kernel->wait();
	}

	int build_task(const cl::Context* ptr_ctx, clDevice* ptr_clDevice, const std::string& task_name,
			const std::string& task_body, const std::string jit_flags);
	

	template <typename T, typename... Args>
	int set_task_args(std::uint8_t& id, T& first, Args &... rest)
	{
		int err = 0;
		err = set_task_arg(id, first);
		if (err != CL_SUCCESS)
			return err;
		id++;
		return set_task_args(id, rest...);
	}


	template <typename T, typename... Args>
	int set_task_outputs(const offload_info& offload_devices, std::uint8_t& id, T& first, Args &... rest)
	{
		int err = 0;
		err = set_task_output(offload_devices, id, first);
		if (err != CL_SUCCESS)
			return err;
		id++;
		return set_task_outputs(offload_devices,id, rest...);
	}

	int set_arg(std::uint8_t id, std::unique_ptr<clMemory>& clmemory)
	{
		auto err = set_task_arg(id, clmemory);
		return err;
	}

	std::string name()const { return _name; }

	int wait()const;

	void set_queue_id(const std::uint8_t queue_id) { _queue_id = queue_id; }

	void add_dependence(const std::vector<clAppEvent>& events_in);
	
	void add_dependence(const clAppEvent& event_in);
	
	void add_dependence(const clTask* other_task);

	const std::vector <const clAppEvent*> get_wait_list()const
	{
		std::vector<const clAppEvent*> events_to_wait;
		
		//First copy events added/set by the add_dependence API, for example copy_memory command
		if (!_events_to_wait_for_memory_copy.empty())
		{
			for(const auto& ev:_events_to_wait_for_memory_copy)
				events_to_wait.push_back(ev);
		}
		
		//Scan predecessors and get their events
		for (auto& dep_task : _dependent_tasks) 
		{
			auto event = dep_task->get_event_wait_kernel();
			events_to_wait.push_back(event);
		}
		return events_to_wait;
	}

	const clAppEvent* get_event_wait_kernel()const 
	{
		return _event_exec_kernel.get();		
	}

	float duration(const float scale_factor)const;

	void wait_clear_events();

	void clear_dependences()
	{
		_dependent_tasks.clear();
	}

	int async_execute(
		const std::vector<const clAppEvent*>& wait_events_in,
		const std::array<size_t, 3>& global_sizes, const std::array<size_t, 3>& group_sizes,
		const std::map<const cl::Context*, ndr_division>& ndr_divisions, const bool use_cout);

	int async_execute(
		const std::vector<std::unique_ptr<clAppEvent>>& wait_events_in,
		const std::array<size_t, 3>& global_sizes,
		const std::array<size_t, 3>& group_sizes,
		const std::map<const cl::Context*, ndr_division>& ndr_divisions,
		const bool use_cout);

	template <typename T, typename... Args>
	int transfer_kernel_inputs_async(
		const std::vector<const clAppEvent*>& wait_events_in,
		std::vector<std::unique_ptr<clAppEvent>>& wait_events_out,
		const offload_info& offload_devices, std::uint8_t& id, T& first, Args &... rest)
	{			
		auto wait_event_out = std::make_unique<clAppEvent>();

		int err = 0;
		err = transfer_memory_async(wait_events_in,*wait_event_out,offload_devices, first);
		if (err != CL_SUCCESS)
			return err;
		id++;
		
		if(wait_event_out->is_created())
			wait_events_out.push_back(std::move(wait_event_out));

		return transfer_kernel_inputs_async(wait_events_in, wait_events_out, offload_devices, id, rest...);
	}
		
};


class TaskArgs
{

private:
	clTask* _ptr_Task{ nullptr };

	std::array<size_t, 3> _ndrange{ 0,0,0 };
	std::array<size_t, 3> _group_size{ 0,0,0 };
	std::array<size_t, 3> _offset{ 0,0,0 };
	std::array<size_t, 3> _chunk{ 0,0,0 };

	offload_info _ndr_division_params{};

	std::string _name = "";

	void create(clTask& task,
		const std::array<size_t, 3>& ndrange,
		const std::array<size_t, 3>& group_size,
		const offload_info& info)
	{
		_ptr_Task = &task;

		_ndrange = ndrange;
		_group_size = group_size;

		_name = task.name();

		_ndr_division_params = info;
	}

public:
	TaskArgs() = default;

	TaskArgs(
		clTask& task,
		const std::array<size_t, 3>& ndrange,
		const std::array<size_t, 3>& group_size,
		const offload_info& info,
		const std::array<size_t, 3> offset_sizes = { 0,0,0 })
	{
		create(task, ndrange, group_size, info);

		_offset = offset_sizes;
	}

	void dump(std::ostream& ost)
	{
		std::string schedule_txt, distribution_txt;
		ost << "Task:\t" << _ptr_Task->name() << "\n";
		ost << "#########" << "\n";
	}

	static std::map<const cl::Context*, ndr_division>
		calculate_ndr_division(
			const std::array<size_t, 3>& global_size,
			const std::array<size_t, 3>& group_sizes,
			const std::map< const cl::Context*, clDevice>& devices,
			const offload_info& ndr_division_info,
			const std::array<size_t, 3>& begin_offset_sizes = { 0,0,0 },
			const int ndr_dim_id_to_split = -1)
	{
		std::map < const cl::Context*, ndr_division> global_chunk_offset;
		size_t sum_split_sizes{ 0 };
		size_t offset{ 0 };
		float sum_all_offloads{ 0 };

		size_t ndr_dim_id = 0;

		if (ndr_dim_id_to_split == -1)//heuristic, ndrange-runtime-based
		{
			//check if ndr_dim_y(1) is equal 1 than split dimension_x(0)
			ndr_dim_id = global_size[1] == 1 ? 0 : 1;
		}
		else //manual via in_arg_value
			ndr_dim_id = ndr_dim_id_to_split;

		if (global_size[0] == 0 || global_size[1] == 0 || global_size[2] == 0)
			throw std::runtime_error("Expected Global sizes >= 1, FIXME!!!");

		if (global_size[0] <= 1 && ndr_dim_id == 0)
			throw std::runtime_error("Expected Global_size_X > 1, FIXME!!!");

		if (global_size[1] <= 1 && ndr_dim_id == 1)
			throw std::runtime_error("Expected Global_size_Y > 1, FIXME!!!");

		for (auto& [ctx, cldevice] : devices)
		{
			// for each device in the contexts calculate global_size_chunk and global_offsets
			for (auto& [offload, dev_type_dev_id] : ndr_division_info)
			{
				const auto& [dev_type, dev_id] = dev_type_dev_id;

				//find device id and type to calculate offload ndr_chunk and offsets
				if (dev_id == cldevice.device_id() && dev_type == cldevice.device_type())
				{
					if (offload == 0.0f)
						continue;

					size_t split{ 0 };
					auto chunk = global_size[ndr_dim_id] * offload;

					//round to multiple of group_size
					split = static_cast<size_t>(std::floor(chunk / (float)group_sizes[ndr_dim_id])) * group_sizes[ndr_dim_id];

					if (split < group_sizes[ndr_dim_id])
						split = group_sizes[ndr_dim_id];

					auto& global_rect = global_chunk_offset[ctx].global_chunk_sizes;
					auto& offsets = global_chunk_offset[ctx].global_offset_sizes;

					if (ndr_dim_id == 0)
					{
						global_rect = { split, global_size[1],global_size[2] };
						offsets = { offset,0,0 };
					}
					else
					{
						global_rect = { global_size[0],split,global_size[2] };
						offsets = { 0,offset,0 };
					}

					offset += split;
					sum_all_offloads += offload;
					sum_split_sizes += split;

					//because fractional ndr_division is applied via offload variable need to check if round last chunk size ?
					if (sum_all_offloads > 0.999f && sum_split_sizes != global_size[ndr_dim_id])
					{
						const auto rest_items = global_size[ndr_dim_id] - sum_split_sizes;

						if (ndr_dim_id == 0)
							global_rect = { split + rest_items,global_size[1],global_size[2] };
						else
							global_rect = { global_size[0],split + rest_items,global_size[2] };
					}
				}
			}
		}

		//apply if any offset !
		for (auto& [ctx, ndr_div] : global_chunk_offset)
			ndr_div.global_offset_sizes.at(1) += begin_offset_sizes.at(1);

		//global_chunk_offset.begin()->second.global_offset_sizes.at(1) += begin_offset_sizes.at(1);
		return global_chunk_offset;
	}

	//use context to find device type and id
	static map_device_info find_device_type_and_id(const cl::Context* ctx_query,
		const std::map<const cl::Context*, clDevice>& devices)
	{
		for (auto& [ctx, cldevice] : devices)
		{
			if (ctx == ctx_query)
			{
				return { cldevice.device_type(),cldevice.device_id() };
			}
		}
		return{};
	}

	std::string Name()const { return _name; }
	void Name(const std::string name) { _name = name; }

	offload_info Schedule_static() const
	{
		return  _ndr_division_params;
	}

	float offload_value() const
	{
		if (!_ndr_division_params.empty()) 
		{
			const auto& [off, dev_type_dev_id] = _ndr_division_params.at(0);
			return  off;
		}
		return 0.0f;
	}

	std::array<size_t, 3>
		Ndrange() const { return _ndrange; }

	std::array<size_t, 3>
		Group_size() const { return _group_size; }

	std::array<size_t, 3>
		Offset() const { return _offset; }

	clTask* ptr_Task() { return _ptr_Task; }

	
};
