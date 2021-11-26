#pragma once
#include "clEvent.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

class clMemory
{

	// TWO_BUFFERS:
	// 1) APPLICATION_BUFFER in HOST_ACCESS_PINNED_MEMORY_HEAP
	// 
	// This buffer is pinned at OS. via GPU driver-runtime. 
	// That enables fast transfers between APP(Host)<-->GPUs, max. BUS-interconnect speed. 
	// The CPU-HOST via map has also zero-latency access 
	// 
	// 2)DEVICE in GPU on-chip MEMORY_HEAP
	// Internal low-latency, on-chip GPU memories. Best for kernel processing
	std::map<const cl_device_id, std::pair<cl::Buffer, cl::Buffer>> _map_device_app_buffers;
	
	
	std::map<const cl_device_id, const cl::CommandQueue*> _map_device_queue_io;
	std::map<const cl_device_id, const cl::Context*> _map_device_context;
	std::map<const cl_device_id, map_device_info> _map_device_type_id;
	std::map<map_device_info, const cl::Context*> _map_device_type_conetxt;

	map_device_info _device_with_coherent_memory;
	bool _is_input = false;
	bool _is_contiguous_access = false;

	size_t _single_item_size_bytes{ 0 };
	size_t _size_bytes{ 0 };
	
private:	

	int async_transfer_d2d(
		cl::Buffer* src_buff_device,
		cl::Buffer* dst_buff_device,
		clAppEvent& wait_event_out,
		const std::vector<const clAppEvent*>& wait_events_in,
		const size_t begin_byte,
		const size_t size,
		const map_device_info& destination_device);

	int enqueue_async_transfer_device_appliction(
		const cl::CommandQueue& cq_device,
		cl::Buffer& application_buffer,
		cl::Buffer& device_memory,
		const bool use_transfer_h2d,
		cl::Event& wait_event, const std::vector<cl::Event>& wait_list, const size_t size, const size_t offset);

	int async_transfer_from_to(
		const bool copy_h2d,
		cl::Buffer* src_buff_device, cl::Buffer* dst_buff_device,
		clAppEvent& wait_event_out,
		const std::vector<const clAppEvent*>& wait_events_in,
		const size_t begin_byte,
		const size_t size,
		const map_device_info& source_device,
		const map_device_info& destination_device);
	
	const void* map_read_data(const bool map_read_device_memory,
		const map_device_info& device_type_id, const size_t begin_byte = 0, const size_t end_byte = 0)const;

	int check_scatter_to_gpus();

	int check_map_access_latency_from_host_to_device_mem(const bool map_read, const std::uint8_t gpu_id = 0)const;

	int check_map_access_latency_from_host_to_pinned_mem(const bool map_read, const std::uint8_t gpu_id = 0)const;

	int check_copy_from_to_pinned_mem(const bool write_pinned, const std::uint8_t gpu_id)const;

	int check_copy_from_gpu_to_gpu(const std::uint8_t gpu_id_src, const std::uint8_t gpu_id_dst)const;

	int copy_async_h2h(
		const bool h2d,
		const std::vector<const clAppEvent*>& wait_events_in,
		clAppEvent& wait_event_out,
		const size_t begin_byte, const size_t end_byte,
		const map_device_info& destination_device);

public:
	clMemory() = default;

	int allocate(
		const std::tuple<const cl_device_id, const cl::CommandQueue*, const cl::Context*>& device_command_queue_context,
		const size_t cnt_items,
		const size_t size_items,
		const bool read_only = false,
		const size_t device_type = CL_DEVICE_TYPE_CPU,
		const std::uint8_t device_id = 0);

	const cl::Buffer* buffer_device(cl_device_id device)const{ return &_map_device_app_buffers.at(device).first;}
	cl::Buffer* buffer_device(cl_device_id device){ return &_map_device_app_buffers.at(device).first; }

	const cl::Buffer* buffer_application(cl_device_id device)const{ return &_map_device_app_buffers.at(device).second; }
	cl::Buffer* buffer_application(cl_device_id device){ return &_map_device_app_buffers.at(device).second; }

	///Read memory in the internal clBuffer
	template<typename T>
	const T* data_in_buffer_application(const map_device_info& device_type_id, const size_t begin_byte = 0, const size_t end_byte = 0)const
	{
		auto map_ptr = map_read_data(false, device_type_id, begin_byte, end_byte);
		return static_cast<const T*>(map_ptr);
	}

	///Read memory in the internal clBuffer
	template<typename T>
	const T* data_in_buffer_device(const map_device_info& device_type_id, const size_t begin_byte=0, const size_t end_byte=0)const
	{
		auto map_ptr = map_read_data(true, device_type_id, begin_byte, end_byte);
		return static_cast<const T*>(map_ptr);
	}

	bool is_input()const { return _is_input; }
	void set_as_input(const bool is_input) { _is_input = is_input; }

	bool is_contiguous_access()const { return _is_contiguous_access; }
	void set_contiguous_access(const bool is_cont_pattern) { _is_contiguous_access = is_cont_pattern; }

	size_t item_size()const { return _single_item_size_bytes; }
	size_t size()const { return _size_bytes; }
	size_t items()const { return _size_bytes / _single_item_size_bytes; }
	

	const map_device_info get_device_with_coherent_memory()const { return _device_with_coherent_memory; }
	void set_device_with_coherent_memory(const map_device_info device) { _device_with_coherent_memory = device; }

	void benchmark_memory();

	int copy_application_memory(const void* src_application, const size_t copy_to_device_type = CL_DEVICE_TYPE_CPU,
		const std::uint8_t copy_to_device_id = 0);

	int copy_async(const std::vector<const clAppEvent*>& wait_events_in,
		clAppEvent& wait_event_out,
		const size_t begin_byte, const size_t end_byte,
		const map_device_info& source_device,
		const map_device_info& destination_device);

	int merge_async(const std::vector<const clAppEvent*>& wait_events_in,
		clAppEvent& wait_event_out,
		const size_t begin_byte, const size_t end_byte,
		const map_device_info& source_device,
		const map_device_info& destination_device);
	
	
};




