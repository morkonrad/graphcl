
#include "clVirtualDevice.h"

#include <chrono>
#include <sstream>
#include <iostream>
#include "utils.h"

static int test_async_copy_H2D(
	std::unique_ptr<clMemory>& mem_device,
	const size_t begin_byte,
	const size_t end_byte,
	const map_device_info src_device_type_id,
	const map_device_info dst_device_type_id,
	const int iterations = 1)
{
	int err = 0;

	for (int i = 0; i < iterations; i++)
	{
		clAppEvent wait_events_in;
		clAppEvent wait_events_out;

		auto start = std::chrono::system_clock::now();

		err = mem_device->copy_async(
			{ &wait_events_in }, wait_events_out,
			begin_byte, end_byte, src_device_type_id, dst_device_type_id);

		if (err != 0)
			return err;

		err = wait_events_out.wait();
		if (err != 0)
			return err;

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "copy_async duration:\t" << diff.count() * 1e3f << " ms\n";

		std::cout << "---------------------------------------------\n";
		std::cout << "Check source and destination buffers ... ";

		const char* begin_src;
		const char* begin_dst;

		const auto& [src_device_type, src_device_id] = src_device_type_id;
		const bool is_requested_h2d_transfer = src_device_type == CL_DEVICE_TYPE_CPU ? true : false;

		begin_src = mem_device->data_in_buffer_device<char>(src_device_type_id, begin_byte, end_byte);
		begin_dst = mem_device->data_in_buffer_device<char>(dst_device_type_id, begin_byte, end_byte);


		begin_src = mem_device->data_in_buffer_application<char>(dst_device_type_id, begin_byte, end_byte);
		begin_dst = mem_device->data_in_buffer_device<char>(dst_device_type_id, begin_byte, end_byte);

		const int size = end_byte - begin_byte;
		for (int i = begin_byte; i < size; i++)
		{
			if (begin_src[i] != begin_dst[i])
			{
				std::cerr << "BufferA[" << i << "] = " << (int)begin_src[i] << std::endl;
				std::cerr << "BufferB[" << i << "] = " << (int)begin_dst[i] << std::endl;
				std::cerr << "Some value_error at byte [" << i << "] fixme !!!" << std::endl;
				return -1;
			}
		}
		std::cout << "<OK>\n";
	}
	return err;
}

static int unit_test(	
	std::unique_ptr<clMemory>& mem_device,
	const size_t begin_byte,
	const size_t end_byte,
	const map_device_info src_device_type_id,
	const map_device_info dst_device_type_id,
	const int iterations=1)
{	
	int err = 0;

	for (int i = 0; i < iterations; i++)
	{
		clAppEvent wait_events_in;
		clAppEvent wait_events_out;

		auto start = std::chrono::system_clock::now();

		err = mem_device->copy_async(
			{ &wait_events_in }, wait_events_out,
			begin_byte, end_byte, src_device_type_id, dst_device_type_id);

		if (err != 0)
			return err;

		err = wait_events_out.wait();
		if (err != 0)
			return err;

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "copy_async duration:\t" << diff.count() * 1e3f << " ms\n";


		std::cout << "---------------------------------------------\n";
		std::cout << "Check source and destination buffers ... ";

		const char* begin_src;
		const char* begin_dst;

		const auto& [src_device_type, src_device_id] = src_device_type_id;
		const bool is_requested_h2d_transfer = src_device_type == CL_DEVICE_TYPE_CPU ? true : false;

		if (is_requested_h2d_transfer) {
			begin_src = mem_device->data_in_buffer_application<char>(dst_device_type_id, begin_byte, end_byte);
			begin_dst = mem_device->data_in_buffer_device<char>(dst_device_type_id, begin_byte, end_byte);
		}
		else
		{
			begin_src = mem_device->data_in_buffer_device<char>(src_device_type_id, begin_byte, end_byte);
			begin_dst = mem_device->data_in_buffer_application<char>(dst_device_type_id, begin_byte, end_byte);
		}	
		
		const int size = end_byte - begin_byte;
		for (int i = begin_byte; i < size; i++)
		{
			if (begin_src[i] != begin_dst[i])
			{
				std::cerr << "BufferA[" << i << "] = " << (int)begin_src[i] << std::endl;
				std::cerr << "BufferB[" << i << "] = " << (int)begin_dst[i] << std::endl;
				std::cerr << "Some value_error at byte [" << i << "] fixme !!!" << std::endl;
				return -1;
			}
		}
		std::cout << "<OK>\n";
	}
	return err;
}


static int call_test(virtual_device& device, const int items, map_device_info& h2d_d2h_gpu,const size_t iterations=10)
{
	int err = 0; 
	std::vector<int> d1(items);
	std::cout << "----------------------------\n";
	std::cout << "Allocate and transfer:\t" << sizeof(int)*d1.size()*1e-6f << " MB" << std::endl;
	std::cout << "----------------------------\n";
	utils::generate_rand(d1, 1, 10);
	auto md1 = device.alloc(d1, true, { CL_DEVICE_TYPE_CPU,0 });
	
	//--------------------------------
	//h2d
	std::cout << "Check H2D_async transfers ... \n";
	err = unit_test( md1, 10, items,
		{ CL_DEVICE_TYPE_CPU,0 },
		h2d_d2h_gpu,iterations);

	if (err != 0)return err;
	std::cout << "Check H2D_async transfers <OK> \n";
	std::cout << "----------------------------\n";
	
	//--------------------------------
	//d2h
	std::vector<int> d2(items);
	utils::generate_rand(d2, 10, 20);
	auto md2 = device.alloc(d2, true, { CL_DEVICE_TYPE_GPU, 0 });

	std::cout << "Check D2H_async transfers ... \n";
	err = unit_test( md2, 20, items,
		h2d_d2h_gpu,
		{ CL_DEVICE_TYPE_CPU,0 }, iterations);
	if (err != 0)return err;
	std::cout << "Check D2H_async transfers <OK> \n";
	std::cout << "----------------------------\n";
	
	//-----------------------------
	//d2d
	if (device.cnt_gpus() >= 2) 
	{		
		std::vector<int> d3(items);
		utils::generate_rand(d3, 11, 100);
		auto md3 = device.alloc(d3, true, { CL_DEVICE_TYPE_GPU, 0 });

		std::cout << "Check D2D_async transfers ... \n";
		err = unit_test( md3, 0, items,
			{ CL_DEVICE_TYPE_GPU,0 },
			{ CL_DEVICE_TYPE_GPU,1 }, iterations);
		if (err != 0)return err;
		std::cout << "Check D2D_async transfers <OK> \n";
		std::cout << "----------------------------\n";
	}
	std::flush(std::cout);

	return err;
}

int main(int argc, char** argv)
{
	int err = 0;

	std::string status;
	virtual_device device(status);
	if (!status.empty())
	{
		std::cerr << status << std::endl;
		return -1;
	}

	const auto iterations = 10;
	const auto items = 8e6; //8e6=8MB, 32e6=128MB

	map_device_info h2d_d2h_gpu = { CL_DEVICE_TYPE_GPU,0 };

	err = call_test(device, items, h2d_d2h_gpu, iterations);
	if (err != CL_SUCCESS) {
		std::cerr << "Some error: " << err << " fixme!!!" << std::endl;
		return err;
	}
	
	if (device.cnt_gpus() >= 2)
	{
		std::cout << " ------ Now other GPU -------------------\n";
		
		h2d_d2h_gpu = { CL_DEVICE_TYPE_GPU,1 };

		err = call_test(device, items, h2d_d2h_gpu, iterations);
		if (err != CL_SUCCESS) {
			std::cerr << "Some error: " << err << " fixme!!!" << std::endl;
			return err;
		}
	}

	std::cout << "-------------------------\n";
	std::cout << " Passed ...!" << std::endl;
	std::cout << "-------------------------\n";
	return err;
}

