
#include "clVirtualDevice.h"

#include <chrono>
#include <sstream>
#include <iostream>
#include "utils.h"


static int test_async_execute_and_copy_gpu_cpu(
    virtual_device& device,
    std::unique_ptr<clMemory>& mem_device,
    std::unique_ptr<clTask>& task,
    const size_t items, const float offload, const std::uint8_t gpu_id = 0)
{

    int err = 0;
	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
	const auto gpu = std::make_pair(CL_DEVICE_TYPE_GPU, gpu_id);
	

    auto start = std::chrono::system_clock::now();

    const size_t end_byte = mem_device->size();
    const std::array<size_t, 3> gs = { items,1,1 };
    const std::array<size_t, 3> ls = { 64,1,1 };

    //execute task with some offload part on CPU and part in parallel on GPU
    const offload_info offload_ndr = { {offload,cpu}, {1.0f - offload,gpu} };
    err = device.execute_async(task, offload_ndr, gs, ls, mem_device);
    on_coopcl_error(err);

    /// copy GPU--->APP
    auto wait_task = task->get_event_wait_kernel();

    clAppEvent wait_events_out_d2h;
    auto data_split = static_cast<size_t>(std::floor(static_cast<float>(items) * offload / static_cast<float>(ls[0]))) * ls[0];
    const size_t offset = data_split * mem_device->item_size();

    err = mem_device->copy_async({ wait_task }, wait_events_out_d2h, offset, end_byte,gpu,cpu);

    if (err != 0) return err;

    err = device.flush();
    if (err != 0)return err;


    err = wait_events_out_d2h.wait();
    if (err != 0)return err;

    const auto end = std::chrono::system_clock::now();
    const std::chrono::duration<double> diff = end - start;
    std::cout << "Test duration:\t" << diff.count() * 1e3f << " ms" << std::endl;


    std::cout << "Compare results .... ";
    auto begin_cpu = mem_device->data_in_buffer_device<int>(cpu);
    /*std::vector<int> tmp_cpu(items, 0);
    std::memcpy(tmp_cpu.data(), begin_cpu, sizeof(int) * items);*/

    auto begin_gpu0 = mem_device->data_in_buffer_application<int>(gpu);
    /*std::vector<int> tmp_gpu0(items, 0);
    std::memcpy(tmp_gpu0.data(), begin_gpu0, sizeof(int) * items);*/

    for (size_t i = 0; i < items; i++)
    {
        const int* begin = i < data_split ? begin_cpu : begin_gpu0;
        if (begin[i] != 2)
        {
            std::vector<int> tmp_cpu(items, 0);
            std::memcpy(tmp_cpu.data(), begin, sizeof(int) * items);
            std::cerr << "Some value_error at position [" << i << "] fixme !!!" << std::endl;
            return -1;
        }
    }
    std::cout << "<OK>\n";
    std::cout << "--------------------------------\n";
    return err;
}

static int test_async_execute_and_copy_2gpus(
    virtual_device& device,
    std::unique_ptr<clMemory>& mem_device,
    std::unique_ptr<clTask>& task,
    const size_t items,
    const float offload,
    const size_t iterations=1)
{
	const auto cpu = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto gpu0 = std::make_pair(CL_DEVICE_TYPE_GPU, 0);
    const auto gpu1 = std::make_pair(CL_DEVICE_TYPE_GPU, 1);

    int err = 0;
    for (size_t i = 0; i < iterations; i++)
    {
        const auto start = std::chrono::system_clock::now();

        const size_t end_byte = mem_device->size();
        const std::array<size_t, 3> gs = { items,1,1 };
        const std::array<size_t, 3> ls = { 64,1,1 };

        //execute task with some offload part on GPU_a and part in parallel on GPU_b
        const offload_info offload_ndr = { {offload,gpu0}, {1.0f - offload,gpu1} };
        err = device.execute_async(task, offload_ndr, gs, ls, mem_device);
        on_coopcl_error(err);

        //get wait_event
        auto wait_task = task->get_event_wait_kernel();

        //calculate offset
        auto data_split = static_cast<size_t>(std::floor(static_cast<float>(items) * offload / static_cast<float>(ls[0]))) * ls[0];
        const size_t offset = data_split * mem_device->item_size();

        /// copy d2h GPU--->APP
        clAppEvent wait_events_out_d2h0;
        clAppEvent wait_events_out_d2h1;
        //read in parallel from GPU_a and GPU_b
        err = mem_device->copy_async({ wait_task }, wait_events_out_d2h0, 0, offset,gpu0,cpu);
        if (err != 0)return err;

        err = mem_device->copy_async({ wait_task }, wait_events_out_d2h1, offset, end_byte, gpu1,cpu);
        if (err != 0) return err;

        err = device.flush();
        if (err != 0)return err;

        err = wait_events_out_d2h0.wait();
        if (err != 0)return err;

        err = wait_events_out_d2h1.wait();
        if (err != 0)return err;

        const auto end = std::chrono::system_clock::now();
        const std::chrono::duration<double> diff = end - start;
        std::cout << "Test duration:\t" << diff.count() * 1e3f << " ms" << std::endl;

        std::cout << "Compare results .... ";
        /*auto begin_cpu = mem_device->data_in_buffer_device<int>({ CL_DEVICE_TYPE_CPU,0 });
                    std::vector<int> tmp_cpu(items, 0);
                    std::memcpy(tmp_cpu.data(), begin_cpu, sizeof(int) * items);
        */

        //---application_buffers
        auto begin_gpu0 = mem_device->data_in_buffer_application<int>(gpu0);
        /*std::vector<int> tmp_gpu0_app(items, 0);
        std::memcpy(tmp_gpu0_app.data(), begin_gpu0, sizeof(int) * items);*/

        auto begin_gpu1 = mem_device->data_in_buffer_application<int>(gpu1);
        /*std::vector<int> tmp_gpu1_app(items, 0);
        std::memcpy(tmp_gpu1_app.data(), begin_gpu1, sizeof(int) * items);*/

        //---device_buffers
        /*auto begin_gpu0_dev = mem_device->data_in_buffer_device<int>({ CL_DEVICE_TYPE_GPU,0 });
        std::vector<int> tmp_gpu0_dev(items, 0);
        std::memcpy(tmp_gpu0_dev.data(), begin_gpu0_dev, sizeof(int) * items);

        auto begin_gpu1_dev = mem_device->data_in_buffer_device<int>({ CL_DEVICE_TYPE_GPU,1 });
        std::vector<int> tmp_gpu1_dev(items, 0);
        std::memcpy(tmp_gpu1_dev.data(), begin_gpu1_dev, sizeof(int) * items);*/

        for (size_t i = 0; i < items; i++)
        {
            const int* begin = i < data_split ? begin_gpu0 : begin_gpu1;
            if (begin[i] != 2)
            {
                std::cerr << "Some value_error at position [" << i << "] fixme !!!" << std::endl;
                return -1;
            }
        }
        std::cout << "<OK>\n";
        std::cout << "--------------------------------\n";
    }
    return err;
}


static int call_test(const int items,
                     virtual_device& device,
                     std::unique_ptr<clTask>& task)
{
    int err = 0;

    if (device.cnt_devices() < 2) return CL_INVALID_PLATFORM;
	const auto offload_dev1 = 0.5f;
    std::vector<int> d1(items, 1);

	
    auto mem_device_a = device.alloc(d1, true, { CL_DEVICE_TYPE_CPU,0 });
	const auto gpu_id = 0;    
	std::cout << "Call test test_async_execute_and_copy_gpu_cpu ..." << std::endl;
    err = test_async_execute_and_copy_gpu_cpu(device, mem_device_a, task, items, offload_dev1,gpu_id);
    if (err != 0)return err;

    const auto cnt_gpus = device.cnt_gpus();
    if (err != 0)return err;

	if (cnt_gpus >= 2)
	{
        auto mem_device_b = device.alloc(d1, true, { CL_DEVICE_TYPE_CPU, 0 });
		std::cout << "Call test test_async_execute_and_copy_2gpus ..." << std::endl;
		err = test_async_execute_and_copy_2gpus(device, mem_device_b, task, items, offload_dev1,5);
		if (err != 0)return err;
	}
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

    const auto task_name = "set";
    const std::string task_body = R"(
		__kernel void set (global int* output)
		{
			const int tid = get_global_id(0);
			output[tid] = 2;
		}
	)";

    auto task = device.create_task(task_body, task_name);
    if (task == nullptr)return -1;

    err = call_test(256,device,task); //1KB
    if (err != CL_SUCCESS) {
        std::cerr << "Some error: " << err << " fixme!!!" << std::endl;
        return err;
    }

	const auto size = 256 * 64e3f;//4096*4096=67MB
	std::cout << "Memory size:\t" << size*4 * 1e-6f << " MB\n";

    err = call_test(size, device, task); // x MB
    if (err != CL_SUCCESS) {
        std::cerr << "Some error: " << err << " fixme!!!" << std::endl;
        return err;
    }

    std::cout << "-------------------------\n";
    std::cout << " Passed ...!" << std::endl;
    std::cout << "-------------------------\n";
    return err;
}

