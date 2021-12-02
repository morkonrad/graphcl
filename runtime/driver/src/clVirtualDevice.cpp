#include "clVirtualDevice.h"

#include <iostream>
#include <future>
#include <mutex>

#include <sstream>
#include <cmath>
#include <numeric>
#include <chrono>


//#define _search_sequential_

/**
 * @brief virtual_device::intialize
 * @param status
 */
void virtual_device::intialize(std::string& status, const cl_device_type device_type)
{
    int err = 0;
    const auto start = std::chrono::system_clock::now();

    err = cl::Platform::get(&_cl_platforms);
    if (err != CL_SUCCESS) {
        on_coopcl_error(err);
        status = "Some error on scan platforms";
        return;
    }

    auto find_device_platform_context = [](
                                            std::string& status,
                                            std::vector<std::pair<cl::Context, cl::Device>>& cl_items,
                                            const std::vector<cl::Platform>& cl_platforms,
                                            const cl_device_type devt)->int
    {
        int err = 0;
        auto encode_dev_type = [](const cl_device_type dev_type)->std::string
        {
            const std::string dev_type_name = dev_type == CL_DEVICE_TYPE_CPU ? "CPU" : dev_type == CL_DEVICE_TYPE_GPU ? "GPU" : "ACC";
            return dev_type_name;
        };

        //Scan platforms to find devices
        bool ctx_created = false;
        cl::Context common_ctx;

        for (auto& p : cl_platforms)
        {
            std::vector<cl::Device> devices;
            err = p.getDevices(devt, &devices);
            if (err == CL_DEVICE_NOT_FOUND)
            {
                err = 0; continue;
            }

            if (on_coopcl_error(err) != CL_SUCCESS) {
                status = "Some error on scan devices";
                return err;
            }

            if (!devices.empty())
            {
                const auto pname = p.getInfo<CL_PLATFORM_NAME>(&err);
                if (on_coopcl_error(err) != CL_SUCCESS) {
                    status = "Some error on scan platforms";
                    return err;
                }
                std::cout << "Found platform:\t" << pname << "\n";

                int did = 0;
                for (auto& device : devices)
                {
                    const auto dt = device.getInfo<CL_DEVICE_TYPE>(&err);
                    std::cout << "Device[" << did++ << "]:\t" << device.getInfo<CL_DEVICE_NAME>(&err) <<
                        " <" << encode_dev_type(dt) << ">" << "\n";

                    if (dt == CL_DEVICE_TYPE_CPU)
                    {
                        //#ifdef __linux__
                        // create single context pro device
                        auto ctx = cl::Context(device, nullptr, nullptr, nullptr, &err);
                        if (on_coopcl_error(err) != 0) {
                            status = "Some error on create context";
                            return err;
                        }
                        cl_items.push_back({ctx,device});
                        //#else
                        //						//If CPU than split cores and create a sub-device
                        //						//USE CPU_FISSION extension! to
                        //						//decouple host thread from device threads.
                        //						//create a single sub_device with CU-1

                        //						const auto compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
                        //						if (on_coopcl_error(err) != 0) {
                        //							status = "Some error on scan devices";
                        //							return err;
                        //						}

                        //						const cl_device_partition_property props[] = {
                        //							CL_DEVICE_PARTITION_BY_COUNTS,compute_units - 2,
                        //							CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,0 };

                        //						std::vector<cl::Device> sub_devs;
                        //						err = device.createSubDevices(&props[0], &sub_devs);
                        //						if (on_coopcl_error(err) != 0) {
                        //							status = "Some error on scan devices";
                        //							return err;
                        //						}

                        //						auto ctx = cl::Context(sub_devs, nullptr, nullptr, nullptr, &err);
                        //						if (on_coopcl_error(err) != 0) {
                        //							status = "Some error on scan devices";
                        //							return err;
                        //						}

                        //						cl_items.push_back({ctx,sub_devs[0]});
                        //#endif
                    }
                    else
                    {
                        /// create shared context pro device-type
                        if(!ctx_created)
                        {
                            // create single context pro device
                            common_ctx = cl::Context(devices, nullptr, nullptr, nullptr, &err);
                            if (on_coopcl_error(err) != 0) {
                                return err;
                            }
                            ctx_created=true;
                        }
                        cl_items.push_back({ common_ctx,device });

                        /// create single context pro device
                        //                        auto ctx = cl::Context(device, nullptr, nullptr, nullptr, &err);
                        //                        if (on_coopcl_error(err) != 0) {
                        //                            status = "Some error on create context";
                        //                            return err;
                        //                        }
                        //                        cl_items.push_back({ctx,device});
                    }
                }
                std::cout << std::flush;
                break;
            }
        }

        return err;
    };

    std::mutex cout_mutex;  // cout guard

    auto async_find_device_context_pairs = [&cout_mutex](
                                               const std::vector<cl::Platform>& cl_platforms,
                                               const cl_device_type devt)->std::vector<std::pair<cl::Context,cl::Device>>
    {
        std::vector<std::pair<cl::Context, cl::Device>> cl_items;

        auto encode_dev_type = [](const cl_device_type dev_type)->std::string
        {
            const std::string dev_type_name = dev_type == CL_DEVICE_TYPE_CPU ? "CPU" : dev_type == CL_DEVICE_TYPE_GPU ? "GPU" : "ACC";
            return dev_type_name;
        };

        bool ctx_created = false;
        cl::Context common_ctx;

        //Scan platforms to find devices
        for (auto& p : cl_platforms)
        {
            std::vector<cl::Device> devices;
            auto err = p.getDevices(devt, &devices);
            if (err == CL_DEVICE_NOT_FOUND) {
                err = 0; continue;
            }

            if (on_coopcl_error(err) != CL_SUCCESS) {
                return cl_items;
            }

            if (!devices.empty())
            {
                const auto pname = p.getInfo<CL_PLATFORM_NAME>(&err);
                if (on_coopcl_error(err) != CL_SUCCESS) {
                    return cl_items;
                }

                {
                    const std::lock_guard<std::mutex> lock(cout_mutex);
                    std::cout << "Found platform:\t" << pname << "\n";
                }

                int did = 0;
                for (auto& device : devices)
                {
                    const auto dt = device.getInfo<CL_DEVICE_TYPE>(&err);
                    {
                        const std::lock_guard<std::mutex> lock(cout_mutex);
                        std::cout << "Device[" << did++ << "]:\t" << device.getInfo<CL_DEVICE_NAME>(&err) <<
                            " <" << encode_dev_type(dt) << ">" << "\n";
                    }

                    if (dt == CL_DEVICE_TYPE_CPU)
                    {
                        //#ifdef __linux__
                        // create single context pro device
                        auto ctx = cl::Context(device, nullptr, nullptr, nullptr, &err);
                        if (on_coopcl_error(err) != 0) {
                            return cl_items;
                        }
                        cl_items.push_back({ctx,device});
                        //#else
                        //                        //If CPU than split cores and create a sub-device
                        //                        //USE CPU_FISSION extension! to
                        //                        //decouple host thread from device threads.
                        //                        //create a single sub_device with CU-1
                        //						const auto compute_units = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err);
                        //						if (on_coopcl_error(err) != 0){
                        //							return cl_items;
                        //						}

                        //						const cl_device_partition_property props[] = {
                        //							CL_DEVICE_PARTITION_BY_COUNTS,compute_units - 2,
                        //							CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,0 };

                        //						std::vector<cl::Device> sub_devs;
                        //						err = device.createSubDevices(&props[0], &sub_devs);
                        //						if (on_coopcl_error(err) != 0) {
                        //							return cl_items;
                        //						}

                        //						auto ctx = cl::Context(sub_devs, nullptr, nullptr, nullptr, &err);
                        //						if (on_coopcl_error(err) != 0) {

                        //							return cl_items;
                        //						}

                        //						cl_items.push_back({ ctx,sub_devs[0] });
                        //#endif
                    }
                    else
                    {
                        /// create shared context pro device-type
                        if(!ctx_created)
                        {
                            // create single context pro device
                            common_ctx = cl::Context(devices, nullptr, nullptr, nullptr, &err);
                            if (on_coopcl_error(err) != 0) {
                                return cl_items;
                            }
                            ctx_created=true;
                        }
                        cl_items.push_back({ common_ctx,device });

                        //// create single context pro device
                        //                        auto ctx = cl::Context(device, nullptr, nullptr, nullptr, &err);
                        //                        if (on_coopcl_error(err) != 0) {
                        //                            return cl_items;
                        //                        }
                        //                        cl_items.push_back({ ctx,device });
                    }
                }
                std::cout << std::flush;
                break;
            }
        }
        return cl_items;
    };

    if (device_type == CL_DEVICE_TYPE_CPU) {
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_CPU);
    }
    else if (device_type == CL_DEVICE_TYPE_GPU) {
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_GPU);
    }
    else if (device_type == CL_DEVICE_TYPE_ACCELERATOR)
    {
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_ACCELERATOR);
    }
    else if (device_type == CL_DEVICE_TYPE_ALL)
    {

#ifdef _search_sequential_
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_GPU);
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_CPU);        
        find_device_platform_context(status, _cl_devices_context, _cl_platforms, CL_DEVICE_TYPE_ACCELERATOR);
#else
        std::vector<std::future<std::vector<std::pair<cl::Context, cl::Device>>>> future_context_devices(3);
		future_context_devices[0] = std::async(async_find_device_context_pairs, _cl_platforms, CL_DEVICE_TYPE_GPU);
		future_context_devices[1] = std::async(async_find_device_context_pairs, _cl_platforms, CL_DEVICE_TYPE_CPU);        
        future_context_devices[2] = std::async(async_find_device_context_pairs, _cl_platforms, CL_DEVICE_TYPE_ACCELERATOR);

        std::map<std::string, bool> names;
        for (auto& future : future_context_devices)
        {
            const auto pairs = future.get();

            for (auto& p : pairs)
            {
                auto name = p.second.getInfo<CL_DEVICE_NAME>(&err);
                if (on_coopcl_error(err) != CL_SUCCESS)return;

                const auto it = names.find(name);
                if (it==names.end()) {
                    names[name] = 0;
                    _cl_devices_context.push_back(p);
                }
                //_cl_devices_context.push_back(p);
            }
        }
#endif
    }


    size_t did_cpu = 0;
    size_t did_gpu = 0;
    size_t did_acc = 0;
    size_t did = 0;

    std::vector<std::future<int>> future_cldevices(_cl_devices_context.size());

#ifndef _search_sequential_
    size_t id = 0;
#endif

    for (auto& item : _cl_devices_context)
    {
        auto& [ctx, device] = item;

        const auto devtype = device.getInfo<CL_DEVICE_TYPE>(&err);
        if (on_coopcl_error(err) != CL_SUCCESS)return;

        if (devtype == CL_DEVICE_TYPE_CPU) did = did_cpu++;
        else if (devtype == CL_DEVICE_TYPE_GPU) did = did_gpu++;
        else if (devtype == CL_DEVICE_TYPE_ACCELERATOR) { did = did_acc++; }

#ifdef _search_sequential_
        //size_t count_of_platform_devices = 0;
        _map_devices[&ctx] = clDevice(err, ctx, device, did);
#else
        auto& cldevice = _map_devices[&ctx];
        future_cldevices[id++] = std::async([&cldevice, &ctx, &device, did]
                                            {
                                                return cldevice.create_async(ctx, device, did);
                                            });
#endif
        if (on_coopcl_error(err) != CL_SUCCESS)return;
    }

    for (auto& future : future_cldevices)
    {
        if (future.valid()) {
            auto err = future.get();
            if (on_coopcl_error(err) != CL_SUCCESS)return;
        }
    }

    if (_cl_devices_context.size() > _MAX_DEVICES_COUNT)
    {
        std::cerr << "Driver will not work, Design Failure --> EXIT--> FIXME!!!\n";
        err = -9999;
        return;
    }

    const auto end = std::chrono::system_clock::now();
    const std::chrono::duration<double> diff = end - start;
    std::cout << "Virtual_device build time:\t" << diff.count()*1e3f << std::endl;
    std::cout << "---------------------------------------"<<std::endl;
    //set_status_virtual_device(true);
}

std::unique_ptr<clTask> virtual_device::create_task(
    const std::string& task_body,
    const std::string& task_name,
    const std::string jit_flags)
{
    auto task = std::make_unique<clTask>();
    int err = 0;

    std::cout << "JIT-build kernel: " << task_name << " ... ";

    for (auto&[ctx, device] : _map_devices)
    {
        err = task->build_task(ctx, &device, task_name, task_body, jit_flags);
        if (on_coopcl_error(err) != 0) return nullptr;
    }

    std::cout << " <CREATED> \n";
    return task;
}

void virtual_device::calculate_statistics(const std::vector<float>& data,
                                          const std::string msg,
                                          std::ostream* ptr_stream_out)const
{
    if (data.empty())return;

    const float zero = 0.0f;
    const auto min_max = std::minmax_element(data.begin(), data.end());
    const auto mean_time = std::accumulate(data.begin(), data.end(), zero) / data.size();
    const auto err_time_min = mean_time - *min_max.first;
    const auto err_time_max = *min_max.second - mean_time;

    if (ptr_stream_out != nullptr)
    {
        *ptr_stream_out << msg << "," <<
            mean_time << "," <<
            err_time_min << "," <<
            err_time_max << "\n";
    }
};

void virtual_device::calculate_workload_distribution_statistics(
    const std::vector<workload_distribution_single_device>& data,
    const std::string msg,
    std::ostream* ptr_stream_out)const
{
    if (data.empty())return;

    std::map<std::uint8_t, std::vector<float>> cpu_workitems;
    std::map<std::uint8_t, std::vector<float>> gpu_workitems;

    for (const auto& it : data)
    {
        if (it.device_type == CL_DEVICE_TYPE_GPU)
            gpu_workitems[it.device_id].push_back(it.value);
        else
            cpu_workitems[it.device_id].push_back(it.value);
    }

    auto remove_min_max = [](std::vector<float>& data, bool erase_min = false)
    {
        if (data.empty())return;
        if (erase_min)
            data.erase(std::min_element(data.begin(), data.end()));

        data.erase(std::max_element(data.begin(), data.end()));
    };
    for (auto& [key, collection] : cpu_workitems) {
        remove_min_max(collection);
        std::stringstream txt;
        txt << "CPU" << (int)key << " [%]";
        calculate_statistics(collection, txt.str(), ptr_stream_out);
    }

    for (auto& [key, collection] : gpu_workitems)
    {
        std::stringstream txt;
        txt << "GPU" << (int)key << " [%]";
        calculate_statistics(collection, txt.str(), ptr_stream_out);
    }
}




static auto count_devices = [](const cl_device_type query_type,const std::map<const cl::Context*, clDevice>& devices)->std::uint8_t
{
    std::uint8_t count = 0;
    for (auto&[ctx, device] : devices)
    {
        if (query_type == device.device_type())
            count++;
    }
    return count;
};

std::uint8_t virtual_device::cnt_gpus()const { return count_devices(CL_DEVICE_TYPE_GPU,_map_devices); }

std::uint8_t virtual_device::cnt_cpus()const { return count_devices(CL_DEVICE_TYPE_CPU, _map_devices); }

std::uint8_t virtual_device::cnt_accelerators()const { return count_devices(CL_DEVICE_TYPE_ACCELERATOR, _map_devices); }

std::uint8_t virtual_device::cnt_devices()const { 
    std::uint8_t count_all = 0;
    auto gpus = cnt_gpus();
    auto cpus = cnt_cpus();
    auto accs = cnt_accelerators();
    count_all = gpus + cpus + accs;
    return count_all;
}
