#include "clMemory.h"
#include <iostream>


int clMemory::allocate(
    const std::tuple<const cl_device_id, const cl::CommandQueue*, const cl::Context*>&  device_command_queue_context,
    const size_t cnt_items,
    const size_t size_items_bytes,
    const bool read_only,
    const size_t in_device_type,
    const std::uint8_t in_device_id)
{
    int err = 0;
    // set size of allocated memory and size of a single item
    _single_item_size_bytes = size_items_bytes / cnt_items;
    _size_bytes = size_items_bytes;
    _is_input = false;

    if (read_only)
        _is_input = true;

    const cl_mem_flags flag_alloc_device_buffer = CL_MEM_READ_WRITE; // buffer used by kernel function
    const cl_mem_flags flag_alloc_app_buffer = flag_alloc_device_buffer | CL_MEM_ALLOC_HOST_PTR; // buffer for user_API

    auto&[cldevice, ptr_commnad_queue,context] = device_command_queue_context;

    size_t dev_type = 0;
    err = clGetDeviceInfo(cldevice, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, nullptr);
    if (on_coopcl_error(err) != 0)return  err;

    //Select device that allocates memory
    if (in_device_type != CL_DEVICE_TYPE_ALL)
    {
        //check if type of cldevice is same as input  device_type arg.
        //if not same exit!
        if (in_device_type != dev_type)
            return 0;
    }

    // store pair cldevice, ptr_command_queue
    const map_device_info device_type_id = { dev_type,in_device_id };
    _map_device_queue_io[cldevice] = ptr_commnad_queue;
    _map_device_context[cldevice] = context;
    _map_device_type_id[cldevice] = device_type_id;
    _map_device_type_conetxt[device_type_id] = context;

    //allocate memory on device and on host(app)
    if (dev_type == CL_DEVICE_TYPE_CPU)
    {
        _map_device_app_buffers[cldevice] = { cl::Buffer(*context, flag_alloc_device_buffer, size_items_bytes, nullptr, &err),cl::Buffer() };
        if (on_coopcl_error(err) != CL_SUCCESS)return err;
    }
    else
    {
        int err_alloc_dev_mem = 0;
        int err_alloc_app_mem = 0;

#ifdef USE_NV_DRIVER

        const auto devs = context->getInfo<CL_CONTEXT_DEVICES>(&err);
        if (on_coopcl_error(err) != 0)return  err;

        auto platform = devs.begin()->getInfo<CL_DEVICE_PLATFORM>(&err);
        if (on_coopcl_error(err) != 0)return  err;

        cl_mem(*clCreateBufferNV)(cl_context, cl_mem_flags, cl_mem_flags_NV, size_t, void*, cl_int*) =
            (cl_mem(*)(cl_context, cl_mem_flags, cl_mem_flags_NV, size_t, void*, cl_int*)) clGetExtensionFunctionAddressForPlatform(platform, "clCreateBufferNV");
        if (clCreateBufferNV == NULL) { std::cerr << "Invalid function pointer request for clCreateBufferNV extension \n"; return CL_MEM_OBJECT_ALLOCATION_FAILURE; }

        //cl_mem_properties props[3] = { CL_MEM_PROPERTIES,CL_MEM_PINNED_NV, 0 };
        //auto application_buffer = cl::Buffer(clCreateBufferWithProperties((*context)(), &props[0], flag_alloc_app_buffer, size_items_bytes, nullptr, &err_alloc_app_mem));
        
        auto application_buffer = cl::Buffer(clCreateBufferNV((*context)(), flag_alloc_app_buffer , CL_MEM_PINNED_NV /*CL_MEM_LOCATION_HOST_NV*/, size_items_bytes, nullptr, &err_alloc_app_mem));
        auto device_buffer = cl::Buffer(*context, flag_alloc_device_buffer, size_items_bytes, nullptr, &err_alloc_dev_mem);

#else		
        auto application_buffer = cl::Buffer(*context, flag_alloc_app_buffer, size_items_bytes, nullptr, &err_alloc_app_mem);
        auto device_buffer = cl::Buffer(*context, flag_alloc_device_buffer, size_items_bytes, nullptr, &err_alloc_dev_mem);
#endif

        ///Pre-pin this buffer on first-use time!
        cl::Event wait_map;
        auto cq = _map_device_queue_io.at(cldevice);
        int* map_ptr =static_cast<int*>(cq->enqueueMapBuffer(application_buffer, true, CL_MAP_WRITE, 0, _size_bytes, nullptr, nullptr, &err));
        if (on_coopcl_error(err) != 0)return  err;
        *map_ptr = 0;
        err = cq->enqueueUnmapMemObject(application_buffer, map_ptr, nullptr, &wait_map);
        if (on_coopcl_error(err) != 0)return  err;

        if (on_coopcl_error(err_alloc_app_mem) != CL_SUCCESS)return err_alloc_app_mem;
        if (on_coopcl_error(err_alloc_dev_mem) != CL_SUCCESS)return err_alloc_dev_mem;
        _map_device_app_buffers[cldevice] = { device_buffer,application_buffer };
    }

    return err;

}

int clMemory::copy_application_memory(const void* src_application, const size_t copy_to_device_type, const std::uint8_t copy_to_device_id)
{
    int err = 0;

    /// copy from app_src pointer in application buffer if user provided src!=nullptr and clMemeory object is marked as input
    auto map_memory_sync = [](cl::Buffer& buffer_memory, const cl::CommandQueue& cq,
                              const size_t flag_map, void* src_application, const size_t size, const size_t offset = 0)->int
    {
        cl::Event wait_map; int err{ 0 };
        auto map_ptr = cq.enqueueMapBuffer(buffer_memory, true, flag_map, offset, size,nullptr,nullptr,&err);
        if (on_coopcl_error(err) != CL_SUCCESS)return err;

        memcpy(map_ptr, src_application, size);

        err = cq.enqueueUnmapMemObject(buffer_memory, map_ptr,nullptr,&wait_map);
        if (on_coopcl_error(err) != CL_SUCCESS)return err;

        return wait_map.wait();
        
    };

    if (_is_input && src_application != nullptr)
    {
        for (auto& [cldevice, device_app_buffers] : _map_device_app_buffers)
        {
            auto& [dev_type, dev_id] = _map_device_type_id.at(cldevice);

            if (copy_to_device_type == CL_DEVICE_TYPE_ALL || (copy_to_device_type == dev_type && copy_to_device_id == dev_id ))
            {
                //mark/store this device as a device that has valid memory
                _device_with_coherent_memory = { copy_to_device_type,copy_to_device_id };

                auto& [device_memory, app_memory] = device_app_buffers;

                // cpu has only device buffer, cpu shares device memory with application, thus use device_memory
                auto& dst_buffer = dev_type == CL_DEVICE_TYPE_CPU ? device_memory : app_memory;
                err = map_memory_sync(dst_buffer, *_map_device_queue_io.at(cldevice), CL_MAP_WRITE_INVALIDATE_REGION, (void*)src_application, _size_bytes);
                if (on_coopcl_error(err) != CL_SUCCESS)return err;
            }
        }
    }

    return err;
}

auto find_device = [](const map_device_info& query_device_type_id, 
	const std::map<const cl_device_id, map_device_info>& map_with_devices)->cl_device_id
{
    cl_device_id device_cl = nullptr;
    const auto& [in_dev_type, in_dev_id] = query_device_type_id;

    // find context based on the device_type and device_id
    for (const auto& [device, dev_type_id] : map_with_devices)
    {
        const auto& [dev_type, dev_id] = dev_type_id;
        if (dev_type == in_dev_type && in_dev_id == dev_id)
        {
            device_cl = device; break;
        }
    }

    if (device_cl == nullptr)
        throw std::runtime_error("CL_INVALID_DEVICE on data_in_buffer call, FIXME");
    return device_cl;
};

int clMemory::copy_async_h2h(const bool h2d, const std::vector<const clAppEvent*>& wait_events_in, 
	clAppEvent& wait_event_out, const size_t begin_byte, const size_t end_byte, const map_device_info& other_device_id)
{	
    //get device,queue,buffers CPU-HOST
    const auto cpu_id = std::make_pair(CL_DEVICE_TYPE_CPU, 0);
    const auto cpu_device = find_device(cpu_id,_map_device_type_id);
    auto cq_cpu = _map_device_queue_io.at(cpu_device);
    auto& [app_buffer_cpu, device_buffer_cpu] = _map_device_app_buffers.at(cpu_device);
    auto ctx_cpu = _map_device_type_conetxt.at(cpu_id);

    //get device,queue,buffers other device
    const auto other_device = find_device(other_device_id, _map_device_type_id);
    auto& [app_buffer_other, device_buffer_other] = _map_device_app_buffers.at(other_device);
    auto cq_other = _map_device_queue_io.at(other_device);
    auto ctx_other = _map_device_type_conetxt.at(other_device_id);

    const auto size = end_byte - begin_byte;
    int err{ 0 };


    std::vector<cl::Event> wait_list;
    //gather wait_events from context
    for (auto& wait_event_in : wait_events_in) {
        err = wait_event_in->get_events_in_context(ctx_other, wait_list);
        if (on_coopcl_error(err) != 0) return err;
    }

    cl::Event wait_ev_copy;
    if (h2d)
    {
        auto map_ptr_cpu = cq_cpu->enqueueMapBuffer(app_buffer_cpu,true,CL_MAP_READ, begin_byte, size);
        if (on_coopcl_error(err) != 0)return err;

        err = cq_other->enqueueWriteBuffer(app_buffer_other, false, begin_byte, size, map_ptr_cpu, &wait_list, &wait_ev_copy);
        if (on_coopcl_error(err) != 0)return err;

        err = cq_cpu->enqueueUnmapMemObject(app_buffer_cpu, map_ptr_cpu);
        if (on_coopcl_error(err) != 0)return err;

    }
    else
    {
        auto map_ptr_cpu = cq_cpu->enqueueMapBuffer(app_buffer_cpu, true, CL_MAP_WRITE, begin_byte, size);
        if (on_coopcl_error(err) != 0)return err;

        err = cq_other->enqueueReadBuffer(app_buffer_other, false, begin_byte, size, map_ptr_cpu, &wait_list, &wait_ev_copy);
        if (on_coopcl_error(err) != 0)return err;

        err = cq_cpu->enqueueUnmapMemObject(app_buffer_cpu, map_ptr_cpu);
        if (on_coopcl_error(err) != 0)return err;
    }


    err = cq_other->flush();
    if (on_coopcl_error(err) != 0) return err;

    err = wait_event_out.register_and_create_user_events(wait_ev_copy, _map_device_type_conetxt);
    if (on_coopcl_error(err) != 0) return err;


    return err;
}

int clMemory::enqueue_async_transfer_device_appliction(
    const cl::CommandQueue& cq_device,
    cl::Buffer& application_buffer,
    cl::Buffer& device_memory,
    const bool use_transfer_h2d,
    cl::Event& wait_event,
    const std::vector<cl::Event>& wait_list,
    const size_t size,
    const size_t offset)
{
    int err{ 0 };

    //map and unmap pinned-application memory to get address
    if (use_transfer_h2d) //host-app wants write to device_mem
    {
        err = cq_device.enqueueCopyBuffer(application_buffer, device_memory, offset, offset, size, &wait_list, &wait_event);
        if (on_coopcl_error(err) != 0)return err;
    }
    else //host-app wants read from device_mem
    {
        err = cq_device.enqueueCopyBuffer(device_memory, application_buffer, offset, offset, size, &wait_list, &wait_event);
        if (on_coopcl_error(err) != 0)return err;
    }
    return err;
}

int clMemory::async_transfer_d2d(
    cl::Buffer* src_buff_device,
    cl::Buffer* dst_buff_device,
    clAppEvent& wait_event_out,
    const std::vector<const clAppEvent*>& wait_events_in,
    const size_t begin_byte,
    const size_t size,
    const map_device_info& destination_device)
{
    int err = 0;

    cl::Event wait_ev;
    std::vector<cl::Event> wait_list;
    auto ctx_device = _map_device_type_conetxt.at(destination_device);
    const cl::CommandQueue* cq_device = nullptr;

    for (auto&[device, ctx] : _map_device_context)
    {
        if (ctx_device == ctx)
            cq_device = _map_device_queue_io.at(device);
    }

    if (cq_device == nullptr)return CL_INVALID_COMMAND_QUEUE;
    if (src_buff_device == nullptr) return CL_INVALID_MEM_OBJECT;
    if (dst_buff_device == nullptr) return CL_INVALID_MEM_OBJECT;

    //gather wait_events from context
    for (auto& wait_event_in : wait_events_in) {
        err = wait_event_in->get_events_in_context(ctx_device, wait_list);
        if (on_coopcl_error(err) != 0) return err;
    }

    //WARNING: blocking call with NVIDIA_driver non_blocking call with AMD_driver
    err = cq_device->enqueueCopyBuffer(*src_buff_device, *dst_buff_device, begin_byte, begin_byte, size, &wait_list, &wait_ev);
    if (on_coopcl_error(err) != 0) return err;

    err = cq_device->flush();
    if (on_coopcl_error(err) != 0) return err;

    err = wait_event_out.register_and_create_user_events(wait_ev, _map_device_type_conetxt);
    if (on_coopcl_error(err) != 0) return err;

    return err;
}

const void* clMemory::map_read_data(const bool map_read_device_memory,
                                    const map_device_info& device_type_id, const size_t begin_byte, const size_t end_byte)const
{
    int err = 0;

    size_t offset_read = begin_byte;
    size_t size_read = end_byte - begin_byte;

    if ((size_read == 0) || (size_read > _size_bytes) || (offset_read + size_read > _size_bytes))
    {
        //read whole buffer
        offset_read = 0;
        size_read = _size_bytes;
    }

    cl_device_id src_device_cl = nullptr;
    const auto&[in_dev_type, in_dev_id] = device_type_id;

    // find context based on the device_type and device_id
    for (const auto&[device, dev_type_id] : _map_device_type_id)
    {
        const auto&[dev_type, dev_id] = dev_type_id;
        if (dev_type == in_dev_type && in_dev_id == dev_id)
        {
            src_device_cl = device; break;
        }
    }

    if (src_device_cl == nullptr)
        throw std::runtime_error("CL_INVALID_DEVICE on data_in_buffer call, FIXME");

    auto&[buffer_device, buffer_app] = _map_device_app_buffers.at(src_device_cl);
    auto& cq = _map_device_queue_io.at(src_device_cl);

    if (cq == nullptr)
        throw std::runtime_error("CL_INVALID_QUEUE on data_in_buffer call, FIXME");

    void* map_ptr{ nullptr };

    bool select_device_memory = map_read_device_memory;

    //ADD extra check because CPU has only device buffer!
    if (in_dev_type == CL_DEVICE_TYPE_CPU && !select_device_memory)
        select_device_memory = true;

    if (select_device_memory)
    {
        map_ptr = cq->enqueueMapBuffer(buffer_device, true, CL_MAP_READ, offset_read, size_read, nullptr, nullptr, &err);
        if (on_coopcl_error(err) != 0)return  nullptr;

        err = cq->enqueueUnmapMemObject(buffer_device, map_ptr);
        if (on_coopcl_error(err) != 0)return  nullptr;
    }
    else
    {
        map_ptr = cq->enqueueMapBuffer(buffer_app, true, CL_MAP_READ, offset_read, size_read, nullptr, nullptr, &err);
        if (on_coopcl_error(err) != 0)return  nullptr;
        //std::vector<int> dummy(size_read / sizeof(int));
        //std::memcpy(dummy.data(), map_ptr, size_read);
        err = cq->enqueueUnmapMemObject(buffer_app, map_ptr);
        if (on_coopcl_error(err) != 0)return  nullptr;
    }
    return map_ptr;
}

int clMemory::async_transfer_from_to(
    const bool copy_h2d,
    cl::Buffer* buff_device, 
	cl::Buffer* buff_application,
    clAppEvent& wait_event_out,
    const std::vector<const clAppEvent*>& wait_events_in,
    const size_t begin_byte,
    const size_t size,
    const map_device_info& source_device,
    const map_device_info& destination_device)
{
    int err = 0;

    if (buff_device == nullptr) return CL_INVALID_MEM_OBJECT;
    if (buff_application == nullptr) return CL_INVALID_MEM_OBJECT;

    cl_device_id gpu_device = nullptr;
    //check all devices to find GPU device, since its H2D or D2H transfer
    for (auto&[device, dev_type_id] : _map_device_type_id)
    {
        if (copy_h2d)
        {
            // if H2D dst is GPU
            if (destination_device.first == dev_type_id.first && destination_device.second == dev_type_id.second)
            {
                gpu_device = device; break;
            }
        }
        else
        {
            // if D2H  source is GPU
            if (source_device.first == dev_type_id.first && source_device.second == dev_type_id.second)
            {
                gpu_device = device; break;
            }
        }
    }
    if (gpu_device == nullptr)return CL_INVALID_DEVICE;

    // Select context to wait for possible event from other context (CPU_context)
    //auto ctx_cpu = copy_h2d ? _map_device_type_conetxt.at(source_device) : _map_device_type_conetxt.at(destination_device);
    auto ctx_gpu = _map_device_type_conetxt.at(_map_device_type_id.at(gpu_device));

    // Gather wait_events from CPU context, since H2D and D2H use GPU context and commands than need to gather only CPU events
    // assumed that CPU is in different context as GPU
    std::vector<cl::Event> wait_list;
    for (auto& wait_event_in : wait_events_in) {
        err = wait_event_in->get_events_in_context(ctx_gpu, wait_list);
        if (on_coopcl_error(err) != 0) return err;
    }

    const cl::CommandQueue* cq_gpu = _map_device_queue_io.at(gpu_device);
    if (cq_gpu == nullptr)return CL_INVALID_COMMAND_QUEUE;

    cl::Event wait_ev;
    err = enqueue_async_transfer_device_appliction(*cq_gpu, *buff_application, *buff_device, copy_h2d, wait_ev, wait_list, size, begin_byte);
    if (on_coopcl_error(err) != 0) return err;


    err = cq_gpu->flush();
    if (on_coopcl_error(err) != 0) return err;

    err = wait_event_out.register_and_create_user_events(wait_ev, _map_device_type_conetxt);
    if (on_coopcl_error(err) != 0) return err;

    return err;
}

int clMemory::copy_async(
    const std::vector<const clAppEvent*>& wait_events_in,
    clAppEvent& wait_event_out,
    const size_t begin_byte,
    const size_t end_byte,
    const map_device_info& source_device,
    const map_device_info& destination_device)
{
    int err = 0;

    // check source and destination devices if:
    // to decide if copy from H2D (CPU-->GPU) or  D2D (GPU-->GPU) or D2H (GPU-->CPU)

    // flag CL_DEVICE_TYPE_ALL means each application buffer have valid data, thus copy from CPU
    map_device_info src_device = source_device;
    src_device.first = src_device.first == CL_DEVICE_TYPE_ALL ? CL_DEVICE_TYPE_CPU : src_device.first;

    // get buffers src and dest.
    const auto& [src_device_type, src_device_id] = src_device;
    const auto& [dst_device_type, dst_device_id] = destination_device;

    if (src_device_type == dst_device_type && src_device_id == dst_device_id)
        return 0;

    const size_t size = end_byte - begin_byte;
    if (size == 0)
        return 0;

    cl_device_type src_dev_type = 0;
    cl_device_type dst_dev_type = 0;

    //Select target and destination device types to decide if D2H or H2D
    for (auto& [cldev, dev_type_id] : _map_device_type_id)
    {
        auto& [dev_type, dev_id] = dev_type_id;

        if (dev_type == src_device_type && dev_id == src_device_id)
            src_dev_type = dev_type;
        if (dev_type == dst_device_type && dev_id == dst_device_id)
            dst_dev_type = dev_type;
    }

    //Select buffers
    cl::Buffer* buff_application{ nullptr };
    cl::Buffer* buff_device{ nullptr };
    if (src_dev_type == CL_DEVICE_TYPE_GPU && dst_dev_type == CL_DEVICE_TYPE_GPU)
    {
        //Here select device_local buffer and device_local buffer
        for (auto& [cldev, dev_type_id] : _map_device_type_id)
        {
            auto& [dev_type, dev_id] = dev_type_id;

            if (dev_type == src_device_type && dev_id == src_device_id)
                buff_application = buffer_device(cldev);

            if (dev_type == dst_device_type && dev_id == dst_device_id)
                buff_device = buffer_device(cldev);
        }

        err = async_transfer_d2d(buff_application, buff_device, wait_event_out, wait_events_in,
                                 begin_byte, size, destination_device);
    }
    else
    {
        bool copy_h2d = true;
        if (src_dev_type == CL_DEVICE_TYPE_CPU && dst_dev_type == CL_DEVICE_TYPE_GPU)
        {
            //Select for GPU, since copy data to GPU: transfer to device_local buffer from application_local buffer
            for (auto& [cldev, dev_type_id] : _map_device_type_id)
            {
                auto& [dev_type, dev_id] = dev_type_id;
                if (dev_type == dst_device_type && dev_id == dst_device_id) {
                    buff_application = buffer_application(cldev);
                    buff_device = buffer_device(cldev);
                    break;
                }

            }

            //copy from CPU application buffer to GPU pinned_application memory
            clAppEvent wait_event_copy_device_app_memory;
            auto other_device_id = copy_h2d ? destination_device : source_device;
            err = copy_async_h2h(copy_h2d,wait_events_in, wait_event_copy_device_app_memory, begin_byte, end_byte, other_device_id);
            if (on_coopcl_error(err) != 0)return  err;

            //copy from GPU pinned_application memory to GPU device memory
            err = async_transfer_from_to(copy_h2d, buff_device, buff_application,
                                         wait_event_out, { &wait_event_copy_device_app_memory },
                                         begin_byte, size, source_device, destination_device);

        }
        else if (src_dev_type == CL_DEVICE_TYPE_GPU && dst_dev_type == CL_DEVICE_TYPE_CPU)
        {
            copy_h2d = false; // now set d2h transfer
            //Select for GPU, since copy data from GPU: transfer from device_local buffer to application_local buffer
            for (auto& [cldev, dev_type_id] : _map_device_type_id)
            {
                auto& [dev_type, dev_id] = dev_type_id;

                if (dev_type == src_device_type && dev_id == src_device_id) {
                    buff_device = buffer_device(cldev);
                    buff_application = buffer_application(cldev);
                    break;
                }
            }

            //copy from GPU device memory to GPU pinned_application memory
            clAppEvent wait_event_copy_device_app_memory;
            err = async_transfer_from_to(copy_h2d, buff_device, buff_application,
                                         wait_event_copy_device_app_memory, wait_events_in,
                                         begin_byte, size, source_device, destination_device);
            if (on_coopcl_error(err) != 0)return  err;

            //copy from GPU pinned_application memory to CPU application buffer
            auto other_device_id = copy_h2d ? destination_device : source_device;
            err = copy_async_h2h(copy_h2d, { &wait_event_copy_device_app_memory }, wait_event_out, begin_byte, end_byte, other_device_id);
        }
    }
    return err;
}

int clMemory::merge_async(const std::vector<const clAppEvent*>& wait_events_in, clAppEvent& wait_event_out, 
	const size_t begin_byte, const size_t end_byte, 
	const map_device_info& source_device, const map_device_info& destination_device)
{
    return 0;
}

void clMemory::benchmark_memory()
{

//    check_map_access_latency_from_host_to_device_mem(0, 0);
//    check_map_access_latency_from_host_to_device_mem(1, 0);

//    /*mem_device_b->check_map_access_latency_from_host_to_device_mem(0, 1);
//	mem_device_b->check_map_access_latency_from_host_to_device_mem(1, 1);*/

//    bool write_to_pinned = 0;
//    check_copy_from_to_pinned_mem(write_to_pinned, 0);

//    write_to_pinned = 1;
//    check_copy_from_to_pinned_mem(write_to_pinned, 0);

//    //mem_device_b->check_copy_from_gpu_to_gpu(0, 1);

//    bool d2h = true;
//    check_map_access_latency_from_host_to_pinned_mem(d2h);

//    d2h = false;
//    check_map_access_latency_from_host_to_pinned_mem(d2h);

    check_scatter_to_gpus();

}

const static std::pair<const cl::CommandQueue*,cl_device_id> select_device_queueue(
    const std::map<const cl_device_id, map_device_info>& map_device_type_id,
    const std::map<const cl_device_id, const cl::CommandQueue*>& map_device_queue_io,
    const std::uint8_t gpu_id)
{
    cl_device_id src_device_cl = nullptr;
    // find device based on the device_type and device_id
    for (const auto& [device, dev_type_id] : map_device_type_id)
    {
        const auto& [dev_type, dev_id] = dev_type_id;
        if (dev_type == CL_DEVICE_TYPE_GPU && dev_id == gpu_id)
        {
            src_device_cl = device; break;
        }
    }

    if (src_device_cl == nullptr)
        throw std::runtime_error("CL_INVALID_DEVICE on data_in_buffer call, FIXME");

    auto& cq = map_device_queue_io.at(src_device_cl);

    if (cq == nullptr)
        throw std::runtime_error("CL_INVALID_QUEUE on data_in_buffer call, FIXME");

    return{ cq,src_device_cl };
};

int clMemory::check_scatter_to_gpus()
{
    int err = 0;
    const size_t offset = 0;
    const size_t size = _size_bytes;
    const auto itearations = 5;
    for (int it = 0; it < itearations; it++)
    {
        std::cout << "Iteration:\t[" << it << "/" << itearations << "]\r" << std::flush;
        const auto begin = std::chrono::system_clock::now();
        std::vector<std::future<int>> async_calls_h2d;
        for (auto& [dev, cq] : _map_device_queue_io)
        {
            //only copy to from GPUs
            const auto& [dev_type, dev_id] = _map_device_type_id.at(dev);
            if (dev_type == CL_DEVICE_TYPE_GPU)
            {
                auto& [buf_app, buf_dev] = _map_device_app_buffers.at(dev);

                //parallel copy
                auto async_call = std::async([cq, buf_app, buf_dev, offset, size]
                                             {
                                                 cl::Event copy_event;
                                                 auto err = cq->enqueueCopyBuffer(buf_app, buf_dev, offset, offset, size, nullptr, &copy_event);
                                                 if (on_coopcl_error(err) != 0)return  err;

                                                 err = copy_event.wait();
                                                 if (on_coopcl_error(err) != 0)return  err;

                                                 return err;
                                             });
                async_calls_h2d.push_back(std::move(async_call));
            }
        }

        for (auto& copy : async_calls_h2d)
            copy.wait();
        async_calls_h2d.clear();

        const auto end = std::chrono::system_clock::now();

        const std::chrono::duration<double> diff = end - begin;
        const auto duration_sec = diff.count();
        const float GBsec = _size_bytes * 1e-9f / diff.count();

        std::cout << "Parallel scatter duration:\t" << duration_sec * 1e3f << " ms, Bus-throughput:\t" << GBsec << " GB/sec" << std::endl;
        std::cout << "------------------" << std::endl;
    }
    return err;
}

int clMemory::check_map_access_latency_from_host_to_device_mem(const bool map_read, const std::uint8_t gpu_id)const
{
    int err = 0;

    const size_t offset = 0;
    const size_t size = _size_bytes;

    const auto itearations = 5;
    for (int it = 0; it < itearations; it++)
    {
        std::cout << "Iteration:\t[" << it << "/" << itearations << "]\r" << std::flush;
        auto& [cq, src_device_cl] = select_device_queueue(_map_device_type_id, _map_device_queue_io, gpu_id);

        void* map_ptr{ nullptr };
        auto&[dev, app] = _map_device_app_buffers.at(src_device_cl);
        std::vector<char> dummy(size, 1);

        const auto begin = std::chrono::system_clock::now();
        cl::Event wait_map;
        if (map_read)
        {
            //D2H
            map_ptr = cq->enqueueMapBuffer(dev, true, CL_MAP_READ, offset, size, nullptr, nullptr, &err);
            if (on_coopcl_error(err) != 0)return  err;

            //std::memcpy(dummy.data(), map_ptr, size);

            err = cq->enqueueUnmapMemObject(dev, map_ptr, nullptr, &wait_map);
            if (on_coopcl_error(err) != 0)return  err;
        }
        else
        {
            //H2D
            map_ptr = cq->enqueueMapBuffer(dev, true, CL_MAP_WRITE, offset, size, nullptr, nullptr, &err);
            if (on_coopcl_error(err) != 0)return  err;

            //std::memcpy(map_ptr, dummy.data(), size);

            err = cq->enqueueUnmapMemObject(dev, map_ptr, nullptr, &wait_map);
            if (on_coopcl_error(err) != 0)return  err;
        }

        err = wait_map.wait();
        if (on_coopcl_error(err) != 0)return  err;

        const auto end = std::chrono::system_clock::now();

        const std::chrono::duration<double> diff = end - begin;
        const auto duration_sec = diff.count();
        const float GBsec = _size_bytes * 1e-9f / diff.count();
        const auto d2h = map_read == true ? "read to host " : "write from host";
        std::cout << "Mapping_pageable " << d2h << " duration:\t" << duration_sec * 1e3f << " ms, Bus-throughput:\t" << GBsec << " GB/sec" << std::endl;
        std::cout << "------------------" << std::endl;
    }
    return err;
}

int clMemory::check_map_access_latency_from_host_to_pinned_mem(const bool map_read, const std::uint8_t gpu_id)const
{
    int err = 0;

    const size_t offset = 0;
    const size_t size = _size_bytes;

    auto& [cq, src_device_cl] = select_device_queueue(_map_device_type_id, _map_device_queue_io, gpu_id);
    void* map_ptr{ nullptr };

    const auto begin = std::chrono::system_clock::now();

    auto& [dev, app] = _map_device_app_buffers.at(src_device_cl);

    cl::Event wait_map;
    if (map_read)
    {
        //D2H
        map_ptr = cq->enqueueMapBuffer(app, true, CL_MAP_READ, offset, size, nullptr, nullptr, &err);
        if (on_coopcl_error(err) != 0)return  err;

        err = cq->enqueueUnmapMemObject(app, map_ptr,nullptr,&wait_map);
        if (on_coopcl_error(err) != 0)return  err;
    }
    else
    {
        //H2D
        map_ptr = cq->enqueueMapBuffer(app, true, CL_MAP_WRITE, offset, size, nullptr, nullptr, &err);
        if (on_coopcl_error(err) != 0)return  err;

        err = cq->enqueueUnmapMemObject(app, map_ptr,nullptr, &wait_map);
        if (on_coopcl_error(err) != 0)return  err;
    }

    err = wait_map.wait();
    if (on_coopcl_error(err) != 0)return  err;

    const auto end = std::chrono::system_clock::now();

    const std::chrono::duration<double> diff = end - begin;
    const auto duration_sec = diff.count();
    const float GBsec = _size_bytes * 1e-9f / diff.count();
    const auto d2h = map_read == true ? "reads<->host" : "write<->host";
    std::cout << "Mapping pinned " << d2h << " duration: \t" << duration_sec * 1e3f << " ms, Bus-throughput:\t" << GBsec << " GB/sec" << std::endl;
    std::cout << "------------------" << std::endl;

    return err;
}

int clMemory::check_copy_from_to_pinned_mem(const bool write_pinned, const std::uint8_t gpu_id)const
{
    int err = 0;
    const size_t offset = 0;
    const size_t size = _size_bytes;

    auto& [cq, src_device_cl] = select_device_queueue(_map_device_type_id, _map_device_queue_io, gpu_id);


    cl::Event copy_event;
    const auto begin = std::chrono::system_clock::now();

    auto& [dev, app] = _map_device_app_buffers.at(src_device_cl);

    if (write_pinned)
    {
        err = cq->enqueueCopyBuffer(dev, app, offset, offset, size, nullptr, &copy_event);
        if (on_coopcl_error(err) != 0)return  err;
    }
    else
    {
        err = cq->enqueueCopyBuffer(app, dev, offset, offset, size, nullptr, &copy_event);
        if (on_coopcl_error(err) != 0)return  err;
    }

    err = copy_event.wait();
    if (on_coopcl_error(err) != 0)return  err;

    const auto end = std::chrono::system_clock::now();

    const std::chrono::duration<double> diff = end - begin;
    const auto duration_sec = diff.count();
    const float GBsec = _size_bytes * 1e-9f / diff.count();
    const auto d2h = write_pinned == true ? "from device to pinned" : "from pinned to device";
    std::cout << "Enqueue copy " << d2h << " duration:\t" << duration_sec * 1e3f << " ms, Bus-throughput:\t" << GBsec << " GB/sec" << std::endl;
    std::cout << "------------------" << std::endl;
    return err;
}

int clMemory::check_copy_from_gpu_to_gpu(const std::uint8_t gpu_id_src, const std::uint8_t gpu_id_dst)const
{
    int err = 0;
    const size_t offset = 0;
    const size_t size = _size_bytes;

    cl_device_id src_device_cl = nullptr;
    cl_device_id dst_device_cl = nullptr;

    // find device based on the device_type and device_id
    for (const auto& [device, dev_type_id] : _map_device_type_id)
    {
        const auto& [dev_type, dev_id] = dev_type_id;
        if (dev_type == CL_DEVICE_TYPE_GPU && dev_id == gpu_id_src)
        {
            src_device_cl = device;
        }

        if (dev_type == CL_DEVICE_TYPE_GPU && dev_id == gpu_id_dst)
        {
            dst_device_cl = device;
        }
    }

    if (src_device_cl == nullptr)
        throw std::runtime_error("CL_INVALID_DEVICE on data_in_buffer call, FIXME");

    if (dst_device_cl == nullptr)
        throw std::runtime_error("CL_INVALID_DEVICE on data_in_buffer call, FIXME");

    auto& cq = _map_device_queue_io.at(src_device_cl);

    if (cq == nullptr)
        throw std::runtime_error("CL_INVALID_QUEUE on data_in_buffer call, FIXME");

    void* map_ptr{ nullptr };
    std::vector<std::uint8_t> dummy(size);

    cl::Event copy_event;
    const auto begin = std::chrono::system_clock::now();

    auto& [dev_src, app_src] = _map_device_app_buffers.at(src_device_cl);
    auto& [dev_dst, app_dst] = _map_device_app_buffers.at(dst_device_cl);

    err = cq->enqueueCopyBuffer(dev_src, dev_dst, offset, offset, size, nullptr, &copy_event);
    if (on_coopcl_error(err) != 0)return  err;

    err = copy_event.wait();
    if (on_coopcl_error(err) != 0)return  err;

    const auto end = std::chrono::system_clock::now();

    const std::chrono::duration<double> diff = end - begin;
    const auto duration_sec = diff.count();
    const float GBsec = _size_bytes * 1e-9f / diff.count();

    std::cout << "Enqueue copy d2d duration:\t" << duration_sec * 1e3f << " ms, Bus-throughput:\t" << GBsec << " GB/sec" << std::endl;
    std::cout << "------------------" << std::endl;
    return err;
}
