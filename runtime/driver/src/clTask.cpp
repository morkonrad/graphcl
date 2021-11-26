#include "clTask.h"
#include "clCommon.h"
#include <sstream>

static size_t tid = 0;

int clTask::build_task(
    const cl::Context* ptr_ctx,
    clDevice* ptr_clDevice,
    const std::string& task_name,
    const std::string& task_body,
    const std::string jit_flags)
{
    int err = 0;

    tid = 0;

    _name = task_name;
    err = ptr_clDevice->build_kernel(task_name, task_body, jit_flags);
    if (err != CL_SUCCESS)return err;

    _map_devices[ptr_ctx] = ptr_clDevice;
    if (ptr_clDevice->device_type() == CL_DEVICE_TYPE_CPU)
    {
        // do analysis with Intel-CPU OpenCL-driver, since all others do not provide this feature!
        err = identify_kernel_input_output_memory(ptr_clDevice->kernel());
        if (err != CL_SUCCESS)return err;
    }
    return err;
}


int clTask::set_task_arg(std::uint8_t& id, std::unique_ptr<clMemory>& arg)
{
    int err = 0;

    if (_kernel_arguments[id]._ADDRESS_QUALIFIER != CL_KERNEL_ARG_ADDRESS_GLOBAL)
        return -1;

    bool is_input = false;
    const auto is_set = _kernel_arguments[id]._TYPE_QUALIFIER & CL_KERNEL_ARG_TYPE_CONST;
    if (is_set == CL_KERNEL_ARG_TYPE_CONST) is_input = true;
    arg->set_as_input(is_input);

    for (auto& [ctx,device] : _map_devices)
    {
        //auto ptr_buff = arg->buffer(ctx);
        auto ptr_buff = arg->buffer_device(device->device_ptr());
        err = device->set_arg(id, *ptr_buff);
        if (on_coopcl_error(err) != CL_SUCCESS)return err;
    }
    return err;
}

int clTask::set_task_arg(std::uint8_t& id, cl::LocalSpaceArg& arg)
{
    int err = 0;
    for (auto& [ctx, device] : _map_devices)
    {
        err = device->set_arg(id, arg);
        if (err != CL_SUCCESS)return err;
    }
    return err;
}

int clTask::async_execute(
    const std::vector<const clAppEvent*>& wait_events_in,
    const std::array<size_t, 3>& global_sizes, const std::array<size_t, 3>& group_sizes,
    const std::map<const cl::Context*, ndr_division>& ndr_divisions, const bool use_cout)
{
    int err{ 0 };

    _event_exec_kernel = std::make_unique<clAppEvent>();

    std::map<map_device_info, const cl::Context*> map_device_context;
    for (auto&[ctx_dev, cldevice] : _map_devices) {
        const auto& key = std::make_pair(cldevice->device_type(), cldevice->device_id());
        map_device_context[key] = ctx_dev;
    }

    for (auto& [ctx_dev, cldevice] : _map_devices)
    {
        auto it = ndr_divisions.find(ctx_dev);
        if (it == ndr_divisions.end())continue;

        //fill wait_list with events in matching context
        std::vector<cl::Event> wait_list;
        for (auto& wait_event_in : wait_events_in)
        {
            err = wait_event_in->get_events_in_context(ctx_dev, wait_list);
            if (on_coopcl_error(err) != 0)
                return err;
        }

        const auto& offsets = ndr_divisions.at(ctx_dev).global_offset_sizes;
        const auto& global_chunk_sizes = ndr_divisions.at(ctx_dev).global_chunk_sizes;

        cl::Event wait_kernel;
        err = cldevice->enqueue_async_ndrange_block(wait_kernel, wait_list,
                                                    global_sizes, group_sizes, offsets, global_chunk_sizes, _queue_id, use_cout);

        if (err != CL_SUCCESS) return err;

        err = cldevice->flush();
        if (err != CL_SUCCESS) return err;

        err = _event_exec_kernel->register_and_create_user_events(wait_kernel, map_device_context, true);
        if (err != CL_SUCCESS) return err;

    }

    return err;
}


int clTask::async_execute(const std::vector<std::unique_ptr<clAppEvent>>& wait_events_in, 
	const std::array<size_t, 3>& global_sizes, const std::array<size_t, 3>& group_sizes, 
	const std::map<const cl::Context*, ndr_division>& ndr_divisions, const bool use_cout)
{
    std::vector<const clAppEvent*> wait_list;
    wait_list.reserve(wait_events_in.size());
    for (auto& ev_in : wait_events_in)
        wait_list.emplace_back(ev_in.get());

    return async_execute(wait_list, global_sizes,group_sizes,ndr_divisions, use_cout);
}

int clTask::wait()const 
{
    int err = 0;

    for (auto& ev : _events_to_wait) {
        err = ev->wait();
        if (err != CL_SUCCESS)
            return err;
    }

    if(_event_exec_kernel){
        err = _event_exec_kernel->wait();
        if (err != CL_SUCCESS)
            return err;
    }

    return err;
}

void clTask::wait_clear_events()
{
    auto err = wait();
    if (err != 0) throw std::runtime_error("Some error on call wait_clear_events, FIXME!!");

    if(_event_exec_kernel)
        _event_exec_kernel.release();

    _events_to_wait.clear();
}

void clTask::add_dependence(const clAppEvent& event_in)
{
    _events_to_wait.push_back(&event_in);
}

void clTask::add_dependence(const std::vector<clAppEvent>& events_in)
{
    for(auto& ev:events_in)
        _events_to_wait.push_back(&ev);

}

void clTask::add_dependence(const clTask* other_task)
{
    _dependent_tasks.push_back(other_task);
}

float clTask::duration(const float scale_factor) const
{
    float acc_duration = 0;

    if(_event_exec_kernel)
        acc_duration += _event_exec_kernel->duration(scale_factor);

    return acc_duration;

}

int clTask::set_task_output(const offload_info& offload_devices, std::uint8_t& id, std::unique_ptr<clMemory>& arg)
{
    int err = 0;

    if (_kernel_arguments[id]._ADDRESS_QUALIFIER != CL_KERNEL_ARG_ADDRESS_GLOBAL)
        return -1;

    const auto is_set = _kernel_arguments[id]._TYPE_QUALIFIER & CL_KERNEL_ARG_TYPE_CONST;
    if (is_set != CL_KERNEL_ARG_TYPE_CONST)
    {
        map_device_info target_device = { CL_DEVICE_TYPE_ALL,0 };
        if (offload_devices.empty())
        {
            return -1;
        }
        else if (offload_devices.size() == 1)
        {
            const auto& [offload_ndr, device_type_device_id] = offload_devices[0];
            //target_device = { device_type,device_id };
            target_device = device_type_device_id;
        }

        arg->set_device_with_coherent_memory(target_device);
    }

    return err;
}

int clTask::transfer_memory_async(const std::vector<const clAppEvent*>& wait_events_in,
                                  clAppEvent& wait_event_out, const offload_info& offload_devices, std::unique_ptr<clMemory>& mem)
{
    int err = 0;

    if (!mem->is_input())return err;

    //TODO: here if the input access pattern is contiguous than can transfer only chunks
    const size_t offset = 0;
    const size_t size = mem->size();
    const map_device_info source_device = mem->get_device_with_coherent_memory();

    for (auto& [ndr_chunk,device_type_device_id]: offload_devices)
    {
        //const map_device_info destination_device = { device_type,device_id };
        err = mem->copy_async(wait_events_in,wait_event_out, offset, size, source_device, device_type_device_id);
        if (err != CL_SUCCESS)return err;
    }
    return err;
}

int clTask::identify_kernel_input_output_memory(const cl::Kernel& kernel)
{

    int err = 0;
    std::string err_log;
    size_t args = kernel.getInfo<CL_KERNEL_NUM_ARGS>(&err);
    if (err != CL_SUCCESS) {
        err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        std::cerr << err_log << std::endl;
        return err;

    }

    for (size_t id = 0; id < args; id++)
    {
        const size_t aq = kernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(id, &err);
        if (err != CL_SUCCESS) {
            err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            std::cerr << err_log << std::endl;
            return err;
        }

        cl_kernel_arg_type_qualifier tq;
        err = kernel.getArgInfo(id, CL_KERNEL_ARG_TYPE_QUALIFIER, &tq);
        if (err != CL_SUCCESS) {
            err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            std::cerr << err_log << std::endl;
            return err;

        }

        std::string type_name = kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(id, &err);
        if (err != CL_SUCCESS) {
            err_log = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            std::cerr << err_log << std::endl;
            return err;
        }

        const auto pos = type_name.find('\000');
        if (pos != std::string::npos)
            type_name.replace(pos, 4, "");


        _kernel_arguments.push_back(arg_info{aq,tq,type_name});
    }

    return err;
}



