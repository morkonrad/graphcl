#pragma once
#include "clMemory.h"

class clDevice
{
    cl::Program _program_kernel_execution;
    cl::Kernel _task_exec;
    std::map<std::string,cl::Kernel> _map_kernels;

    size_t _item_size{ 0 };
    const cl::Buffer* _out_device_buffer{ nullptr };
    void* _out_host_buffer{ nullptr };


    cl::CommandQueue _cq_io;
    std::vector<cl::CommandQueue> _cqs_kernels;

    const cl::Context* _ptr_ctx{ nullptr };
    const cl::Device* _ptr_device{ nullptr };

    cl_device_type _device_type{ CL_DEVICE_TYPE_CPU };
    std::uint8_t _device_id{ 0 };
    //std::string _device_name{ "" };
	std::string _name{ "" };
    size_t _compute_units{ 0 };
    size_t _wave_size{ 0 };
    size_t _max_wgs{ 0 };
    

    int create_command_queues(
            const cl::Context& ctx,
            const cl::Device& device,
            const std::uint8_t did);

    int enqueue_async_exec(
            cl::Event& wait_event,
            const std::vector<cl::Event>& wait_list,
            const std::array<size_t, 3>& global_sizes,
            const std::array<size_t, 3>& local_sizes,
            const std::array<size_t, 3>& offsets = { 0,0,0 },
            const std::uint8_t queue_id=0)const;

    int create(const cl::Context& ctx,
               const cl::Device& device,
               const std::uint8_t did);


public:
    clDevice() = default;

    const cl::Kernel& kernel()const { return _task_exec; }
	
    auto device_ptr()const { return (*_ptr_device)(); }
    auto cq_io_ptr()const { return &_cq_io; }

    int create_async(
            const cl::Context& ctx,
            const cl::Device& device,
            const std::uint8_t did);

    clDevice(int& err,             
             const cl::Context& ctx,
             const cl::Device& device,
             const std::uint8_t did,
             const std::string& task_body,
             const std::string& task_exec_name,
             const std::string jit_flags = "");

    clDevice(int& err,             
             const cl::Context& ctx,
             const cl::Device& device,
             const std::uint8_t did);

    int build_kernel(
            const std::string& task_name,
            const std::string& task_body,
            const std::string jit_flags);

    int set_arg(const size_t id, const cl::Buffer* arg);

    int set_arg(const size_t id, const cl::LocalSpaceArg& arg)
    {
        return _task_exec.setArg(id, arg.size_,nullptr);
    }

    std::string name()const { return _name; }

    template<typename T>
    int set_arg(const size_t id, const T& arg){
        return _task_exec.setArg(id, sizeof(T), (void*)&arg);
    }

    cl_device_type device_type()const { return _device_type; }

    std::uint8_t device_id()const { return _device_id; }

    int flush()const;
    int finish()const;

    size_t compute_units()const { return _compute_units; }

    size_t wave_size()const { return _wave_size; }

    const cl::Context* ctx()const { return _ptr_ctx; }

    void select_kernel_to_execute(const std::string task_name)
    {
        _task_exec = _map_kernels.at(task_name);
    }

    int enqueue_async_ndrange_block(
            cl::Event& wait_event_kernel,
            const std::vector<cl::Event>& wait_list,
            const std::array<size_t, 3>& global_sizes,
            const std::array<size_t, 3>& local_sizes,
            const std::array<size_t, 3>& offsets,
            const std::array<size_t, 3>& global_chunk_sizes,
            const std::uint8_t queue_id =0,
            const bool use_cout = false) const;
};

