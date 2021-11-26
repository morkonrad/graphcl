#pragma once

#include "CL/cl.hpp"

#include <string>
#include <tuple>
#include <vector>
#include <array>


#define POOL_EVENT_TIMEOUT -999
#define COOPCL_BAD_ALLOC -998
#define COOPCL_BAD_DEVICE_INIT - 997
#define _MIN_STATS_ 0			// dump only accumulative statistics 

//------------------------------------------------------------------
#define _ELAPSED_TIMES_ 1		// dump execution and transfer times
#define _NDR_CHUNKS_ 2			// dump enqueued ndr_chunks 
#define _MEMORY_OUTPUTS_ 3		// dump content of the modified buffer
#define _THROUGHPUT_ 4			// dump estimated throughput of devices
#define _WORK_DISTRO_ 5			// dump workload distribution

//------------------------------------------------------------------
constexpr auto _SCALE_VAL_ = 2048;
#define _VERBOSE_LVL_	_ELAPSED_TIMES_

//------------------------------------------------------------------
constexpr auto _use_manual_transfers_ = false; // control if transfer memory to host in manual mode ?

//------------------------------------------------------------------
struct workload_distribution_single_device
{
    float value{ 0 };
    size_t device_type{ 0 };
    size_t device_id{ 0 };
};

struct ndr_division
{
    std::array<size_t, 3> global_chunk_sizes{ 0,0,0 };
    std::array<size_t, 3> global_offset_sizes{ 0,0,0 };
};

//device_type and device_id 
using map_device_info = std::pair<size_t, std::uint8_t>;
//offload <0.0:1.0> and device_type and device_id 
using offload_set = std::tuple<float, map_device_info>;//size_t, std::uint8_t>;
// many offload_set types
using offload_info = std::vector<offload_set>;


//------------------------------------------------------------------
///CAUTION: if you modify this than need to change also all references!!
///follow usage of _MAX_CPU_COUNT and _MAX_DEVICES_COUNT
static constexpr auto _MAX_CPU_COUNT = 1; static constexpr auto _MAX_GPU_COUNT = 4;
static constexpr auto _MAX_DEVICES_COUNT = _MAX_CPU_COUNT+ _MAX_GPU_COUNT;

int on_coopcl_error(const int err);

int on_coopcl_error(const std::string err);

//bool read_status_virtual_device();
//void set_status_virtual_device(const bool status);
