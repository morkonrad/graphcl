#include "clEvent.h"

#include <iostream>
#include <algorithm>
#include <stdexcept>

static void preciseSleep(double seconds) {
    /*
	* Check this blog: https://blat-blatnik.github.io/computerBear/making-accurate-sleep-function/
	* to see code source and to understand why the std::this_thread::sleep_for is badly unaccurate
    */
    using namespace std;
    using namespace chrono;

    static double estimate = 5e-3;
    static double mean = 5e-3;
    static double m2 = 0;
    static int64_t count = 1;

    while (seconds > estimate) {
        auto start = high_resolution_clock::now();
        this_thread::sleep_for(milliseconds(1));
        auto end = high_resolution_clock::now();

        double observed = (end - start).count() / 1e9;
        seconds -= observed;

        ++count;
        double delta = observed - mean;
        mean += delta / count;
        m2 += delta * (observed - mean);
        double stddev = sqrt(m2 / (count - 1));
        estimate = mean + stddev;
    }

    // spin lock
    auto start = high_resolution_clock::now();
    while ((high_resolution_clock::now() - start).count() / 1e9 < seconds);
}

static std::mutex chrono_mutex;  // chrono _start_calback guard
static std::chrono::steady_clock::time_point _start_calback;
static std::chrono::steady_clock::time_point _end_calback;
static std::mutex cout_mutex;  // cout guard

//constexpr auto _use_opencl_callback_ = false; // enables callbacks with OpenCL-driver or own impl.
constexpr auto _debug_callback_ = false; 

static int register_end_of_event(const size_t pool_id,const float sleep_duration_sec, const cl_event event_in)
{
    int err = 0;
    if (_debug_callback_)
    {
        _end_calback = std::chrono::steady_clock::now();
        const std::chrono::duration<double> host_callback_duration = _end_calback - _start_calback;

        cl_ulong ev_duartion_ns = 0;
        cl_ulong start = 0, end = 0;
        err = clGetEventProfilingInfo(event_in, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        if (on_coopcl_error(err) != 0)
            return err;

        err = clGetEventProfilingInfo(event_in, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        if (on_coopcl_error(err) != 0)
            return err;

        ev_duartion_ns = (cl_ulong)(end - start);
        const auto pool_user_event_ready_sec = ((float)pool_id * sleep_duration_sec);
        const auto event_duration_sec = (ev_duartion_ns * 1e-9f);
        {
            const std::lock_guard<std::mutex> lock(cout_mutex);
            /*std::cout << " user_event duration:\t" << pool_user_event_ready_sec << " sec \n";
			std::cout << " ocl_event duration:\t" << event_duration_sec << " sec \n";
            std::cout << " callback duration:\t" << host_callback_duration.count() << " sec \n";*/
            std::cout << " callback delay:\t" << std::abs(host_callback_duration.count() - event_duration_sec) * 1e3f << " msec" << std::endl;
        }
    }
    return err;
}

static std::future<int> start_user_events_callback(const cl_event event_in, std::vector<cl::UserEvent>& user_events)
{	
    //std::cout << "Pool event status ... \n" << std::flush;
    //auto async_call = std::async([event_in,&user_events]
    return std::async([event_in, &user_events]
                      {
                          // pool event_in status
                          // as soon as detected that event_in completed set all user_events as completed too.

                          const auto sleep_duration_sec = 1e-4f; //0.1ms
                          const std::uint64_t timeout_pool = static_cast<std::uint64_t>(5.0f/sleep_duration_sec);//5 sec
                          std::uint64_t pool_id = 0;
                          cl_int status = 0;
                          int err = 0;

                          while (pool_id++ <= timeout_pool)
                          {
                              preciseSleep(sleep_duration_sec);

                              //pool event status
                              err = clGetEventInfo(event_in, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, nullptr);
                              if (on_coopcl_error(err) != 0)
                                  return err;

                              if (status == CL_COMPLETE)
                              {
                                  for (auto& user_event : user_events)
                                  {
                                      err = user_event.setStatus(CL_COMPLETE);
                                      if (on_coopcl_error(err) != 0)
                                          return err;
                                  }

                                  register_end_of_event(pool_id, sleep_duration_sec, event_in);
                                  return err;
                              }
                          }
                          return POOL_EVENT_TIMEOUT;
                      });

    /*auto err = async_call.get();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	std::cout << "Waited with pooling " << elapsed.count() << " ms\n";
    */
    //return async_call;
}

static int set_user_events_callback(cl_event event_in, std::vector<cl::UserEvent>& user_events)
{
    return clSetEventCallback(event_in, CL_COMPLETE, [](cl_event ev, cl_int status, void* user_data)
        {
            auto user_events = static_cast<std::vector<cl::UserEvent>*>(user_data);
            if (user_events == nullptr)return;

            for (auto& user_event : *user_events)
            {
                auto err = user_event.setStatus(status);
                if (on_coopcl_error(err) != 0)return;
            }

            register_end_of_event(0, 0, ev);
            return;
        }, &user_events);
}

/**
 * @brief clEvent::register_and_create_user_events: Copy the input event and create user_events in other context than the input event
 * @param event
 * @param map_device_context
 * @return
 */
int clAppEvent::register_and_create_user_events(const cl::Event& event, const std::map<map_device_info, const cl::Context*>& map_device_context, const bool is_enqueue_ndr_event)
{
    //copy the input event and create user_events in other context than the input event
    int err = 0;

    if (event() == nullptr)return CL_INVALID_EVENT;

    auto ev_ctx = event.getInfo<CL_EVENT_CONTEXT>(&err);
    if (on_coopcl_error(err) != CL_SUCCESS)
        return err;

    const auto ptr_ctx_current = ev_ctx();
    auto& [event_in, user_events] = _user_events[_created_event_id++];
    event_in = event; //copy this event;
    
    /// Cross-context (CPU-GPU) communication with user_event callbacks is possible, it works! 
    ///  it generates communication overhead, that depends on GPU_driver
    ///  ----
    ///  For exmaple: WIN64_NV_driver is bit-slower than Linux64_NV_driver
    ///  NV-driver has strange bug realted to the event_callback of write buffer command, ndr enqueue works fine!
    ///  ----
    ///  AMD_WIN_LINUX_driver still have to benchmark
    for (auto& [deviece_type_id, ctx_dev_other] : map_device_context)
    {
        const auto ptr_ctx_other = (*ctx_dev_other)();

        if (ptr_ctx_other != ptr_ctx_current)
        {
            user_events.push_back(cl::UserEvent(*ctx_dev_other, &err));
            if (on_coopcl_error(err) != 0)
                return err;
        }
    }

    if(_debug_callback_)
    {
        const std::lock_guard<std::mutex> lock(chrono_mutex);
        _start_calback = std::chrono::steady_clock::now();
    }

    _use_async_event_callback = false;
    if (is_enqueue_ndr_event)
        err = set_user_events_callback(event_in(), user_events);
    else
    {
        _async_call_pool_event_status = start_user_events_callback(event_in(), user_events);
        _use_async_event_callback = true;
    }

    return err;
}

clAppEvent::~clAppEvent()
{
    int err = 0;
    if (_use_async_event_callback)
    {
        if (_async_call_pool_event_status.valid())
        {
            err = _async_call_pool_event_status.get();
            if (on_coopcl_error(err) != CL_SUCCESS)
                std::cerr << ("On destroy clEvent: logic-usage fatal error, FIXME !!!") << std::endl;
        }
    }
    err = wait();
    if (on_coopcl_error(err) != CL_SUCCESS)
        std::cerr << ("On destroy clEvent: logic-usage fatal error, FIXME !!!") << std::endl;
}

int clAppEvent::get_events_in_context(const cl::Context* query_ctx, std::vector<cl::Event>& found_events) const
{
    int err = 0;
    for (auto id=0;id< _created_event_id;id++)
    {
        auto& [event, user_events] = _user_events[id];
        auto ev_ctx = event.getInfo<CL_EVENT_CONTEXT>(&err);
        if (on_coopcl_error(err) != CL_SUCCESS)
            return err;

        if (ev_ctx() == (*query_ctx)())
            found_events.push_back(event);

        for (auto& uev : user_events)
        {
            auto uev_ctx = uev.getInfo<CL_EVENT_CONTEXT>(&err);
            if (on_coopcl_error(err) != CL_SUCCESS)
                return err;
            if (uev_ctx() == (*query_ctx)())
                found_events.push_back(uev);
        }
    }
    return err;
}

int clAppEvent::wait() const
{
    int err = 0;
    for (auto id = 0; id < _created_event_id; id++)
    {
        auto& [event, user_events] = _user_events[id];

        if (event() != nullptr)
        {
            err = event.wait();
            if (on_coopcl_error(err) != CL_SUCCESS)
                return err;
        }
        for (auto& uev : user_events)
        {
            if (uev() != nullptr) {
                err = uev.wait();
                if (on_coopcl_error(err) != CL_SUCCESS)
                    return err;
            }
        }
    }
    return err;
}

float clAppEvent::duration(const float scale_factor) const
{
    std::vector<cl_ulong> parallel_durations;

    for (auto id = 0; id < _created_event_id; id++)
    {
        auto& [event, user_events] = _user_events[id];

        auto err = wait();
        if (err != 0) return 0.0f;

        cl_ulong ev_duartion = 0;     

        cl_ulong start = 0, end = 0;
        err = clGetEventProfilingInfo(event(), CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        if (on_coopcl_error(err) != 0)
            return 0.0f;

        err = clGetEventProfilingInfo(event(), CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        if (on_coopcl_error(err) != 0)
            return 0.0f;

        ev_duartion = (cl_ulong)(end - start);
        parallel_durations.push_back(ev_duartion);
    }

    if (parallel_durations.empty())
        return  0.0f;

    const auto& [min, max] = std::minmax_element(parallel_durations.begin(), parallel_durations.end());
    //std::cout<<"min parallel duration:\t"<<static_cast<float>(*max) * scale_factor<<std::endl;
    return (static_cast<float>(*max) * scale_factor);
}
