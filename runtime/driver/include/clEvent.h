#pragma once

#include "clCommon.h"
#include <cmath>
#include <array>
#include <string>
#include <map>

#include <future>

static constexpr int _max_events{ 10 };

class clAppEvent
{
private:    
    
    //stores std::pair<cl::Event, std::vector<cl::UserEvent>
    //
    //1)each device that do some clTask or copy memory generates some event_x
    //for each event_x created by some device in context_a, the runtime creates as many user_events as existing devices in other contexts
    //in this way finish of event_x is via callback reported to all other devices
    //
    //
    //2) Since one task can be executed in parallel by several devices , the runtime gathers pairs: event_ctx_A, list of user events in contex != ctx_A
    //
    //
	

    std::array<
		std::pair<
		std::unique_ptr<cl::Event>, 
		std::vector<std::unique_ptr<cl::UserEvent>>
	>, _max_events> _user_events;

    std::string  _task_name;
    std::future<int> _async_call_pool_event_status;
    std::uint8_t _created_event_id{ 0 };
    bool _use_async_event_callback{false};
	
	
	std::future<int> _wait_copy_async;

public:
	clAppEvent() = default;
    clAppEvent(const clAppEvent &) = delete;
    ~clAppEvent();    
    int register_and_create_user_events(std::unique_ptr<cl::Event>&, const std::map<map_device_info, const cl::Context*>& map_device_context,const bool is_enqueue_ndr_event=false);
    int get_events_in_context(const cl::Context* query_ctx, std::vector<cl::Event>& found_events)const;
    int wait()const;    
    float duration(const float scale_factor)const;
    bool is_created()const { return _created_event_id > 0 ? true : false; }
	
	void set_async_wait_copy(std::future<int>& future_handle) {
		_wait_copy_async = std::move(future_handle);
	}
    
};
