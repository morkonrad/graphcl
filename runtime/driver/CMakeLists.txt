cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(graphcl_lib)
find_package(OpenCL REQUIRED)

add_library(${PROJECT_NAME} 
	./src/clCommon.cpp
	./src/clMemory.cpp
	./src/clTask.cpp
	./src/clDevice.cpp
	./src/clVirtualDevice.cpp
	./src/clEvent.cpp)

option(USE_NV_DRIVER "Use Nvidia specific OpenCL driver-runtime and API ?" OFF)
if (USE_NV_DRIVER)
  add_definitions(-DUSE_NV_DRIVER)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PRIVATE ./include)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCL_LIBRARIES})

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)#/WX)
else(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)#-Werror)
endif(MSVC)


if (UNIX)
    find_package(Threads REQUIRED) # add pthreads
    find_package(TBB REQUIRED) # expect this package to support parallel execution policies of c++ std_lib
    target_link_libraries(${PROJECT_NAME} PRIVATE tbb)
    target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
endif (UNIX)
