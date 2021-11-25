# GraphCL

What is this ? 
--------------
It's header only library that supports collaborative CPU-GPU workload processing. It enables parallel and asynchronous tasks execution described by the task graph.

**Features:**
1. Task graph API+Runtime
2. Parallel+asynchronous tasks/kernels execution on CPU+GPU
3. Variable workload splitting, partial offload to GPU
4. Support for APUs and CPUs+dGPUs

Requierments ?
---------------
1. C++14 compiler
2. CMake 3.x
3. OpenCL 2.x headers and lib, support for CPU and GPU
3. GPU driver with OpenCL and SVM_FINE_GRAIN_BUFFER support
4. For unit-tests CTest

How to build ?
---------------
  1. git clone CoopCL /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 14 2015 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2015 is a minimal tested version. For Linux it's tested with GCC 7.0 and Clang 5.0. In general, compiler must support C++14. 

After succesfull build you can call unit tests to check if they pass:  
 1. cd /clDriver
 2. ctest 
  
How to use it ?
----------------
After successful build and tests, the CoopCL should be ready to go. 

It's header only library so you need to only link whith your app.

Check sample usage/application below.

Example:
----------------


Current state
----------------
CoopCL is still in an early stage of development. It can successfully execute many tasks with a variable offload ratio on Intel and AMD platforms, but not yet with NVIDIA GPUs. Current NVIDIA drivers support only OpenCL 1.x. 

The extension for NVIDIA Platforms and multi-GPU is in progress.

**Tested systems:**

| HW-Vendor | CPU       | GPU     | GPU-Driver     | OS    | Platform          |
| --------- | --------- | ------- | -------------- | ----- | ----------------- |
| Intel+AMD | I7-3930k  | R9-290  | 2906.10        | win64 | Desktop dCPU+dGPU |
| Intel	    | I7-660U   | HD-520  | 26.20.100.7158 | win64 | Notebook APU      |
| Intel	    | I7-8700   | UHD-630 | 26.20.100.7158 | win64 | Notebook APU      |
| AMD	    | R5-2400GE | Vega-11 | 2639.5         | win64 | Notebook APU      |
| AMD	    | R7-2700U  | Vega-10 | 2639.5         | win64 | Notebook APU      |

References
------------

