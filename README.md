# GraphCL

What is this ? 
--------------
It's header only library that supports collaborative CPU-GPU workload processing. 
It enables parallel and asynchronous tasks execution described by the task graph.

**Features:**

1. Task graph API and runtime that implements a) the graph schedule calculation b) based on calculated schedule executes the task/kernels. 
2. It supports parallel+asynchronous tasks/kernels execution on: multiple-CPU and/or multiple-GPU  
4. Runtime-driver enables Variable workload (Ndrange) splitting and execution
5. Support for Intel,AMD CPUs and AMD/Nvidia GPUs

Structure  
--------------
Whole project consists of separate modules. 
Graphcl:
  A) schedule
  B) driver-runtime
  
First folder schedule includes implementation of graph schedule calculation. The schedule-module uses the HEFT implementation ( ref. guthub ...). In short it recieves as input some OpenCL kernel-graph. In the graph each node represents a kernel function. For each kernel-function there are profiled execution times to build node weights. For graph-edge weights the schedule-module uses profiled bandwidth for interconnection-bus between CPUs-GPUs. Once the schedule is calculated the schedule-module analyzes the data-flow between nodes and genereates graphCL-commands.  

Second folder driver-runtime includes implementation of GraphCL. GraphCL-API includes high-level commands that enable two asnychronous types of operations. First operation basicly enqueues some OpenCL-C kernel on CPU or GPU. Second operation enables asynchronous (non-blocking call) transfers between CPU<->GPU GPU<->GPU. The whole driver uses OpenCL-API and runtime from different Hardware-Vendors. Currently its OpenCL Platform from Intel for AMD-Intel CPUs and OpenCL Platform for AMD or NVidia GPUs. Finally, parallel-conccuerent execution of kernels on different processors and data-transfers are synchronized and managed by the GraphCL-runtime. The cross-platform synchronization is implemented with OpenCL-queueus, OpenCl-events and asynchronous user-event callbacks.  
   

Requierments ?
---------------
1. C++17 compiler
2. CMake 3.x
3. OpenCL 1.x headers and lib. with OpenCL-runtime.
4. For unit-tests CTest
5. Pyton with several standard packages such as matplotlib, pandas, numpy ...

How to build ?
---------------
  1. git clone CoopCL /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 14 2015 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2015 is a minimal tested version. For Linux it's tested with GCC 7.0 and Clang 5.0. In general, compiler must support C++17. 

After succesfull build you can call unit tests to check if they pass:  
 1. cd /clDriver
 2. ctest 
  
How to use it ?
----------------
After successful build and tests, the CoopCL should be ready to go. 

It's header only library so you need to only link whith your app.

Check sample usage/application below.


Current state
----------------
GraphpCL is still work in progress. But, it the driver can already successfully execute different task(kernel)-graphs on Intel, NVIDIA and AMD platforms. 

**Tested systems:**

| HW-Vendor   | CPU       | GPU         | GPU-Driver     | OS    | Platform          |
| ----------- | --------- | ----------- | -------------- | ----- | ----------------- |
| Intel+Nvidia| I7-660U   | GTX-780Ti   | 26.20.100.7158 | win64 | Desktop CPU+GPU   |
| Intel+AMD   | I7-3930k  | R9-290      | 2906.10        | win64 | Desktop 2CPU+2GPU |
|             |           | WX-7100     | 2906.10        | win64 | Desktop 2CPU+2GPU |
| Intel+Nvidia| I7-3930k  | GTX-1080Ti  | 2906.10        | win64 | Desktop CPU+3GPU  |
|             |           | GTX-1080    | 2906.10        | win64 | Desktop CPU+3GPU  |
|             |           | GTX-TitanX  | 2906.10        | win64 | Desktop CPU+3GPU  |

References
------------

