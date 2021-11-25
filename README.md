# GraphCL

What is this ? 
--------------
It's framework for execution of OpenCL-kernel graphs on multiple-devices such as mix. of CPUs and GPUs. In short, the whole concept is to provide the kernel-graph by the user without tedious distribution and mapping to the execution platform. The mapping and scheduling of graph-nodes its delegated to the GraphCL-runtime. Runtime distributes and mapps tasks automatically. It enables also parallel and asynchronous tasks execution, regardles of the number, type and power of processors.

**Features:**

1. OpenCL-C kernel graph-API that enables to code data-flow graphs of OpenCL-kernels. 
2. Runtime that implements a) the graph schedule calculation b) based on calculated schedule executes the task/kernels. 
3. Runtime supports parallel+asynchronous tasks/kernels execution on: multiple-CPU and/or multiple-GPU  
4. Runtime-driver enables variable workload (Ndrange) splitting and execution
5. Support for Intel/AMD CPUs and Intel/AMD/Nvidia GPUs

What for ? 
--------------


Structure  
--------------
Whole project consists of separate modules. A) schedule B) driver-runtime
  
First folder includes implementation of graph schedule. The schedule-module uses the HEFT implementation (https://github.com/mackncheesiest/heft). In short it recieves as input some OpenCL kernel-graph. In the graph each node represents a kernel function. For each kernel-function there are profiled execution times to build node weights. For graph-edge weights the schedule-module uses profiled bandwidth for platform-specific interconnection-bus between CPUs-GPUs. Once the schedule is calculated the schedule-module analyzes the data-flow between nodes and genereates graphCL-commands.  

Second folder includes implementation of GraphCL API+driver-runtime. GraphCL-API includes high-level commands that enable two asnychronous types of operations. First operation basicly enqueues some OpenCL-C kernel on CPU or GPU. Second operation enables asynchronous (non-blocking call) transfers between CPU<-->GPU and GPU<--->GPU. The whole driver uses OpenCL-runtimes from different Hardware-Vendors. Currently its a Intel-OpenCL-Platform that supports AMD/Intel CPUs, and AMD/NVIDIA-OpenCL-Platform for AMD/NVidia GPUs. Finally, parallel-conccuerent execution of kernels on different processors and data-transfers are synchronized and managed by the GraphCL-runtime. The cross-platform synchronization is implemented with OpenCL-queueus, OpenCl-events and asynchronous user-event callbacks. Additionally, the driver includes several utility functions/apps. such as: unti-tests, some benchmark and profiling code. 

Requierments ?
---------------
1. C++17 compiler
2. CMake 3.x
3. OpenCL 1.x headers and lib. with OpenCL-runtime.
4. For unit-tests CTest
5. Python 3.x with several standard packages such as matplotlib, pandas, numpy ...

How to build ?
---------------
(example for some Windows platform toolchain)
  1. git clone GraphCL /dst
  2. cd /dst
  3. mkdir build
  4. cd build
  5. cmake -G"Visual Studio 15 2017 Win64" .. 
  6. cmake --build . --config Release
  
For Windows, Visual Studio 2017 is a minimal tested version. For Linux it's tested with GCC 8.0 and Clang 9.0. 
In general, it needs a compiler that supports C++17. 


Current state
----------------

GraphpCL is still work in progress. But, the whole-concept was already experimentally tested and works. The driver can successfully execute different task(kernel)-graphs on Intel, NVIDIA and AMD platforms. 

**Tested systems:**

| HW-Vendor   | CPU       | GPU         | GPU-Driver     | OS    | Platform          |
| ----------- | --------- | ----------- | -------------- | ----- | ----------------- |
| Intel+Nvidia| I7-660U   | GTX-780Ti   | 26.20.100.7158 | win64 | Desktop CPU+GPU   |
| Intel+AMD   | I7-3930k  | R9-290      | 2906.10        | win64 | Desktop 2CPU+2GPU |
|             |           | WX-7100     | 2906.10        | win64 |  |
| Intel+Nvidia| I7-3930k  | GTX-1080Ti  | 2906.10        | win64 | Desktop CPU+3GPU  |
|             |           | GTX-1080    | 2906.10        | win64 |   |
|             |           | GTX-TitanX  | 2906.10        | win64 |   |

References
------------

