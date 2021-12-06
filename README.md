# GraphCL

What is this ? 
--------------

It's framework for execution of OpenCL-kernel graphs on multiple-devices. 

In short, the main concept of GraphCL is to enable automatic distribution and mapping of any kernel-graph. GraphCL attempts to avoid fixed-direct code that maps kernels to devices. Designer uses GraphCL-API to define only the kernel-graph without tedious and complex mapping to the execution platform. The mapping and scheduling of graph-nodes is delegated to the GraphCL-runtime. Runtime distributes and maps tasks automatically, regardless of the number, type and power of processors. 

**Main features:**

1. OpenCL-C kernel graph-API that enables to code data-flow graphs of OpenCL-kernels.(imp. inside folder driver)
2. Task-schedule module that implements: a) the graph schedule calculation b) generates the dispatch commands for GraphCL-runtime.
3. Driver supports parallel+asynchronous tasks/kernels execution and data transfers. It also enables multi-device synchronization on: multiple-CPU and/or multiple-GPU  
4. Driver enables variable partial-workload(Ndrange) execution via splitting and gathering. 
5. Support for Intel/AMD CPUs and Intel/AMD/Nvidia GPUs

What for ? 
--------------

GraphCL is designed for any application that consist of many OpenCL-C kernels. Examples are multi-kernel apps., such as linear-algebra graphs (any DNN-model), multi-kernel machine-vision algorithms ...    


Structure  
--------------
Whole project consists of separate modules. A) schedule B) driver-runtime
  
First folder includes implementation of graph-schedule. The schedule-module uses the HEFT implementation (https://github.com/mackncheesiest/heft). In short it receives as input some OpenCL kernel-graph. In the graph each node represents a kernel function. For each kernel-function there are profiled execution times to build node-weights. For the graph-edge weights the schedule-module uses profiled bandwidth for platform-specific interconnection-bus between CPUs-GPUs. Once the schedule is calculated the schedule-module analyzes the data-flow between nodes and generates graphCL-commands.  

Second folder includes implementation of GraphCL API and driver-runtime. GraphCL-API includes high-level commands that enable two asynchronous types of operations. First operation basically enqueues some OpenCL-C kernel on CPU or GPU. Second operation enables asynchronous (non-blocking call) transfers between any CPU<-->GPU and GPU<--->GPU. The whole driver uses OpenCL-runtimes from different Hardware-Vendors. Currently it is a Intel-OpenCL-Platform that supports AMD/Intel CPUs, and AMD/NVIDIA-OpenCL-Platform for AMD/NVidia GPUs. Finally, parallel-concurrent execution of kernels on different processors and data-transfers are synchronized and managed by the GraphCL-runtime. The cross-platform synchronization is implemented with OpenCL-queueus, OpenCl-events and asynchronous user-event callbacks. Additionally, the driver includes several utility functions/apps. such as: unit-tests, some benchmark and profiling code. For more details check Readme inside folders. 

Requirements ?
---------------
1. C++17 compiler
2. CMake 3.x
3. OpenCL 1.x headers and lib. with OpenCL-runtime.
4. For unit-tests CTest
5. Python 3.x with several standard packages such as matplotlib, pandas, numpy ...



References
------------

