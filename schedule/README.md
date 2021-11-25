What is this ? 
--------------

It's framework for execution of OpenCL-kernel graphs on multiple-devices. 

In short, the main concept of GraphCL is the automatic distribution and mapping of any kernel-graph, no fixed-direct code that maps kernels to devices. Designer uses GraphCL-API to define only the kernel-graph without tedious and complex mapping to the execution platform. The mapping and scheduling of graph-nodes is delegated to the GraphCL-runtime. Runtime distributes and maps tasks automatically, regardless of the number, type and power of processors. 

What for ? 
--------------

GraphCL is designed for any application that consist of many OpenCL-C kernels. Examples are multi-kernel apps., such as linear-algebra graphs (any DNN-model), multi-kernel machine-vision algorithms ...    


Structure  
--------------
Whole project consists of separate modules. A) schedule B) driver-runtime
  
First folder includes implementation of graph-schedule. The schedule-module uses the HEFT implementation (https://github.com/mackncheesiest/heft). In short it receives as input some OpenCL kernel-graph. In the graph each node represents a kernel function. For each kernel-function there are profiled execution times to build node-weights. For the graph-edge weights the schedule-module uses profiled bandwidth for platform-specific interconnection-bus between CPUs-GPUs. Once the schedule is calculated the schedule-module analyzes the data-flow between nodes and generates graphCL-commands.  

Second folder includes implementation of GraphCL API+driver-runtime. GraphCL-API includes high-level commands that enable two asynchronous types of operations. First operation basically enqueues some OpenCL-C kernel on CPU or GPU. Second operation enables asynchronous (non-blocking call) transfers between any CPU<-->GPU and GPU<--->GPU. The whole driver uses OpenCL-runtimes from different Hardware-Vendors. Currently it is a Intel-OpenCL-Platform that supports AMD/Intel CPUs, and AMD/NVIDIA-OpenCL-Platform for AMD/NVidia GPUs. Finally, parallel-concurrent execution of kernels on different processors and data-transfers are synchronized and managed by the GraphCL-runtime. The cross-platform synchronization is implemented with OpenCL-queueus, OpenCl-events and asynchronous user-event callbacks. Additionally, the driver includes several utility functions/apps. such as: unit-tests, some benchmark and profiling code. For more details check Readme inside folders. 

Requirements ?
---------------
check Requirements.txt to find out last-working version of python modules
