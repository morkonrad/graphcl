What is this ? 
--------------

The schedule module calculates the execution schedule for some input OpenCL kernel-graph. There are several test-applications consisting of multiple kerenels (see mkmd_apps.py module). In the kernel-graph each node represents a kernel function. For each kernel-function there are profiled execution times to calculate a node-weights. For the graph-edges and theit weights the schedule-module uses profiled bandwidth for platform-specific interconnection-bus between CPUs-GPUs. Once the schedule is calculated the schedule-module analyzes the data-flow between nodes and generates graphCL-commands.

Structure 
--------------
TODO

main.py
dag.py
evaluate.py
heft.py
mkmd.py
mkmd_apps.py
schedule.py
visualize.py

How to use it 
--------------
TODO


Requirements 
---------------
check Requirements.txt to find out last-working version of python modules

References 
---------------
The schedule-module uses the HEFT implementation (https://github.com/mackncheesiest/heft).
