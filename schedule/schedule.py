import copy
import os
from time import time as timer

from clyent import color

import dag
import visualize
import heft
from dag import Buffer
from dag import Device
from dag import Node
from dag import calculate_offload_profile

_DBG_ = False


def update_node_dependencies(inputs: [Buffer], predecessors_old: [Node],
                             key_items: [Node], insertion_map: {Node: Node}) -> ([Buffer], [Node]):
    inputs_new = []
    predecessors_new = []
    """
    update_node_dependencies method uses:
    - replaced_nodes_map  
    - predecessors_old
    to create a new list with new predecessors_after_replace and input buffers after nodes replace
                
    check all predecessors, if any dependent node is replaced by the sub-kernel nodes update his
    inputs and predecessors
    """
    for predecessor in predecessors_old:
        # pn = predecessor.name()
        # print(node_name, pn)
        is_replaced_by_subkernel_nodes = False

        # check if node is replaced by the sub-kernel nodes ?
        for key in key_items:
            if key == predecessor:
                is_replaced_by_subkernel_nodes = True

                # insert a new node dependence and update data-flow
                predecessors_new.append(insertion_map[key])

                for buffer in insertion_map[key].output_buffers():
                    inputs_new.append(buffer)

        if not is_replaced_by_subkernel_nodes:
            # copy old node dependence and data-flow
            predecessors_new.append(predecessor)

            for buffer in predecessor.output_buffers():
                for input in inputs:
                    if buffer.name() == input.name():
                        inputs_new.append(buffer)

    return inputs_new, predecessors_new


def try_insert_sub_kernel_nodes(dag_out: [Node], node: Node, start_end_times: (float, float), replaced_nodes: [Node],
                                replaced_nodes_map: {Node: Node}, threshold_offload: float,
                                profile_merge: (bool, [(float, Device)])) -> \
        ({Node: Node}, [Node]):
    """
    This function updates node dependencies, data-flow if predecessor was replaced, or new merge node inserted
    checks cost of sub_kernel execution if lower than original add sub-kernel and merge nodes
    It checks profiled execution time for merge kernel and transfer times to check if distribute-merge sub-kernels is Ok
    :param dag_out: task-graph with sub-kernel nodes
    :param node:
    :param start_end_times:
    :param replaced_nodes:
    :param replaced_nodes_map:
    :param threshold_offload:
    :param profile_merge:
    :return:
    """
    node_name = node.name()
    outputs = node.output_buffers()
    predecessors_old = node.predecessor_nodes()

    # update node dependencies and data-flow if predecessor was replaced,
    # or new merge node inserted
    inputs_after_replace, predecessors_after_replace = update_node_dependencies(node.input_buffers(),
                                                                                predecessors_old,
                                                                                replaced_nodes,
                                                                                replaced_nodes_map)

    # create new sub-kernel nodes
    # with updated predecessors and inputs (data-flow)
    if _DBG_:
        print('Check partition of: ', node_name)
    profiling_data = node.node_weights_V()

    # estimate the partition sizes
    sub_kernel_profile, offload_ndr = calculate_offload_profile(profiling_data, threshold_offload, start_end_times)

    # Check what access pattern has the kernel
    is_contiguous_output = node.has_contiguous_output()
    sub_kernel_nodes = []
    # scan offload devices
    device_id = 0
    for offload in offload_ndr:
        # build new outputs
        new_outputs = []
        buffer_id = 0
        # scan new buffers
        for buffer in outputs:
            buffer_id += 1
            mem_buffer_size = buffer.size()
            if is_contiguous_output:
                mem_buffer_size = buffer.size() * offload
            buffer_name = buffer.name()
            new_outputs.append(Buffer(buffer_name + "_buff_id_" + str(buffer_id) +
                                      "_dev_id_" + str(device_id), mem_buffer_size))

        str_off = "{:.2f}".format(offload_ndr[device_id])

        # re-distribute profiled execution time to control the schedule mapping
        """
        Without this step the chunks of workload have the same execution times, since they are balanced-> exe. parallel
        
        To differentiate and thus control mapping, the profiles are distributed, 
        in a way that only one device have a proper profile time and other twice the input-balanced profile time
        , see example below:
        --------
        Example:
        --------
        (Node_in balanced)                (Sub-nodes)
        Node_x{p1=10,p2=10,p3=10} ------>  Node_x1{p1=3,p2=20,p3=20}
                                           Node_x2{p1=20,p2=3,p3=20}
                                           Node_x3{p1=20,p2=20,p3=3}
        """
        copy_profile = copy.deepcopy(sub_kernel_profile)
        profiled_duration = sub_kernel_profile[device_id][0]

        profile_id = 0
        for profile in copy_profile:
            if profile_id != device_id:
                copy_profile[profile_id][0] = profiled_duration + (profiled_duration * 2)
            profile_id += 1

        sub_kernel_node = Node(inputs_after_replace, new_outputs,
                               (is_contiguous_output, copy_profile),
                               predecessors_after_replace,
                               node_name + "$_{" + str(device_id) + "}$" + "$^{" + str_off + "}$",
                               0, copy_profile[device_id], offload_ndr[device_id])
        device_id += 1
        sub_kernel_nodes.append(sub_kernel_node)

    # create merge_node
    # gather inputs_after_replace
    merge_inputs = []
    for node_sk in sub_kernel_nodes:
        for buffer in node_sk.output_buffers():
            merge_inputs.append(buffer)

    # FIXME: here is a potential BUG -> assumption MISO(multiple input single output)
    #  assumed that only a single output exist, possible MIMO merge kernel too , where each output have different size
    node_name_merge = 'M_' + node_name
    merge_out = Buffer(node_name_merge, new_outputs[0].size())
    merge_node = Node(merge_inputs, [merge_out], profile_merge,
                      sub_kernel_nodes, node_name_merge)

    node_cost = node.cost()
    costs_sub_kernels = []

    for sk in sub_kernel_nodes:
        costs_sub_kernels.append(sk.cost())

    # max because all nodes execute in parallel
    sub_kernel_exec_cost = max(costs_sub_kernels)
    sub_kernel_exec_cost += merge_node.cost()

    is_node_replaced_by_subnodes = False

    # Check cost of sub_kernel execution if lower than original add sub-kernel and merge nodes
    if sub_kernel_exec_cost < node_cost:
        if _DBG_:
            print('Partitioned node: ', node_name)
        for sk in sub_kernel_nodes:
            dag_out.append(sk)
        dag_out.append(merge_node)

        # add on the place of old node a new merge_node
        replaced_nodes_map[node] = merge_node
        is_node_replaced_by_subnodes = True
    else:
        # print('Not-partitioned node: ', node_name)
        for replaced_node in replaced_nodes:
            for predecessor in node.predecessor_nodes():
                if replaced_node == predecessor:
                    if _DBG_:
                        print("Predecessor after sub_kernel insertion changed, need update!")
                    node.replace_dependency(predecessor, replaced_nodes_map[predecessor])

        dag_out.append(node)

    return is_node_replaced_by_subnodes


def create_dag_sub_kernels(dag_coarse: [Node],
                           profile_merge: (bool, [(float, Device)]),
                           profile_host: (bool, [(float, Device)]),
                           threshold_offload: float) -> [Node]:
    """
    With respect to the graph dependencies (data-flow) partition nodes into sub-nodes
    Handle differently special graph nodes, start-end nodes
    :param dag_coarse: input task-graph with whole kernels (NDrange not divided)
    :param profile_merge:
    :param profile_host:
    :param threshold_offload:
    :return: task-graph with sub-kernels (partial-NDranges, divided)
    """
    dag_subkernel = []
    if len(dag_coarse) == 0:
        return dag_subkernel

    # this dictionary holds references to new nodes inserted in place of old nodes
    replaced_nodes_map = {}

    # 1) scan input dag_nodes
    for node in dag_coarse:

        node_name = node.name()
        replaced_nodes = list(replaced_nodes_map)

        # for each input node check if sub-kernel execution is profitable
        if node.is_entry():
            # entry node just copy
            dag_subkernel.append(node)
        elif node.is_exit():

            outputs = node.output_buffers()
            predecessors_old = node.predecessor_nodes()

            # exit node, update his dependencies and data-flow if predecessor was replaced
            inputs_after_replace, predecessors_after_replace = update_node_dependencies(node.input_buffers(),
                                                                                        predecessors_old,
                                                                                        replaced_nodes,
                                                                                        replaced_nodes_map)
            # create a new node with updated predecessors and inputs (data-flow)
            exit_node = Node(inputs_after_replace, outputs, profile_host, predecessors_after_replace,
                             node_name, 0, profile_host[1][0])

            dag_subkernel.append(exit_node)
        else:
            try_insert_sub_kernel_nodes(dag_subkernel, node, (0, 0), replaced_nodes, replaced_nodes_map,
                                        threshold_offload, profile_merge)

    # update node ids
    new_node_ids = 0
    for node in dag_subkernel:
        node.set_id(new_node_ids)
        new_node_ids += 1

    return dag_subkernel


def calculate_HEFT_schedule(dag_list: [], map_processors: {}, avoid: bool = False, print_duration: bool = True) -> (
        {int: heft.ScheduleEvent}, visualize.nx.DiGraph, float, float):
    """

    :param dag_list:
    :param map_processors:
    :param avoid:
    :param print_duration:
    :return:
    """
    # create adapter input matrices with node and edge weights to use HEFT schedule-module
    file_name_node_matrix = '/tmp/nv.csv'
    file_name_edge_matrix = '/tmp/ne.csv'
    file_name_bw_resources = '/tmp/bw.csv'
    # print('OS name:\t', os.name)
    if os.name == "nt":
        path_tmp = os.environ['TMP']
        # print(path_tmp)
        file_name_node_matrix = path_tmp + '\\nv.csv'
        file_name_edge_matrix = path_tmp + '\\ne.csv'
        file_name_bw_resources = path_tmp + '\\bw.csv'

    dag.build_csv_files(dag_list, map_processors, file_name_node_matrix, file_name_edge_matrix, file_name_bw_resources)

    # call HEFT.py
    comp_matrix_V = heft.readCsvToNumpyMatrix(file_name_node_matrix)
    comm_matrix_BW = heft.readCsvToNumpyMatrix(file_name_bw_resources)
    dag_di_graph = heft.readDagMatrix(file_name_edge_matrix)

    start = timer()
    schedule = {}
    if not avoid:
        schedule, _, _ = heft.schedule_dag(dag_di_graph, communication_matrix=comm_matrix_BW,
                                           computation_matrix=comp_matrix_V)  # , rank_metric=RankMetric.BEST)
    end = timer()
    schedule_duaration = round((end - start) * 1e3, 2)

    cnt_nodes = len(dag_list)
    if print_duration:
        print('-----')
        print('DAG nodes:', cnt_nodes)
        print(f"Calculate schedule took {schedule_duaration} msec!")

    return schedule, dag_di_graph, cnt_nodes, schedule_duaration


def check_schedule_reorder(schedule: {}, map_proc: {int, dag.Device}, dag_nodes: [dag.Node]) -> int:
    # check if any nodes are reordered if yes set new mapping (device_executor id)
    for dispatched_schedule_device_id, dispatched_tasks in schedule.items():
        for schedule_event in dispatched_tasks:
            for node in dag_nodes:
                if node.node_id() == schedule_event.task:
                    if node.executor_id() != dispatched_schedule_device_id:
                        print("Detected re-mapped node ", node.name(), " after schedule")
                        duration_task = schedule_event.end - schedule_event.start
                        node.set_node_executor((duration_task, map_proc[dispatched_schedule_device_id]))
    return 0


def create_dispatch_commands(schedule: {}, map_proc: {int, dag.Device}, dag_nodes: [dag.Node]) -> int:
    check_schedule_reorder(schedule, map_proc, dag_nodes)

    print("-----------------------------------")
    print("Dispatch order of GraphCL-commands ")
    print("-----------------------------------\n")
    cmd_id = 0
    for current_node in dag_nodes:
        for predecessor in current_node.predecessor_nodes():
            if predecessor.name() != "":
                for out_buffer in predecessor.output_buffers():
                    for in_buffer in current_node.input_buffers():
                        if in_buffer.name() == out_buffer.name():  # check if buffers are "connected"
                            cmd_txt = "Copy from: " + map_proc[predecessor.executor_id()].name() + " to: " + \
                                      map_proc[current_node.executor_id()].name() + \
                                      " buffer: " + out_buffer.name() + " wait for " + predecessor.name()
                            print(cmd_txt)
                            cmd_id += 1

        if len(current_node.predecessor_nodes()) == 0:
            cmd_txt = "Execute " + current_node.name() + " on " + \
                      map_proc[current_node.executor_id()].name() + " no wait for "
            print(cmd_txt)
            cmd_id += 1
        else:
            for predecessor in current_node.predecessor_nodes():
                if predecessor.name() != "":
                    cmd_txt = "Execute " + current_node.name() + " on " + \
                              map_proc[current_node.executor_id()].name() + " wait for " + predecessor.name()
                    print(cmd_txt)
                    cmd_id += 1

    print("------------------\n")
    print("Dispatched commands:\t", cmd_id)
    return 0


def decode_data_flow(schedule: {}, map_proc: {int, dag.Device}, dag_nodes: [dag.Node]) -> int:
    check_schedule_reorder(schedule, map_proc, dag_nodes)

    # analyze dataflow to decode data transfers into runtime-commands
    print("Dispatch order: \n")
    for dispatched_schedule_device_id, dispatched_tasks in schedule.items():
        print("---------------------------------")
        print("Commands for " + map_proc[dispatched_schedule_device_id].name())
        print("---------------------------------")
        for schedule_event in dispatched_tasks:
            # scan events dispatched to the device
            # for each event scan predecessors to know data-flow
            # if no predecessor -> nop
            # if predecessor exist, check who is the executor
            # finally generate graphCL-API commands:
            #   1)copy(from,to,what_buffer)
            #   2)Execute which kernel what device
            # ------------------------------------------------------------

            # read the task inside the dag via the task_id in schedule
            current_node = dag_nodes[schedule_event.task]

            for predecessor in current_node.predecessor_nodes():
                # DBG only
                """node_id = current_node.executor_id()
                if dispatched_schedule_device_id != node_id:
                    print("----", "Remap dag node after schedule: ", current_node.name(),
                          map_proc[dispatched_schedule_device_id].name(), "<--->",
                          map_proc[node_id].name(), "----")
                """
                if predecessor.name() != "":
                    for out_buffer in predecessor.output_buffers():
                        for in_buffer in current_node.input_buffers():
                            if in_buffer.name() == out_buffer.name():  # check if buffers are "connected"
                                cmd_txt = "Copy from: " + map_proc[predecessor.executor_id()].name() + " to: " + \
                                          map_proc[dispatched_schedule_device_id].name() + \
                                          " buffer: " + out_buffer.name()
                                print(cmd_txt)
                                print("Connection between " + current_node.name() + " and " + predecessor.name())

            cmd_txt = "Execute " + current_node.name() + " on " + map_proc[dispatched_schedule_device_id].name()
            print(cmd_txt)
    return


def store_schedule_in_file(store_graphcl_commands: bool, schedule: {},
                           file_name: str, map_proc: {int, dag.Device}, dag_nodes: [dag.Node]) -> int:
    """
    This function analyzes the data-flow of graph and generates for each processor the GraphCL-API commands
    The commands are topologically ordered to satisfy the input dependencies between graph-nodes

    :param store_graphcl_commands: flag that triggers this function
    :param schedule: schedule for input task-graph
    :param file_name: where to store
    :param map_proc: processor names and parameters
    :param dag_nodes: input task-graph
    :return:
    """
    if not store_graphcl_commands:
        return 0

    print("------------------------------------------")
    print("Store schedule in file:\t", file_name)
    print("------------------------------------------")

    # decode_data_flow(schedule, map_proc, dag_nodes)
    create_dispatch_commands(schedule, map_proc, dag_nodes)
    return 0
