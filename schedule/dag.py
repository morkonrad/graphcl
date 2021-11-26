import pandas as pd
import numpy as np
import heft

_DBG_ = False


class Device:
    _name: str
    # in bytes/sec
    _bandwidth: float = 0
    _latency_enqueue_sec: float = 10e-6
    _id: int = 0

    def __init__(self, name: str, bandwidth: float, id: int = 0):
        self._bandwidth = bandwidth
        self._name = name
        self._id = id
        return

    def name(self) -> str:
        return self._name

    def bandwidth(self) -> float:
        return self._bandwidth

    def latency_enqueue(self) -> float:
        return self._latency_enqueue_sec

    def id(self):
        return self._id

    def info(self):
        print('Device name:\t', self.name())
        print('Device bandwidth:\t', self.bandwidth() / 1e9, ' GB/sec')
        print('Device id:\t', self.id())
        return


class Buffer:
    _name: str = ''
    _size: float = 0  # store in bytes

    def __init__(self, name: str, size: float):
        self._name = name
        self._size = size

    def name(self) -> str:
        return self._name

    def size(self) -> float:
        return self._size


class Node:
    _node_id: int = 0
    _name: str = ''

    # lists with I/O buffers
    _input_buffers: [Buffer]
    _output_buffers: [Buffer]

    # size in bytes of all I/O
    _node_input_size: int = 0
    _node_output_size: int = 0

    # profiled and inferred values
    _node_weights_V: [(float, Device)]
    _edge_weight_E: float = 0

    # dependent nodes
    _predecessor_nodes: []

    _is_entry: bool = False
    _is_exit: bool = False

    # bandwidth of device that executes the node
    _big_val: float = 100e9

    _duration_node_execution: float = 0
    _bandwidth_node_executor: float = 0
    _name_node_executor: str = ''
    _id_node_executor: int = 0
    _latency_enqueue_node_executor: float = 10e-6
    _workload_chunk_dispatched_to_execute: float = 100.0  # chunk of Ndrange dispatched for processing

    def set_device_info(self, processing_duration: float, processor: Device):

        self._duration_node_execution = processing_duration
        self._bandwidth_node_executor = processor.bandwidth()
        self._name_node_executor = processor.name()
        self._latency_enqueue_node_executor = processor.latency_enqueue()
        self._id_node_executor = processor.id()

    def set_node_executor(self, node_executor: (float, Device)):
        """
        if node_executor specified ->read his content
        else
        check all _node_weights_V (profiles) to find the minimum one
        and then set values
        :return:
        """
        profiled_duration, processor = node_executor

        if processor is None:

            min_profile = -1
            min_profile_device = []

            if len(self._node_weights_V) > 0:
                for profile in self._node_weights_V:
                    if min_profile == -1:
                        min_profile = profile[0]
                        min_profile_device.append(profile[1])
                    elif profile[0] < min_profile:
                        min_profile = profile[0]
                        min_profile_device.append(profile[1])

                if min_profile == 0:
                    # Case for nodes: Vs or Ve
                    self._bandwidth_node_executor = self._big_val
                else:
                    self.set_device_info(min_profile, min_profile_device[-1])
        else:
            self.set_device_info(profiled_duration, processor)
        return

    def calculate_edge_weight(self):
        """
        // do data-flow analysis
        -----------------------------
        1) get own input buffers
        2) check in predecessors:
            - if this predecessor includes that data
            - check what processors processed the node to find the interconnect bandwidth
            - calculate the latency (weight)
        :return:
        """
        if len(self._input_buffers) == 0:
            return

        if len(self._predecessor_nodes) == 0:
            return

        latency_sec = 0
        for predecessor in self._predecessor_nodes:
            for in_buffer in self._input_buffers:
                for out_buffer in predecessor.output_buffers():
                    if out_buffer.name() == in_buffer.name():
                        size_in_memory_bytes = out_buffer.size()

                        if predecessor.name_node_executor() == self._name_node_executor:
                            latency_sec += self._latency_enqueue_node_executor
                        else:
                            bandwidth_predecessor_bytes_sec = predecessor.bandwidth_node_executor()

                            bandwidth_interconnect_bytes_sec = min(bandwidth_predecessor_bytes_sec,
                                                                   self._bandwidth_node_executor)

                            latency_sec += round(size_in_memory_bytes / bandwidth_interconnect_bytes_sec, 4)

        weight_ms = (latency_sec + self._latency_enqueue_node_executor) * 1e3
        self._edge_weight_E = weight_ms
        return

    def __init__(self, input_buffers: [Buffer], output_buffers: [Buffer], profiles: (bool, [(float, Device)]),
                 dependent_nodes: [], name: str, node_id: int = 0, node_executor: (float, Device) = (0, None),
                 workload_chunk_dispatched: float = 100):
        """
        :param input_buffers:
        :param output_buffers:
        :param profiles:
        :param dependent_nodes:
        :param name:
        :param node_id:
        """
        self._node_id = node_id
        self._name = name
        self._input_buffers = input_buffers
        self._output_buffers = output_buffers
        self._is_contiguous_output = profiles[0]
        self._node_weights_V = profiles[1]

        self._predecessor_nodes = dependent_nodes
        self.set_node_executor(node_executor)
        self._workload_chunk_dispatched_to_execute = workload_chunk_dispatched

        if len(input_buffers) == 0:
            self._is_entry = True
            self._name_node_executor = 'Host/CPU'
        elif len(output_buffers) == 0:
            self._is_exit = True
            self.calculate_edge_weight()
            self._name_node_executor = 'Host/CPU'
        else:
            self._is_entry = False
            self._is_exit = False
            self.calculate_edge_weight()

        for buffer in self._input_buffers:
            self._node_input_size += buffer.size()

        for buffer in self._output_buffers:
            self._node_output_size += buffer.size()

        return

    def info(self):
        print('Name:\t', self._name)

        is_entry_or_exit = False

        if self._is_entry or self._is_exit:
            is_entry_or_exit = True

        print('Is entry or exit:\t', is_entry_or_exit)
        print('Input buffers:\t', self._input_buffers)
        print('Output buffers:\t', self._output_buffers)
        print('Node weights V:\t', self._node_weights_V)
        print('Edge weights E:\t', self._edge_weight_E)
        print('Node predecessors:\t', self._predecessor_nodes)
        return

    def node_id(self) -> int:
        return self._node_id

    def set_id(self, new_id: int):
        self._node_id = new_id

    def name(self) -> str:
        return self._name

    def input_buffers(self) -> [Buffer]:
        return self._input_buffers

    def output_buffers(self) -> [Buffer]:
        return self._output_buffers

    def predecessor_nodes(self) -> []:
        return self._predecessor_nodes

    def bandwidth_node_executor(self) -> float:
        return self._bandwidth_node_executor

    def name_node_executor(self) -> str:
        return self._name_node_executor

    def duration_node_execution(self) -> float:
        return self._duration_node_execution

    def node_weights_V(self) -> [(float, Device)]:
        return self._node_weights_V

    def edge_weight(self) -> float:
        return self._edge_weight_E

    def executor_id(self) -> int:
        return self._id_node_executor

    def summary(self):
        if _DBG_:
            print('---------------')
            print('Node name:\t', self._name)
            print('-')
            print('Edge weight:{:10.1f}'.format(self._edge_weight_E), 'ms')
            print('Node weight:{:10.1f}'.format(self._duration_node_execution), 'ms')
            print('Node executor:\t', self._name_node_executor)
            print('Node bandwidth:\t', self._bandwidth_node_executor * 1e-9, 'GB/sec')
        return

    def is_entry(self) -> bool:
        return self._is_entry

    def is_exit(self) -> bool:
        return self._is_exit

    def cost(self):
        return self._duration_node_execution + self._edge_weight_E

    def replace_dependency(self, predecessor_to_replace, predecessor_after_subkernel_insert):
        new_predecessors = []
        for node in self.predecessor_nodes():
            if node == predecessor_to_replace:
                new_predecessors.append(predecessor_after_subkernel_insert)
            else:
                new_predecessors.append(node)
        self._predecessor_nodes = new_predecessors
        return

    def has_contiguous_output(self) -> bool:
        return self._is_contiguous_output

    def outputs_size(self) -> int:
        return self._node_output_size

    def inputs_size(self) -> int:
        return self._node_input_size


def build_node_matrix(dag: [], map_processors: {}) -> pd.DataFrame:
    """Ex: Expected format
    TP, P_0, P_1, P_2
    T_0, 0, 0, 0
    T_1, 172, 43, 47
    T_2, 1.4, 1.6, 2.1
    T_3, 172, 43, 47
    T_4, 0, 0, 0
    """

    # create node matrix V (rows,cols)=(#nodes, #proc)

    data = []
    row_names = []
    id_node = 0

    for node in dag:
        id_pro = 0
        row = {}

        # check for the case that some processor is excluded
        if len(node.node_weights_V()) != len(map_processors):
            processors = []
            # this loop creates as many columns as processors
            for i in map_processors:
                key = 'P_' + str(i)
                row[key] = 1e6  # init with some big value
                processors.append([map_processors[i], key])

            # this loop copy proper values to target columns
            for processor in processors:
                for v in node.node_weights_V():
                    if v[1].name() == processor[0].name():
                        row[processor[1]] = v[0]
        else:
            # all processors worked, none excluded
            for v in node.node_weights_V():
                key = 'P_' + str(id_pro)
                row[key] = v[0]
                id_pro += 1

        name_in_format = node.name()  # format inside the program
        name_sch_format = "T_" + str(id_node)  # format expected by the scheduler
        # print(name_in_format + "-->" + name_sch_format)
        row_names.append(name_sch_format)
        id_node += 1
        data.append(row)

    df = pd.DataFrame.from_records(data)
    df.index = row_names

    return df


def scan_node_connections(current_node: Node, dag: []) -> {}:
    """
    scan nodes in dag and check connections with current_node
    :param current_node:
    :param dag:
    :return:
    """
    weights = {}
    # DBG
    # cn = current_node.name()
    # print('current node:\t', cn)

    # scan backwards, from Vexit to Vstart
    for node in reversed(dag):
        key = 'T' + str(node.node_id())

        # DBG
        # ccn = node.name()
        # print('scan node:\t', ccn)

        weights[key] = 0.0
        # scan node connections
        for predecessor in node.predecessor_nodes():
            if predecessor.node_id() == current_node.node_id():
                weights[key] = round(node.edge_weight(), 6)
                continue
    # DBG
    # print(weights)
    return weights


def build_edge_matrix(dag: []) -> pd.DataFrame:
    """
    Take node "v[i]" and scan forward other nodes that are connected with "v[i]"
    :param dag:
    :return:

    """
    """Ex: Expected format
    T,T_0,T_1,T_2,T_3,T_4
    T_0,0,4,0.1,0,0
    T_1,0,0,0,0.1,0
    T_2,0,0,0,3.8,0
    T_3,0,0,0,0,1.8
    T_4,0,0,0,0,0
    """

    # create node matrix E (rows,cols)=(#nodes,#nodes)
    data = []
    row_names = []
    id_node = 0
    for node in dag:
        # id_node = node.id()
        data.append(scan_node_connections(node, dag))
        row_names.append("T" + str(id_node))
        id_node += 1

    df = pd.DataFrame.from_records(data)
    # Sort column names: ascending order
    labels = sorted(df.columns, key=lambda x: float(x[1:]))
    # df = df.reindex_axis(labels, axis=1)
    df = df.reindex(columns=labels)
    df.index = row_names
    # print(df)
    return df


def build_bandwidth_matrix(dag: []) -> pd.DataFrame:
    """
    Ex: Expected format
    P, P_0, P_1, P_2
    P_0, 0, 1, 1
    P_1, 1, 0, 1
    P_2, 1, 1, 0
    """

    # create matrix BW (rows,cols)=(#proc, #proc)
    data = []
    row_names = []
    node = dag[0]
    id_row = 0
    for n in node.node_weights_V():
        row = {}
        id_col = 0
        for v in node.node_weights_V():
            key = 'P_' + str(id_col)
            # set 0 only on diagonal
            if id_col == id_row:
                row[key] = 0.0
            else:
                row[key] = 1.0  # v[1].bandwidth()
            id_col += 1
        row_names.append("P_" + str(id_row))
        id_row += 1
        data.append(row)

    df = pd.DataFrame.from_records(data)
    df.index = row_names
    return df


def build_csv_files(dag: [], map_processors: {},
                    path_file_node_csv: str, path_file_edge_csv: str, path_file_bandwidth_csv: str):
    df_bandwidth = build_bandwidth_matrix(dag)
    if path_file_bandwidth_csv != '':
        # print('Save csv file with processors bandwidth in: ', path_file_bandwidth_csv)
        df_bandwidth.to_csv(path_file_bandwidth_csv)

    df_nodes = build_node_matrix(dag, map_processors)
    if path_file_node_csv != '':
        # print('Save csv file with nodes V matrix in: ', path_file_node_csv)
        df_nodes.to_csv(path_file_node_csv)

    df_edges = build_edge_matrix(dag)
    if path_file_edge_csv != '':
        # print('Save csv file with edges E matrix in: ', path_file_edge_csv)
        df_edges.to_csv(path_file_edge_csv)

    return 0


def print_find_end_times(schedule: {int: [heft.ScheduleEvent]}, processors: [str]) -> float:
    end_times = []
    for i in range(len(schedule)):
        if len(schedule[i]) > 0:
            last_task = schedule[i][-1]
            last_task_end_time = last_task.end
            end_times.append(last_task_end_time)
            # print('Finish time: ' + processors[last_task.proc] + ':\t', last_task_end_time)
        else:
            end_times.append(0.0)

    span_time = max(end_times)
    return span_time


def get_labels_dict(dag: []):
    labels_dict = {}
    node_id = 0
    for node in dag:
        labels_dict[node_id] = node.name()
        node_id += 1
    # print(labels_dict)
    return labels_dict


def calculate_offload_profile(single_device_profile: [(float, Device)], threshold: float,
                              start_end_times: (float, float)) -> [float]:
    """
    This function calculates speeds of processors to process this task and based on this
    calculates the balanced partition of task for parallel multi-device execution
    :param single_device_profile:
    :param threshold:
    :param start_end_times:
    :return:
    """
    node_start_time, node_end_time = start_end_times
    offload_processor_ids = []
    for duration, device in single_device_profile:
        offload_processor_ids.append(device.id())

    # selected_processors = single_device_profile
    selected_processors = []
    map_profiles = {}
    id_prof = 0

    for prof in single_device_profile:
        map_profiles[id_prof] = prof
        id_prof += 1

    for coprocessor_id in offload_processor_ids:
        selected_processors.append(map_profiles[coprocessor_id])

    sum_proc_speeds = 0.0
    processor_speeds = []
    for profile in selected_processors:
        duration = profile[0]
        if duration == 0.0:
            duration = 1.0
        speed = 1 / duration
        sum_proc_speeds += speed
        processor_speeds.append(speed)

    offloads_ndr = []
    for speed in processor_speeds:
        offloads_ndr.append((speed / sum_proc_speeds))

    if _DBG_:
        print('NDR_partition %: ', np.array(offloads_ndr) * 100)

    offload_sub_kernels = []
    proc_id = 0
    for offload in offloads_ndr:
        scaled_processing_duration = selected_processors[proc_id][0] * offload
        device = selected_processors[proc_id][1]
        offload_sub_kernels.append([scaled_processing_duration, device])
        proc_id += 1

    offload_sub_kernels_filter = []
    offloads_ndr_filter = []
    proc_id = 0
    for offload in offload_sub_kernels:
        if offloads_ndr[proc_id] > threshold:
            offload_sub_kernels_filter.append(offload)
            offloads_ndr_filter.append(offloads_ndr[proc_id])
        proc_id += 1

    offload_sum = 0
    fastest_device_id = 0
    max_val = 0
    did = 0
    for offload in offloads_ndr_filter:
        if offload > max_val:
            max_val = offload
            fastest_device_id = did
        offload_sum += offload
        did += 1

    # if whole workload is not distributed ? than the fastest device gets rest
    if offload_sum < 1.0:
        rest = 1.0 - offload_sum
        offloads_ndr_filter[fastest_device_id] += rest

    return offload_sub_kernels_filter, offloads_ndr_filter


def find_span_time_device_id(schedule: {}, map_processors: {}) -> (float, int):
    # find a span_time and the associated processor_id
    """
       schedule is a dictionary with list of ScheduleEevent type
       ScheduleEvent is a namedtuple (see: python collections)
       his fields are:
       -node_id <int>
       -start_time <float>
       -end_time <float>
       -processor_id <int>
    """

    span_time = 0
    device_id_span = 0
    for processor_id in map_processors:
        for schedule_event in schedule[processor_id]:
            # unpack named tuple
            node, start_time, end_time, processor = schedule_event
            # print(node, start_time, end_time, processor)
            if end_time > span_time:
                span_time = end_time
                device_id_span = processor_id
    return span_time, device_id_span


def find_span_tasks(dag_nodes: [], schedule: {}, map_processors: {}) -> [Node, float, float]:
    if len(dag_nodes) == 0:
        print("Warning: dag_nodes is empty, FIXME!")

    if len(schedule) == 0:
        print("Warning: schedule is empty, FIXME!")

    if len(map_processors) == 0:
        print("Warning: dict_processors is empty, FIXME!")

    span_time, device_id_span = find_span_time_device_id(schedule, map_processors)

    # get list with nodes executed on device_id_span
    span_tasks = []
    for schedule_event in schedule[device_id_span]:
        node_id, start_time, end_time, processor = schedule_event
        # span_tasks.append((dag_nodes[node_id], start_time, end_time))
        for node in dag_nodes:
            if node.node_id() == node_id and node.executor_id() == processor:
                span_tasks.append((node, start_time, end_time))
                continue

    return span_tasks


def find_idle_slots(dag_nodes: [Node], schedule: {}, map_processors: {}) -> {int: [(float, float)]}:
    if len(dag_nodes) == 0:
        print("Warning: dag_nodes is empty, FIXME!")

    if len(schedule) == 0:
        print("Warning: schedule is empty, FIXME!")

    if len(map_processors) == 0:
        print("Warning: dict_processors is empty, FIXME!")

    # find span_time
    span_time = 0
    busy_slots = {}
    for processor_id in map_processors:
        busy_slots[processor_id] = []

    for processor_id in map_processors:
        for schedule_event in schedule[processor_id]:
            # unpack named tuple
            node, start_time, end_time, processor = schedule_event
            busy_slots[processor_id].append((start_time, end_time))
            if end_time > span_time:
                span_time = end_time

    idle_slots = {}
    for processor_id in map_processors:
        idle_slots[processor_id] = []

    for processor_id in map_processors:
        if len(busy_slots[processor_id]) == 0:
            idle_slots[processor_id].append((0.0, span_time))
        else:

            is_last_item = False
            start_time_idle = 0
            cnt_items = len(busy_slots[processor_id])
            item = 0

            for busy_slot in busy_slots[processor_id]:

                if item == cnt_items - 1:
                    is_last_item = True
                item += 1

                start_time_busy = busy_slot[0]
                end_time_busy = busy_slot[1]

                if start_time_busy > start_time_idle:
                    idle_slots[processor_id].append((start_time_idle, start_time_busy))
                    start_time_idle = end_time_busy
                else:  # case that two consecutive blocks have: block_a end_time is equal block_b start_time
                    start_time_idle = end_time_busy

                # case that busy_slot is the last in collection
                if is_last_item and end_time_busy < span_time:
                    idle_slots[processor_id].append((end_time_busy, span_time))
    return idle_slots


def calculate_dag_execution_times(dag: [Node], map_processors: {int, Device}) -> {int: float}:
    """
    Get communication and execution costs to properly estimate the node weights
    :param dag:
    :param map_processors:
    :return:
    """
    if len(dag) == 0:
        print('Warning, expected non emtpy list, fixme!')

    cnt_processors = len(map_processors)
    execution_times_processor = {}
    for proc_id in range(cnt_processors):
        execution_times_processor[proc_id] = 0

    size_input_memory = 0
    size_output_memory = 0

    for proc_id in range(cnt_processors):

        for node in dag:
            exec_times = node.node_weights_V()
            # accumulate node weights
            duration, device = exec_times[proc_id]
            execution_times_processor[proc_id] += duration
            if node.node_id() == 0:
                size_input_memory = node.outputs_size()  # read source node outputs

            if node.node_id() == (len(dag) - 1):
                size_output_memory = node.inputs_size()  # read exit node outputs

        execution_times_processor[proc_id] += size_input_memory / map_processors[proc_id].bandwidth()
        execution_times_processor[proc_id] += size_output_memory / map_processors[proc_id].bandwidth()

    return execution_times_processor
