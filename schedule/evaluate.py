import schedule as sch
import visualize as v_plt

'''def update_task_mapping(dag_nodes: [Node], schedule: {}, map_processors: {}) -> [Node]:
    if len(dag_nodes) == 0:
        print("Warning: dag_nodes is empty, FIXME!")

    if len(schedule) == 0:
        print("Warning: schedule is empty, FIXME!")

    if len(map_processors) == 0:
        print("Warning: dict_processors is empty, FIXME!")

    mapped_nodes_coarse = []
    for processor in map_processors:
        for schedule_event in schedule[processor]:
            # unpack named tuple
            node_id, start_time, end_time, processor_id = schedule_event
            # print(node_id, start_time, end_time, processor_id)
            for node in dag_nodes:
                if node.node_id() == node_id:
                    duration = end_time - start_time
                    node.set_node_executor((duration, map_processors[processor_id]))
                    mapped_nodes_coarse.append(node)
                    continue

    mapped_nodes_coarse = sorted(mapped_nodes_coarse, key=lambda node: node.node_id())
    return mapped_nodes_coarse
'''

'''
def find_critical_path_id(sched, ids_proc, work_processors):
    # find for each proc executed work
    for idp in ids_proc:

        id_proc = idp
        # clamp to max 3 processors
        if idp > 2:
            id_proc = 2

        sum_work_proc = 0
        for sch_event in sched[id_proc]:
            duration = sch_event.end - sch_event.start
            sum_work_proc = sum_work_proc + duration

        # special hack for platform with 4 proc
        if len(ids_proc) == 4:
            if idp > 2:
                work_processors.append(0)
            else:
                work_processors.append(sum_work_proc)
        else:
            work_processors.append(sum_work_proc)

    # find id of critcal path
    cp_id_out = 0
    cp_id = -1
    max_work = 0
    for work_p in work_processors:
        cp_id += 1
        if work_p > max_work:
            max_work = work_p
            cp_id_out = cp_id

    print('ids proc:', ids_proc)
    print('work_processors:', work_processors)
    print('CP ID:', cp_id_out)
    return cp_id_out


def calc_utilization(sched, cnt_proc):
    
    work_processors = []
    ids_proc = np.arange(0, cnt_proc)
    cp_id = find_critical_path_id(sched, ids_proc, work_processors)

    sum_cp = 0
    # scan critical path CP
    for sch_event in sched[cp_id]:
        # print(sch_event)
        duration = sch_event.end - sch_event.start
        sum_cp = sum_cp + duration

        # calc. overall parallel work acc_pw
    # assumed ideal parallel exec of CP on all cnt_proc
    cnt_proc = len(work_processors)
    acc_pw = cnt_proc * sum_cp
    Utilization = []
    for work_proc in work_processors:
        Utilization.append((work_proc / acc_pw) * 100)

    perfect_utilization = (1 / cnt_proc) * 100
    diff_util = 0
    for util in Utilization:
        diff_util = diff_util + (perfect_utilization - util)
    diff_util = diff_util / cnt_proc
    perfect_utilization_ratio = 100 - ((diff_util / perfect_utilization) * 100)
    print('Utilization%:', Utilization)
    print('Perfect utilization ratio%:', perfect_utilization_ratio)


def calc_SLR(sched, cp_id):
    sum_cp = 0
    # scan critical path CP
    for sch_event in sched[cp_id]:
        # print(sch_event)
        duration = sch_event.end - sch_event.start
        sum_cp = sum_cp + duration

    span = sched[cp_id][-1].end
    print('SPAN:', span)
    print('CP:', sum_cp)
    print('SLR:', span / sum_cp)

'''


def merge_two_dicts(x, y):
    z = x.copy()  # start with keys and values of x
    z.update(y)  # modifies z with keys and values of y
    return z


def calculate_evaluation(application_name: str, map_in_order_times: {int: float},
                         end_time_coarse: float, end_time_fine: float,
                         map_processors: {int: sch.Device},
                         plot_speedups: bool = False) -> ({str: float}, {str: float}, {str: float}):
    """
    This function calculates:
    a) speedup of coarse and fine schedule policies vs. quickest single-device in-order graph execution
    b) the efficiency as ratio to the theoretical maximum speedup-oracle
    c) normalizes the results to the quickest single-device in-order
    :param application_name:
    :param map_in_order_times:
    :param end_time_coarse:
    :param end_time_fine:
    :param map_processors:
    :param plot_speedups:
    :return:
    """
    values_in_order_exec = map_in_order_times.values()
    fastest_in_order_time = min(values_in_order_exec)

    speedup_coarse = round(fastest_in_order_time / end_time_coarse, 2)

    if end_time_fine == 0:
        end_time_fine = 1.0
    speedup_fine = round(fastest_in_order_time / end_time_fine, 2)

    # Nozal-Perez Max_speedup that uses processing times
    """in_order_times = []
    acc_in_order_times = 0
    for key, in_order_time in map_in_order_times.items():
        acc_in_order_times += in_order_time
        in_order_times.append(in_order_time)
    max_speedup = round(1.0/max(in_order_times)*acc_in_order_times, 2)"""

    # Method in paper: How Are We Doing? An Efficiency Measure for Shared, Heterogeneous Systems Speedup
    # this method uses processing rates-speeds
    in_order_speeds = []
    acc_in_order_speeds = 0
    for key, in_order_time in map_in_order_times.items():
        speed = round(1.0 / in_order_time, 4)
        acc_in_order_speeds += speed
        in_order_speeds.append(speed)

    # max_speedup = round(acc_in_order_speeds / max(in_order_speeds), 2)
    # The same equations but expressed diff.
    max_speedup = round(fastest_in_order_time * acc_in_order_speeds, 2)

    efficiency_c = round(100.0 - ((max_speedup - speedup_coarse) / (max_speedup / 100.0)), 2)
    efficiency_f = round(100.0 - ((max_speedup - speedup_fine) / (max_speedup / 100.0)), 2)

    if plot_speedups:
        print('=== EVALUATION ' + application_name + ' SUMMARY ======\n')
        print(map_in_order_times)
        print('In-Order estimated time fastest device: ', round(fastest_in_order_time, 2))
        print('Parallel estimated time coarse_schedule: ', round(end_time_coarse, 2))
        print('Parallel estimated time fine_schedule: ', round(end_time_fine, 2))

        if speedup_coarse > max_speedup or speedup_fine > max_speedup:
            print('Warning, logic error, speedup > max_speedup ?-> FIXME!')

        print('Speedup coarse: ', speedup_coarse)
        print('Speedup fine: ', speedup_fine)
        print('Maximum speedup: ', max_speedup)
        print('Efficiency coarse%: ', efficiency_c)
        print('Efficiency fine%: ', efficiency_f)
        print('==================================\n')

    map_speedup = {"speedup_coarse": speedup_coarse,
                   "speedup_fine": speedup_fine,
                   "max_speedup": max_speedup}

    map_efficiency = {"efficiency_coarse": efficiency_c,
                      "efficiency_fine": efficiency_f}

    min_exec_time = 1e9
    max_exec_time = 0
    # find min_max exec time
    map_in_order_exec_processors = {}
    for key, processor in map_processors.items():

        exec_time = map_in_order_times[key]
        map_in_order_exec_processors[processor.name()] = exec_time

        if exec_time < min_exec_time:
            min_exec_time = exec_time

        if exec_time > max_exec_time:
            max_exec_time = exec_time

    # normalize to fastest-device ( min exec. time)
    for key, value in map_in_order_exec_processors.items():
        map_in_order_exec_processors[key] = round(min_exec_time / value, 2)

    if plot_speedups:
        map_items = merge_two_dicts(map_in_order_exec_processors, map_speedup)
        df = v_plt.pd.DataFrame.from_dict(map_items, orient='index')
        df = df.rename(columns={0: application_name})
        df.plot.bar(grid=True, rot=10)
        v_plt.plt.show()

    return map_speedup, map_in_order_exec_processors, map_efficiency
