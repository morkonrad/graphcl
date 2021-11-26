import numpy as np
import mkmd_apps as apps
import schedule as sch
import evaluate as evalu

from enum import Enum
from schedule import dag

application_names = {0: 'SVD',
                     1: 'CLYAP',
                     2: 'MEQ',
                     3: 'ABE',
                     4: 'GABE',
                     5: 'BiCG',
                     6: 'TRC'}


class Platforms(Enum):
    PLATFORM_A = 129  # obj_129
    PLATFORM_B = 61  # obj_061
    PLATFORM_C = 119  # obj_119


class MKMDSetup:
    platform: Platforms = Platforms.PLATFORM_A
    matrix_width: int = 0
    matrix_height: matrix_width
    application_name: str
    use_subkernel_mkmd: bool = True
    threshold_offload: float = 1.0
    show_gant_dag: bool = False
    show_dag: bool = False
    store_graphcl_commands: bool = False
    show_speedups: bool = False
    create_subplot: bool = False

    def __init__(self, mkmd_app_name: str,
                 matrix_width: int,
                 offload_threshold: float = 1.0,
                 selected_platform: Platforms = Platforms.PLATFORM_A,
                 use_subkernel_schedule: bool = False,
                 show_gant_dag: bool = False,
                 show_dag: bool = False,
                 store_graphcl_commands: bool = False,
                 show_speedups: bool = False,
                 create_subplot: bool = False):
        self.platform = selected_platform
        self.matrix_width = matrix_width
        self.matrix_height = self.matrix_width
        self.application_name = mkmd_app_name
        self.use_subkernel_mkmd = use_subkernel_schedule
        self.threshold_offload = offload_threshold
        self.show_gant_dag = show_gant_dag
        self.show_dag = show_dag
        self.store_graphcl_commands = store_graphcl_commands
        self.show_speedups = show_speedups
        self.create_subplot = create_subplot
        return


def setup_workload_profiles(matrix_width: float, platform: Platforms):
    """
    This function set profiled execution times in tuple objects
    :param matrix_width:
    :param platform:
    :return:
    """
    MM_access = False
    MT_access = False
    start_exit_node_time = 10  # this is dummy value to force execution of start_exit node on CPU

    # select platform
    if platform == Platforms.PLATFORM_A:
        cpu = dag.Device('CPU 6134G', 100e9, 0)
        gpu1 = dag.Device('GPU WX-7100', 12e9, 1)  # PCIE-x16 measured 12GB/sec, (setup driver-dependent) of 16GB/sec
        gpu2 = dag.Device('GPU R9-290', 12e9, 2)  # PCIE-x16

        map_processors = {0: cpu, 1: gpu1, 2: gpu2}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name()]
        profile_host = (True, [(0, cpu), (start_exit_node_time, gpu1), (start_exit_node_time, gpu2)])

        if matrix_width == 2048:  # MM 15,18
            profile_MM = (MM_access, [(122, cpu), (15, gpu1), (18, gpu2)])
            profile_MT = (MT_access, [(2.6, cpu), (1.5, gpu1), (2.5, gpu2)])

            profile_ME = (True, [(1.53, cpu), (0.5, gpu1), (0.6, gpu2)])  # matrix merge
            profile_MADD = (True, [(1.4, cpu), (1.5, gpu1), (1.6, gpu2)])
            profile_vec_vec = (True, [(0.13, cpu), (0.21, gpu1), (0.23, gpu2)])
            profile_mat_vec = (True, [(0.53, cpu), (0.65, gpu1), (0.67, gpu2)])

        elif matrix_width == 4096:
            profile_MM = (MM_access, [(1996, cpu), (126, gpu1), (156, gpu2)])
            profile_MT = (MT_access, [(10.6, cpu), (25.3, gpu1), (15.1, gpu2)])

            profile_ME = (True, [(7.5, cpu), (1.47, gpu1), (1.16, gpu2)])  # matrix merge
            profile_MADD = (True, [(15.4, cpu), (4.0, gpu1), (3.6, gpu2)])
            profile_vec_vec = (True, [(0.13, cpu), (0.21, gpu1), (0.23, gpu2)])
            profile_mat_vec = (True, [(4.7, cpu), (2.45, gpu1), (2.27, gpu2)])

    if platform == Platforms.PLATFORM_B:
        cpu = dag.Device('CPU i7-4930k', 100e9, 0)
        gpu2 = dag.Device('GPU 1080', 12e9, 1)  # PCIE-x16
        gpu3 = dag.Device('GPU TitanX', 12e9, 2)  # PCIE-x16
        gpu1 = dag.Device('GPU GTX780TI', 6e9, 3)  # PCIE-x8

        map_processors = {0: cpu, 1: gpu1, 2: gpu2, 3: gpu3}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name(), gpu3.name()]

        profile_host = (True, [(0, cpu),
                               (start_exit_node_time, gpu1),
                               (start_exit_node_time, gpu2),
                               (start_exit_node_time, gpu3)])

        if matrix_width == 2048:

            profile_MM = (MM_access, [(590, cpu), (51.5, gpu1), (19, gpu2), (23.7, gpu3)])
            profile_MT = (MT_access, [(3.2, cpu), (0.18, gpu1), (0.14, gpu2), (0.14, gpu3)])

            profile_MADD = (True, [(0.27, cpu), (0.28, gpu1), (0.95, gpu2), (0.9, gpu3)])
            profile_ME = (True, [(1.7, cpu), (0.2, gpu1), (0.21, gpu2), (0.19, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(4.9, cpu), (0.13, gpu1), (0.09, gpu2), (0.12, gpu3)])
            profile_vec_vec = (True, [(0.3, cpu), (0.1, gpu1), (0.05, gpu2), (0.08, gpu3)])

        elif matrix_width == 4096:

            profile_MM = (MM_access, [(4986, cpu), (380, gpu1), (139, gpu2), (187, gpu3)])
            profile_MT = (MT_access, [(11.8, cpu), (0.71, gpu1), (0.59, gpu2), (0.58, gpu3)])

            profile_MADD = (True, [(16.3, cpu), (1.8, gpu1), (3.6, gpu2), (3.5, gpu3)])
            profile_ME = (True, [(6.5, cpu), (0.79, gpu1), (0.83, gpu2), (0.78, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(19.5, cpu), (0.35, gpu1), (0.28, gpu2), (0.34, gpu3)])
            profile_vec_vec = (True, [(1.1, cpu), (0.23, gpu1), (0.097, gpu2), (0.15, gpu3)])

    if platform == Platforms.PLATFORM_C:
        cpu = dag.Device('CPU i9-7980XE', 100e9, 0)
        gpu1 = dag.Device('GPU 1080TI', 12e9, 1)  # PCIE-x16
        gpu2 = dag.Device('GPU 1080', 6e9, 2)  # PCIE-x8
        gpu3 = dag.Device('GPU TitanX', 6e9, 3)  # PCIE-x8
        map_processors = {0: cpu, 1: gpu1, 2: gpu2, 3: gpu3}
        processor_names = [cpu.name(), gpu1.name(), gpu2.name(), gpu3.name()]

        profile_host = (True, [(0, cpu),
                               (start_exit_node_time, gpu1),
                               (start_exit_node_time, gpu2),
                               (start_exit_node_time, gpu3)])

        if matrix_width == 2048:
            """
            --------------------------------------------------------------------
            MM 2048x2048
                        :	time[ms]:	POWER[Watt]
            CPU		    :	110		: 	130
            GTX-1080TI	:	13.5	:	200
            GTX-1080	:	19		:	151
            GTX-TITANX	:	23.7	:	210

            --------------------------------------------------------------------
            MM 2048x2048
                        :	time[ms]:	POWER[Watt]
            CPU		    :	0.7		: 	120
            GTX-1080TI	:	0.096	:	176
            GTX-1080	:	0.14	:	118
            GTX-TITANX	:	0.14	:	178
            """

            profile_MM = (MM_access, [(110, cpu), (13.5, gpu1), (19, gpu2), (23.7, gpu3)])
            profile_MT = (MT_access, [(0.7, cpu), (0.09, gpu1), (0.14, gpu2), (0.14, gpu3)])

            profile_MADD = (True, [(1.6, cpu), (1.1, gpu1), (0.95, gpu2), (0.9, gpu3)])
            profile_ME = (True, [(0.5, cpu), (0.14, gpu1), (0.21, gpu2), (0.19, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(0.74, cpu), (0.08, gpu1), (0.09, gpu2), (0.12, gpu3)])
            profile_vec_vec = (True, [(0.18, cpu), (0.05, gpu1), (0.05, gpu2), (0.08, gpu3)])

        elif matrix_width == 4096:
            """
            --------------------------------------------------------------------
            MM 4096x4096
                        :	time[ms]:	POWER[Watt]
            CPU		    :	1495	: 	130
            GTX-1080TI	:	100		:	230
            GTX-1080	:	139		:	160
            GTX-TITANX	:	187		:	199

            --------------------------------------------------------------------
            MT 4096x4096
                        :	time[ms]:	POWER[Watt]
            CPU		    :	4.8 	: 	141
            GTX-1080TI	:	0.39	:	78
            GTX-1080	:	0.59	:	123
            GTX-TITANX	:	0.58	:	184
            """

            profile_MM = (MM_access, [(1495, cpu), (100, gpu1), (139, gpu2), (187, gpu3)])
            profile_MT = (MT_access, [(4.8, cpu), (0.39, gpu1), (0.59, gpu2), (0.58, gpu3)])

            profile_MADD = (True, [(8.3, cpu), (3.8, gpu1), (3.6, gpu2), (3.5, gpu3)])
            profile_ME = (True, [(2.7, cpu), (0.56, gpu1), (0.83, gpu2), (0.78, gpu3)])  # matrix merge

            profile_mat_vec = (True, [(4.74, cpu), (0.21, gpu1), (0.28, gpu2), (0.34, gpu3)])
            profile_vec_vec = (True, [(0.32, cpu), (0.1, gpu1), (0.097, gpu2), (0.15, gpu3)])

    return profile_ME, profile_host, profile_MM, profile_MT, \
           profile_mat_vec, profile_vec_vec, profile_MADD, map_processors, processor_names


def call_mkmd(setup: MKMDSetup) -> (np.ndarray, np.ndarray, float, float):
    """
    this function calculates:
    a) set-uses profiled execution times
    b) calls mkmd_application to build task graph
    c) calculate HEFT schedule
    d)
    :param setup:
    :return:
    """
    # Platform specific params:
    # input matrix size (WxH) or vector (H,1)
    matrix_width = setup.matrix_width
    matrix_height = setup.matrix_height
    matrix_size = matrix_width * matrix_height * 4  # size in bytes
    vector_size = matrix_height * 4  # size in bytes

    # get profiled execution and transfer times
    profile_ME, profile_host, profile_MM, profile_MT, \
    profile_mat_vec, profile_vec_vec, profile_MADD, \
    map_processors, processor_names = setup_workload_profiles(matrix_width, setup.platform)

    if setup.application_name == "SVD":
        dag_list, title_plot = apps.calculate_dag_SVD(matrix_size, profile_host, profile_MM, profile_MT)
    elif setup.application_name == "CLYAP":
        dag_list, title_plot = apps.calculate_dag_CLYAP(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "MEQ":
        dag_list, title_plot = apps.calculate_dag_MEQ(matrix_size, profile_host, profile_MM, profile_MT)
    elif setup.application_name == "ABE":
        dag_list, title_plot = apps.calculate_dag_ABE(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "GABE":
        dag_list, title_plot = apps.calculate_dag_GABE(matrix_size, profile_host, profile_MM, profile_MT, profile_MADD)
    elif setup.application_name == "BiCG":
        dag_list, title_plot = apps.calculate_dag_BICG(matrix_size, vector_size, profile_host, profile_mat_vec,
                                                       profile_vec_vec, profile_MADD)
    elif setup.application_name == "TRC":
        dag_list, title_plot = apps.calculate_dag_TRC(matrix_size, profile_host, profile_MM, profile_MADD)

    map_in_order_times = dag.calculate_dag_execution_times(dag_list, map_processors)

    # Calculate HEFT coarse_schedule
    schedule, dag_di_graph, cnt_nodes, schedule_duration = \
        sch.calculate_HEFT_schedule(dag_list, map_processors, False, False)

    end_time_coarse = dag.print_find_end_times(schedule, processor_names)

    sch.store_schedule_in_file(setup.store_graphcl_commands,
                               schedule, "coarse_schedule_" + setup.application_name + ".csv", map_processors, dag_list)

    sch.visualize.visualize(setup.show_gant_dag, setup.show_dag,
                            dag_list, schedule, dag_di_graph, processor_names, title_plot, setup.create_subplot)

    end_time_fine = 0
    if setup.use_subkernel_mkmd is True:

        # build sub-kernel graph from coarse HEFT schedule
        dag_list_subkernels = sch.create_dag_sub_kernels(dag_list, profile_ME, profile_host, setup.threshold_offload)

        if len(dag_list_subkernels) == len(dag_list):
            print("----------------------", setup.application_name, "----------------------")
            print("!!! Sub kernel execution not profitable !!!")
            print("--------------------------------------------")
        else:
            # calculate HEFT schedule for graph with sub-kernels
            schedule, dag_di_graph, cnt_nodes, schedule_duration = \
                sch.calculate_HEFT_schedule(dag_list_subkernels, map_processors, False, False)
            end_time_fine = dag.print_find_end_times(schedule, processor_names)

            # analyze graph and generate GraphCL-commands
            sch.store_schedule_in_file(setup.store_graphcl_commands,
                                       schedule, "fine_schedule_" + setup.application_name + ".csv",
                                       map_processors, dag_list_subkernels)
            # plot schedule, task-graph
            sch.visualize.visualize(setup.show_gant_dag, setup.show_dag,
                                    dag_list_subkernels, schedule, dag_di_graph, processor_names,
                                    title_plot, setup.create_subplot)

    # evaluate schedule and task-graph
    map_speedup, map_in_order_exec_processors, map_efficiency = evalu.calculate_evaluation(setup.application_name,
                                                                                           map_in_order_times,
                                                                                           end_time_coarse,
                                                                                           end_time_fine,
                                                                                           map_processors,
                                                                                           setup.show_speedups)
    speedups = np.fromiter(map_speedup.values(), dtype=float)
    in_order_exe = np.fromiter(map_in_order_exec_processors.values(), dtype=float)

    return speedups, in_order_exe, cnt_nodes, schedule_duration
