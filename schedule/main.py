import mkmd
import argparse as arg_parser
import visualize
import numpy as np


def main():
    parser = arg_parser.ArgumentParser(description='Optional app description')
    app_names = mkmd.application_names.values()
    names = ''
    for app_name in app_names:
        names += app_name + ','

    parser.add_argument('--mkmd_name', type=str, help='Select one application name: ' + 'ALL,' + names)
    parser.add_argument('--partition_sub_kernel_on', dest='partition', action='store_true')
    parser.add_argument('--partition_sub_kernel_off', dest='partition', action='store_false')
    parser.set_defaults(partition=True)

    args = parser.parse_args()
    app_name = args.mkmd_name
    """
    flags to control app
    """
    use_subkernel_mkmd = args.partition
    selected_platform = mkmd.Platforms.PLATFORM_A
    matrix_width = 4096  # currently two profiled sizes: 2048, 4096
    threshold_offload = 10 / 100.0  # offload in %
    store_graphcl_commands = True
    do_overall_evaluation = False

    # ----------------------------
    evaluate_speedups = False
    show_gantt_graph = True
    show_graph_dag = True
    create_subplot = True

    if evaluate_speedups:
        show_gantt_graph = False
        show_graph_dag = False
        create_subplot = False

    if do_overall_evaluation:
        show_gantt_graph = False
        show_graph_dag = False
        create_subplot = False
        evaluate_speedups = False

    if app_name == 'ALL':

        speedups = []
        in_order_proc_executions = []
        cnt_nodes = []
        schedule_durations = []

        for key, application_name in mkmd.application_names.items():
            setup = mkmd.MKMDSetup(application_name, matrix_width, threshold_offload,
                                   selected_platform, use_subkernel_mkmd,
                                   show_gantt_graph, show_graph_dag,
                                   store_graphcl_commands, evaluate_speedups, create_subplot)

            speedup, in_order_proc_execution, nodes, schedule_duration = mkmd.call_mkmd(setup)
            speedups.append(speedup)
            in_order_proc_executions.append(in_order_proc_execution)
            cnt_nodes.append(nodes)
            schedule_durations.append(schedule_duration)

        speedups_sorted_per_application = np.array(speedups).transpose().tolist()
        # print('Speedup [coarse,fine,max) schedule\n')
        # print(speedups_sorted_per_application)

        executions_sorted_per_device = np.array(in_order_proc_executions).transpose().tolist()
        # print('In-order single processor durations\n')
        # print(executions_sorted_per_device)

        lables_app = ['SVD', 'CLYAP', 'MEQ', 'ABE', 'GABE', 'BiCG', 'TRC', 'GMEAN']

        if selected_platform == mkmd.Platforms.PLATFORM_C:

            figure_name = "speedup_119.png"
            speedup_sch_cpu = executions_sorted_per_device[0]
            speedup_sch_gpu_1080ti = executions_sorted_per_device[1]
            speedup_sch_gpu_1080 = executions_sorted_per_device[2]
            speedup_sch_gpu_titan = executions_sorted_per_device[3]
            # ----------------------------------------------------------
            speedup_sch_graphcl = speedups_sorted_per_application[1]
            speedup_sch_max = speedups_sorted_per_application[2]

            speedup_map = {
                "CPU-I9-7980XE": speedup_sch_cpu,
                "GPU-GTX-TitanX": speedup_sch_gpu_titan,
                "GPU-GTX-1080": speedup_sch_gpu_1080,
                "GPU-GTX1080-TI": speedup_sch_gpu_1080ti,
                "GraphCL": speedup_sch_graphcl,
                "Oracle": speedup_sch_max}
        elif selected_platform == mkmd.Platforms.PLATFORM_A:

            figure_name = "speedup_129.png"
            speedup_sch_cpu = executions_sorted_per_device[0]
            speedup_sch_gpu_r9290 = executions_sorted_per_device[1]
            speedup_sch_gpu_wx7100 = executions_sorted_per_device[2]
            # ----------------------------------------------------------
            speedup_sch_graphcl = speedups_sorted_per_application[1]
            speedup_sch_max = speedups_sorted_per_application[2]

            speedup_map = {
                "CPU-6430G": speedup_sch_cpu,
                "GPU-R9-290": speedup_sch_gpu_r9290,
                "GPU-WX-7100": speedup_sch_gpu_wx7100,
                "GraphCL": speedup_sch_graphcl,
                "Oracle": speedup_sch_max}
        elif selected_platform == mkmd.Platforms.PLATFORM_B:

            figure_name = "speedup_061.png"
            speedup_sch_cpu = executions_sorted_per_device[0]
            speedup_sch_gpu_1 = executions_sorted_per_device[1]
            speedup_sch_gpu_2 = executions_sorted_per_device[2]
            speedup_sch_gpu_3 = executions_sorted_per_device[3]
            # ----------------------------------------------------------
            speedup_sch_graphcl = speedups_sorted_per_application[1]
            speedup_sch_max = speedups_sorted_per_application[2]
            speedup_map = {
                "CPU i7-4930k": speedup_sch_cpu,
                "GPU-1080": speedup_sch_gpu_1,
                "GPU-TitanX": speedup_sch_gpu_2,
                "GPU-GTX780-TI": speedup_sch_gpu_3,
                "GraphCL": speedup_sch_graphcl,
                "Oracle": speedup_sch_max}

        # if last arg is non-empty store to file else  matplotlib.show is called
        visualize.plot_schedule_overhead(cnt_nodes, schedule_durations, "overhead.png")
        visualize.plot_speedup("YlGnBu", lables_app, speedup_map, figure_name)
    else:

        if app_name is not None and any(app_name in s for s in app_names):
            setup = mkmd.MKMDSetup(app_name, matrix_width, threshold_offload,
                                   selected_platform, use_subkernel_mkmd,
                                   show_gantt_graph, show_graph_dag,
                                   store_graphcl_commands, evaluate_speedups, create_subplot)
            mkmd.call_mkmd(setup)
        else:
            print("Unsupported application name, please check: ./app -h , EXIT!")
            return -1

    return 0


if __name__ == '__main__':
    main()
