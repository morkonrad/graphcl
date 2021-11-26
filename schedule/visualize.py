import matplotlib.font_manager as font_manager
import numpy as np
import matplotlib.axes as mataxs
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

import dag as mkmd_dag


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def setup_mpl(palette):
    sns.set_context("paper")
    sns.set_palette(palette)

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white")
    """sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

    params = {
        'axes.labelsize': 9,
        'font.size': 9,
        # 'legend.fontsize': 10,
        # 'xtick.labelsize': 10,
        # 'ytick.labelsize': 10,
        'text.usetex': False,
        # 'figure.figsize': [12, 9]
    }
    mpl.rcParams.update(params)"""
    return


def set_size_one_column(fig):
    fig.set_size_inches(6, 3)
    plt.tight_layout()


def plot_speedup(palette, lables_app: [str], speedup_map: {str: [float]}, file_name: str):
    setup_mpl(palette)

    ylabel = 'Speedup'
    title = ''
    columns = []
    for key, speedup_list in speedup_map.items():
        gm = round(geo_mean(speedup_list), 2)
        speedup_list.append(gm)
        speedup_map[key] = speedup_list
        columns.append(key)

    plotdata = pd.DataFrame(speedup_map, index=lables_app)

    # set desired columns order
    # columns = ["CPU-I74930K", "GPU-GTX780Ti", "GPU-GTXTitan", "GPU-GTX1080", "GraphCL", "Oracle"]
    plotdata = plotdata[columns]
    ax = plotdata.plot(kind="bar", rot=0, width=0.65)

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.set_yticks([0, 0.25, 0.5, 0.75, 1.00, 2.0, 2.5, 3.0])
    # ax.set_yticks(np.arange(0,3.0,0.5))

    ax.set_ylim(0, 4.0)

    # set individual bar lables using above list
    patch_id = 0
    series_id = 1
    geo_mean_offset = len(lables_app)
    for patch in ax.patches:
        # print(patch)
        # get_x pulls left or right; get_height pushes up or down
        patch_id += 1
        if patch_id == geo_mean_offset * series_id:
            series_id += 1
            ax.text(patch.get_x() - 0.05,
                    patch.get_height() + 0.25,
                    str(round(patch.get_height(), 2)),
                    fontsize=7, color='magenta', rotation=90)
            # patch.set_hatch('////')

    plt.title(title)
    # plt.xlabel("Family Member")
    plt.ylabel(ylabel)

    # Show the major grid lines with dark grey lines
    plt.grid(b=True, which='major', color='#666666', linestyle='dashed')

    # Show the minor grid lines with very faint and almost transparent grey lines
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    set_size_one_column(plt.gcf())

    plt.legend(loc='upper left', ncol=2)
    if len(file_name) == 0:
        plt.show()
    else:
        print('Store figure in ' + file_name)
        fig = ax.get_figure()
        fig.savefig(file_name, dpi=500)
    return


def plot_schedule_overhead(cnt_nodes: [float], schedule_duartion: [float], file_name: str):
    apps = ('SVD', 'CLYAP', 'MEQ', 'ABE', 'GABE', 'BiCG', 'TRC')
    data1 = np.array(cnt_nodes)  # np.array([3, 4, 6, 7, 11, 13, 17])
    t = np.arange(0, len(apps), 1)
    data2 = np.array(schedule_duartion)  # np.exp(t)
    # data3 = data2 * 2

    fig, ax1 = plt.subplots()
    plt.xticks(t, apps)
    plt.grid()
    x = np.arange(len(t))  # the label locations
    width = 0.35  # the width of the bars

    ax1.set_xlabel('Applications')
    ax1.set_ylabel('# of kernels')
    # ax1.set_ylim(0, 19.0)
    ax1.bar(x, data1, width, label='graph nodes', color='gray')
    ax1.tick_params(axis='y')
    ax1.legend(loc="best")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Time (ms)')
    ax2.plot(t, data2, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=6, label='schedule time')
    # ax2.plot(t, data3, color='blue', marker='x', linestyle='dotted', linewidth=2, markersize=6, label='Platform B')
    ax2.tick_params(axis='y')
    ax2.legend(loc="upper center")
    set_size_one_column(fig)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if len(file_name) == 0:
        plt.show()
    else:
        print('Store figure in ' + file_name)
        plt.savefig(file_name, dpi=500)
    return


def plot_graph(G: nx.DiGraph, position_nodes: [], node_labels: {}, edge_labels: {}):
    nx.draw(G, pos=position_nodes, with_labels=True, labels=node_labels, arrows=False,
            edge_color='black', width=1, linewidths=1, node_size=500, node_color='pink', alpha=0.9)
    nx.draw_networkx_edge_labels(G, pos=position_nodes, edge_labels=edge_labels)


def visualize_Graph(dag_list: [], G: nx.DiGraph, title_plot: str, ax: mataxs.Axes = None):
    position_nodes = nx.nx_pydot.graphviz_layout(G, prog='dot')
    node_labels = mkmd_dag.get_labels_dict(dag_list)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        plt.subplot(ax)
        ax.set_title(title_plot)
        plot_graph(G, position_nodes, node_labels, edge_labels)
        # fig.savefig('dag.png', dpi=500)
        return

    ax.set_title(title_plot)
    plot_graph(G, position_nodes, node_labels, edge_labels)
    return


def visualize_Gantt(dag_list: [], schedule: {}, processor_names: [str], title_plot: str, ax: mataxs.Axes = None):
    # -------------------------------------------------
    # Visualize
    labeldict = mkmd_dag.get_labels_dict(dag_list)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 9))
        plt.subplot(ax)
        ax.set_title(title_plot)

        plt.title(title_plot, fontsize='small')
        showGanttChart(schedule, labeldict, processor_names, ax)

        # fig.savefig('gantt.png', dpi=500)
        plt.show()
        return

    plt.title(title_plot, fontsize='small')
    showGanttChart(schedule, labeldict, processor_names, ax)
    return


def visualize(show_gant_dag: bool, show_dag: bool, dag_list: [], schedule: {}, dag_MAT: nx.DiGraph,
              processor_names: [str],
              title_plot: str, subplot: bool = False):
    """
    Plot schedule as Gantt-chart and plot task-graph as NetworkXX directed graph

    :param show_gant_dag:
    :param show_dag:
    :param dag_list:
    :param schedule:
    :param dag_MAT:
    :param processor_names:
    :param title_plot:
    :param subplot:
    :return:
    """
    if subplot:
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        if show_gant_dag:
            visualize_Gantt(dag_list, schedule, processor_names, title_plot, ax[0])
        if show_dag:
            visualize_Graph(dag_list, dag_MAT, title_plot, ax[1])

    else:
        if show_gant_dag:
            visualize_Gantt(dag_list, schedule, processor_names, title_plot)
        if show_dag:
            visualize_Graph(dag_list, dag_MAT, title_plot)
    # fig.savefig('graphcl.png', dpi=500)
    return plt.show()


def showGanttChart(proc_schedules, labels_dict: {}, processor_names: [str], ax):
    """
        Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
        Basic implementation of Gantt chart plotting using Matplotlib
        Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/
        and adapted as necessary (i.e. removed Date logic, etc)
    """

    # fig = plt.figure(figsize=fig_size)
    # ax = fig.add_subplot(111)

    processors = list(proc_schedules.keys())

    # color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
    color_choices = ['blue', 'green']

    ilen = len(processors)
    pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)

    max_end_time = 0

    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:

            ypos = (idx * 0.5) + 0.5
            xpos = job.end - job.start
            task_duration = xpos

            ax.barh(ypos, xpos, left=job.start, height=0.3, align='center',
                    edgecolor='black', color='white', alpha=0.95)

            label = labels_dict[job.task]
            xpos = 0.5 * (job.start + job.end)
            ypos = (idx * 0.5) + 0.5
            rotate_angle = 0
            if task_duration < 3:
                rotate_angle = 90
            color_id = (job.task // 10) % len(color_choices)

            ax.text(xpos, ypos, label, color=color_choices[color_id],
                    fontweight='bold', fontsize=12, alpha=0.75, rotation=rotate_angle)

            if job.end > max_end_time:
                max_end_time = job.end

    ax.set_ylim(ymin=-0.1, ymax=ilen * 0.5 + 0.5)
    ax.set_xlim(xmin=-5, xmax=max_end_time + 1.5)
    ax.grid(color='g', linestyle=':', alpha=0.5)
    font_manager.FontProperties(size='small')

    ax.set_ylabel('Command queues', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_yticks(pos)
    ax.set_yticklabels(processor_names, fontsize=12)
    ax.axvline(x=max_end_time, label='Schedule_span at: {:.2f}"'.format(max_end_time), color='r', linestyle='--')
    ax.legend()

    '''
    locsy, labelsy = plt.yticks(pos, processor_names)
    plt.ylabel('Processor', fontsize=16)
    plt.xlabel('Time', fontsize=16)
    plt.setp(labelsy, fontsize=14)
    plt.axvline(x=max_end_time, 
                label='Schedule_span at: time = {:.2f}"'.format(max_end_time), color='r', linestyle='--')
    plt.legend()
    plt.show()
    '''
    return
