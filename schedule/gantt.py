"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/
and adapted as necessary (i.e. removed Date logic, etc)
"""

# import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np


def showGanttChart(proc_schedules, labeldict: {}, processor_names: [str], ax):
    """
        Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """

    # fig = plt.figure(figsize=fig_size)
    # ax = fig.add_subplot(111)

    processors = list(proc_schedules.keys())

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']

    ilen = len(processors)
    pos = np.arange(0.5, ilen * 0.5 + 0.5, 0.5)

    max_end_time = 0

    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            ax.barh((idx * 0.5) + 0.5, job.end - job.start, left=job.start, height=0.3, align='center',
                    edgecolor='black', color='white', alpha=0.95)
            # job.task+1,
            ax.text(0.5 * (job.start + job.end - len(str(job.task)) - 0.25),
                    (idx * 0.5) + 0.5 - 0.03125,
                    labeldict[job.task],
                    color=color_choices[(job.task // 10) % 5], fontweight='bold', fontsize=12, alpha=0.75)
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
