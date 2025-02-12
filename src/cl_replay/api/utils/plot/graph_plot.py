import os, sys
import re
import json
import numpy as np
import math
import traceback
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path

'''
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=2)
'''

'''
python3 .../graph_plot.py path1 path2 
'''

def load_data(exp_path):
        
    metric_names = []
    metric_values = []
    # loads all npy files into a structure
    exp_path = Path(exp_path)
    if Path.is_dir(exp_path):
        metrics_path = Path(os.path.join(exp_path, 'metrics'))
        path_collection = {}
        for file in metrics_path.iterdir():
            try:
                file_split = str(file).split('/', maxsplit=-1)[-1]
                file_ending = file_split.split('_')[-3]
                print(file_split)
                print(file_ending)
                if file_ending.isdigit():
                    print(file, file_ending)
                    task_descriptor = int(file_ending)
                    if task_descriptor in path_collection:
                        path_collection[task_descriptor].extend([file])
                    else:
                        path_collection[task_descriptor] = [file]
            except Exception as ex:
                print(traceback.format_ewc())
    
        print(path_collection)
    for task in sorted(path_collection.keys()):
        print(task)
        for npy_file in path_collection[task]:
            if str(npy_file).endswith('names.npy'): metric_names.append(np.load(npy_file))
            if str(npy_file).endswith('vals.npy'): metric_values.append(np.load(npy_file))
        last_task = task
    
    
    print(metric_names)    
    print(metric_values[0].shape)
    
    return metric_names, metric_values, last_task


def plot(metric_names, metric_values, num_tasks, save_path):
    colors = [color for color in mcolors.TABLEAU_COLORS]

    fig, ax = plt.subplots()
    # use a gray background
    # ax.set_axisbelow(True)
    # draw solid white grid lines
    
    # hide top and right ticks
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()
    
    # ax.set_ylim(-300., +400.)
    # ax.yaxis.set_major_locator(plt.MaxNLocator(6)) # set number of y ticks
    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.tick_params(axis='both', which='major', labelsize=12)
    
    ax.grid(axis='y', linestyle='solid', color='w')
    ax.grid(axis='x', linestyle='solid', color='w')
    
    # plt.title('some title', pad=8)
    
    plt.xlabel(xlabel='training step', labelpad=4, fontsize=24)
    plt.ylabel(ylabel='task accuracy', labelpad=4, fontsize=24)

    ax.tick_params(axis='y', which='major', colors='white', direction='in', labelsize=12)
    for tick in ax.get_xticklabels(): tick.set_color('black')
    for tick in ax.get_yticklabels(): tick.set_color('black')
    
    num_metrics = metric_names[0].shape[0] // num_tasks
    
    
    metrics = {}    
    
    for m_n, m_v in zip(metric_names, metric_values):
        print(m_n.shape, m_v.shape)
        for m_ in range(0, num_metrics):
            per_task_m_n = m_n[m_::num_metrics]
            per_task_m_v = m_v[m_::num_metrics]
            task_id = per_task_m_n[0].split('-', maxsplit=1)[0]
            print(task_id)
        # NOTE: last iter has L2_acc, adapt if needed..
        metrics.update({ task_id : [per_task_m_n, per_task_m_v]})
    
    print(metrics)
    
    total_iters = 0
    for task_data in metrics.values():
        print(task_data[0].shape, task_data[1].shape)
        total_iters += task_data[1].shape[1]
    print(total_iters)

    x_range = np.arange(0, total_iters)
    
    concat_data = None
    task_boundaries = []
    for task_data in metrics.values():
        task_boundaries.append(task_data[1].shape[1])
        if type(concat_data) is type(None):
            concat_data = task_data[1]
        else:
            concat_data = np.concatenate([concat_data, task_data[1]], axis=1)
    
    print(concat_data.shape)
    
    for row in range(0, num_tasks):
        concat_data[row]
        # plt.plot(x_range, concat_data[row], color=colors[row], linewidth=1.0, label=f'T{row+1}')
        plt.scatter(x_range, concat_data[row], marker='^', color=colors[row], s=5, label=f'T{row+1} acc.')
    last_step = 0
    for i, boundary in enumerate(task_boundaries):
        print(i, len(task_boundaries))
        if i == len(task_boundaries)-1: continue
        if i == len(task_boundaries)-2: label = 'task boundaries'
        else: label = None
        plt.vlines(x=boundary+last_step, ymin=0, ymax=1.0, linestyle='dashdot', label=label, colors=['black'])
        last_step += boundary

    ax.set_xlim(0, 600)
    current_x_ticks = plt.xticks()[0]
    print(current_x_ticks)
    new_x_ticks = list(current_x_ticks * 100.)
    new_x_ticks[0] = None; new_x_ticks[-1] = None
    plt.xticks(ticks=current_x_ticks, labels=new_x_ticks)
    ax.set_ylim(0.0, 1.0)
    # plt.vlines(x=[200, 400, 600, 800], ymin=0, ymax=[16000, 18000, 18000, 18000], linestyle='dashed', label='task switch', colors=['grey'])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4)) 

    """
    for i in range(0, 5):
        plt.plot(task_labels[i], logs[i], alpha=1.0, ls='--', lw=2, label=f'T{i+2}', color=color[i])
        #x_1=[i, i+1]
        #y_1=logs[i][0:2]
        #plt.fill_between(x=x_1, y1=y_1, y2=-300., alpha=0.25, color=color[i])
    """
    ax.legend(frameon=True, loc='center right', ncol=1)
    plt.show()
    #fig.tight_layout()
    fig.set_figwidth(18)
    fig.set_figheight(6)
    fig.subplots_adjust(bottom=.15, left=.15)
    
    save_path = Path(save_path)
    if save_path and Path.exists(save_path):
        if save_path.is_absolute(): pass
        else: save_path = Path('/home/ak/Desktop')
    plt.savefig(f'{save_path}/graph_plot.png')


if __name__ == '__main__':
    exp_path = sys.argv[1]
    save_path = sys.argv[2]
    
    metric_names, metric_values, num_tasks = load_data(exp_path)
    
    with plt.style.context('ggplot'):  # 'bmh'
        plot(metric_names, metric_values, num_tasks, save_path)
