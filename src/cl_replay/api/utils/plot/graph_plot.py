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
                # print(file_split)
                file_ending = file_split.split('_')[-3]
                # print(file_ending)
                if file_ending.isdigit():
                    # print(file, file_ending)
                    task_descriptor = int(file_ending)
                    if task_descriptor in path_collection:
                        path_collection[task_descriptor].extend([file])
                    else:
                        path_collection[task_descriptor] = [file]
            except Exception as ex:
                print(traceback.format_exc())
        # print(path_collection)
    for task in sorted(path_collection.keys()):
        for npy_file in path_collection[task]:
            if str(npy_file).endswith('names.npy'): metric_names.append(np.load(npy_file))
            if str(npy_file).endswith('vals.npy'): metric_values.append(np.load(npy_file))
        last_task = task
    # print(metric_names)    
    # print(metric_values[0].shape)
    
    return metric_names, metric_values, last_task


def plot(metric_names, metric_values, num_tasks, save_path, plot_metric):
    # colors = [color for color in mcolors.TABLEAU_COLORS]
    colors = ['purple','blue','red','purple','blue','red']
    
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.grid(axis='y', linestyle='solid', color='w')
    ax.grid(axis='x', linestyle='solid', color='w')
    
    # plt.title('some title', pad=8)
    
    plt.xlabel(xlabel='training step', labelpad=6, fontsize=20)
    plt.ylabel(ylabel='task accuracy', labelpad=6, fontsize=20)

    ax.tick_params(axis='y', which='major', colors='white', direction='in', labelsize=16)
    for tick in ax.get_xticklabels(): tick.set_color('black')
    for tick in ax.get_yticklabels(): tick.set_color('black')
    
    metrics = {}    
    for m_n, m_v in zip(metric_names, metric_values):
        task_id = m_n[0].split('-', maxsplit=1)[0]
        metrics.update({ task_id : [m_n, m_v]})
    print(metrics)
    
    total_iters = 0
    for task_data in metrics.values():
        # print(task_data[0].shape, task_data[1].shape)
        total_iters += task_data[1].shape[1]
    x_range = np.arange(0, total_iters)

    concat_data = None; concat_names = None
    task_boundaries = []
    for task_data in metrics.values():
        task_boundaries.append(task_data[1].shape[1])
        if type(concat_data) is type(None):
            concat_data = task_data[1]
        else:
            concat_data = np.concatenate([concat_data, task_data[1]], axis=1)

    for x in range(0, len(task_data[0]), 2): # step=2 to only get accuracies for classification layer
        split_label = task_data[0][x].split('-', maxsplit=3)
        # plt.plot(x_range, concat_data[x], label=f'{split_label[-2]} {split_label[-1]}', lw=3)
        plt.scatter(x_range, concat_data[x], marker='v', s=25, c=colors[x], label=f'{split_label[-2]} {split_label[-1]}')
        
    last_step = 0
    for i, boundary in enumerate(task_boundaries):
        # print(i, len(task_boundaries))
        if i == len(task_boundaries)-1: continue
        if i == len(task_boundaries)-2: label = 'task boundaries'
        else: label = None
        plt.vlines(x=boundary+last_step, ymin=-0.1, ymax=1.1, linestyle='dashed', alpha=.5, lw=5, label=label, colors=['grey'])
        last_step += boundary
    # INFO: adapt based on training steps, metrics only contain log data each_n_steps!
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(12)) 
    ax.set_xlim(-100, int(total_iters)+100)    # e.g. SPLIT-MNIST 5^2: 24000/50=480*2=960
    current_x_ticks = plt.xticks()[0]
    new_x_ticks = list(np.asarray(current_x_ticks, dtype=np.int32) * 50)
    new_x_ticks[0] = None; new_x_ticks[-1] = None
    print(new_x_ticks)
    
    def convert_numbers(nums):
        converted = []
        for num in nums:
            if isinstance(num, type(None)):
                converted.append(str(""))
            else:
                num = int(num)
                if 1000 <= num:
                    converted.append(f"{num // 1000}k")
                else:
                    converted.append(str(num))
        print(converted)
        return converted

    converted_x_ticks = convert_numbers(new_x_ticks)
    plt.xticks(ticks=current_x_ticks, labels=converted_x_ticks)
    
    ax.set_ylim(-0.1, +1.1)
    # plt.vlines(x=[200, 400, 600, 800], ymin=0, ymax=[16000, 18000, 18000, 18000], linestyle='dashed', label='task switch', colors=['grey'])
    ax.yaxis.set_major_locator(plt.MaxNLocator(6)) 
    
    """
    for i in range(0, 5):
        plt.plot(task_labels[i], logs[i], alpha=1.0, ls='--', lw=2, label=f'T{i+2}', color=color[i])
        #x_1=[i, i+1]
        #y_1=logs[i][0:2]
        #plt.fill_between(x=x_1, y1=y_1, y2=-300., alpha=0.25, color=color[i])
    """
    ax.legend(frameon=True, loc='best', ncol=1, fontsize='16', markerscale=2.0)
    # plt.show()
    # fig.tight_layout()
    fig.set_figwidth(10)
    fig.set_figheight(4)
    fig.subplots_adjust(bottom=.15, left=.15)
    
    save_path = Path(save_path)
    if save_path and Path.exists(save_path):
        if save_path.is_absolute(): pass
        else: save_path = Path('/home/ak/Desktop')
    plt.savefig(f'{save_path}/cf_plot.svg',
                dpi=300., pad_inches=0.1, facecolor='auto', edgecolor='auto', format='svg')

if __name__ == '__main__':
    exp_path = sys.argv[1]
    save_path = sys.argv[2]
    plot_metric = sys.argv[3]
    
    metric_names, metric_values, num_tasks = load_data(exp_path)
    
    with plt.style.context('seaborn-v0_8'):  # 'bmh', 'ggplot'
        plot(metric_names, metric_values, num_tasks, save_path, plot_metric)
