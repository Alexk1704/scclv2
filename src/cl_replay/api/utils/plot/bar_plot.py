import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


plt.rc('legend', fontsize=16)    # legend fontsize


# insert actual num of gen. samples between each subtask here...
def line_print():
    labels = ('T2', 'T3', 'T4', 'T5', 'T6')

    gen_data = {
        'AR': np.array([5400, 5900, 6200, 5800, 5900]),
        'DGR (b.)': np.array([27000, 35400, 43400, 46400, 53100]),
        
    }

    #x = np.arange(start=0., stop=len(labels), step=1.)  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()
    bottom = np.zeros(5)

    ax.grid(linestyle='solid', color='w')

    # hide ticks
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()
    
    ax.tick_params(axis='y', which='major', colors='white', direction='in')
    for tick in ax.get_xticklabels():
        tick.set_color('black')
    for tick in ax.get_yticklabels():
        tick.set_color('black')

    colors = [color for color in mcolors.TABLEAU_COLORS]

    for (gen_method, gen_samples), color in zip(gen_data.items(), colors):
        p = ax.bar(labels, gen_samples, width, label=gen_method, bottom=bottom, color=color)
        bottom += gen_samples

        ax.bar_label(p, label_type='center')

    #rects1 = ax.bar(labels, x - width/2, vae_gen, width, label='VAE-DGR (b.)', color='blue')
    #rects2 = ax.bar(labels, x + width/2, gmm_gen, width, label='AR', color='green')

    #rects1 = ax.barh(x - width/2, width=gmm_gen, height=0.4, align='edge', linewidth=2., label='AR', color='green')
    #rects2 = ax.barh(x - width/2, width=vae_gen, height=0.4, align='edge', linewidth=2., left=gmm_gen, label='VAE-DGR (b.)', color='cornflowerblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.invert_yaxis()  # labels read top-to-bottom

    #ax.bar_label(rects1, padding=2, fontsize=14)
    #ax.bar_label(rects2, padding=2, fontsize=14)

    #plt.title('Data generation for E-MNIST20-1', pad=8)
    #plt.setp(ax.get_yticklabels(), va="center", ha="right", rotation_mode="anchor")

    plt.xlabel(xlabel='task', labelpad=4, fontsize=24)
    plt.ylabel(ylabel='# replayed samples', labelpad=4, fontsize=24)
    
    #ax.set_yticks(ticks=x, labels=labels)
    ax.set_ylim(-2400, 84000)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6)) # set number of y ticks
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=.15, left=.15)
    plt.savefig('/home/ak/Desktop/bar_plot.png')
    plt.show()

with plt.style.context('ggplot'):         # 'bmh'
    line_print()