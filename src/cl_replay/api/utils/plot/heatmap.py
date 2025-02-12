import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Also check: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def create_heatmap(data_arr, col_labels, row_labels, name, cb_label):
    
    # data_arr = np.array([
    #     [-1,	+4,	    +23,    0,	    -2,	np.nan,	np.nan],
    #     [-1,	+6,	    +14,    -1,	    0,	np.nan,	np.nan],
    #     [+2,	+1,	    -4,	    0,	    0,	np.nan,	np.nan],
    #     [+3,	-4,	    +4,	    0,	    +2,	np.nan,	np.nan],
    #     [+1,	+2,	    -3,	    +2,	    +11,np.nan,	np.nan],
    #     [+2,	+4,	    +10,    +1,	    +10,np.nan,	np.nan],
    #     [0,	    0,	    +5,	    +3,	    -1,	np.nan,	np.nan],
    #     [+1,	+1,	    +12,    +4,	    0,	np.nan,	np.nan],
    #     [np.nan,	np.nan,	    np.nan,	+8,	np.nan,	+9,	-6],
    #     [np.nan,	np.nan,     np.nan,	+7,	np.nan,	+8,	-4]
    # ])
    print(data_arr.shape)
    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap = ax.pcolor(data_arr, cmap=plt.cm.viridis, 
                        #vmin=np.nanmin(data_arr), vmax=np.nanmax(data_arr))
                        vmin=np.min(data_arr), vmax=np.max(data_arr))
    # https://stackoverflow.com/questions/25071968/heatmap-with-text-in-each-cell-with-matplotlibs-pyplot (HYRY)
    def show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        data_arr = np.ravel(pc.get_array())
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), data_arr):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, fontsize='large', ha="center", va="center", color=color, **kw)

    # https://stackoverflow.com/a/16125413/190597 (Joe Kington)
    ax.patch.set(hatch='x', edgecolor='black')
    cb = fig.colorbar(heatmap)
    for t in cb.ax.get_yticklabels(): t.set_fontsize(16)
    cb.set_label(cb_label, fontsize=20)
    
    #show_values(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(data_arr.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(data_arr.shape[0])+0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False, fontsize=16)
    ax.set_yticklabels(col_labels, minor=False, fontsize=16)
    #ax.tick_params(axis='x', labelrotation=45)
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    fig.subplots_adjust(bottom=.15, left=.15)
    
    plt.savefig(f'/home/ak/Desktop/{name}.png')
    
     

er_row_labels = ['const.', 'lw-cls.', 'lw-tsk.']
dgr_row_labels = ['const.', 'balan.', 'lw-cls.', 'lw-tsk.']

mnist_col_header = ['M: U-CP1', 'M: U-CP2', 'M: U-CP3', 'M: D-CP1', 'M: D-CP2'] 
fashion_col_header = ['F: U-CP1', 'F: U-CP2', 'F: U-CP3', 'F: D-CP1', 'F: D-CP2']
svhn_col_header = ['S: U-CP1', 'S: U-CP2', 'S: U-CP3', 'S: D-CP1', 'S: D-CP2'] 
cifar10_col_header = ['C: U-CP1', 'C: U-CP2', 'C: U-CP3', 'C: D-CP1', 'C: D-CP2']
emnist_col_header = ['E: D-CP1', 'E: A-CP1', 'E: A-CP2']

all_col_header = []
all_col_header += mnist_col_header
all_col_header += fashion_col_header
all_col_header += svhn_col_header
all_col_header += cifar10_col_header
all_col_header += emnist_col_header


######
# ER #
######


er_mnist_acc = np.array([
    [.85,		.76,		.15,		.86,		.84,],
    [.84,		.80,		.38,		.86,		.82,],
    [.84,		.82,		.29,		.85,		.84,],
])


er_fashion_acc = np.array([
    [.71,		.60,		.37,		.64,		.69,],
    [.73,		.61,		.33,		.64,		.69,],
    [.74,		.57,		.41,		.64,		.71,],
])


er_svhn_acc = np.array([
    [.81,		.78,		.56,		.82,		.75,],
    [.82,		.80,		.53,		.84,		.86,],
    [.83,		.82,		.66,		.83,		.85,],
])


er_cifar10_acc = np.array([
    [.62,		.62,		.38,		.60,		.65,],
    [.62,		.62,		.43,		.63,		.64,],
    [.63,		.63,		.50,		.64,		.65,],
])


er_emnist_acc = np.array([
    [.64,		.54,		.64,],
    [.72,		.58,		.63,],
    [.71,		.60,		.62,],
])

#-------------------------------------------------------------


er_mnist_forg = np.array([
    [.27,		.26,		.64,		.10,		.21,],
    [.28,		.24,		.53,		.11,		.19,],
    [.29,		.25,		.62,		.12,		.19,],
])


er_fashion_forg = np.array([
    [.47,		.49,		.81,		.20,		.36,],
    [.43,		.48,		.54,		.20,		.34,],
    [.39,		.53,		.44,		.19,		.34,],
])


er_svhn_forg = np.array([
    [.25,		.26,		.40,		.11,		.26,],
    [.22,		.21,		.39,		.10,		.13,],
    [.22,		.22,		.26,		.10,		.17,],
])


er_cifar10_forg = np.array([
    [.40,		.36,		.63,		.16,		.32,],
    [.40,		.35,		.39,		.16,		.26,],
    [.39,		.35,		.27,		.18,		.30,],
])


er_emnist_forg = np.array([
    [.23,		.52,		.47,],
    [.22,		.49,		.47,],
    [.24,		.44,		.46,],
])

###############
# CVAE - WARM #
###############


cvae_warm_mnist_acc = np.array([
    [.97,		.92,		.83,		.95,		.91,],
    [.97,		.92,		.81,		.95,		.92,],
    [.97,		.91,		.83,		.94,		.87,],
    [.97,		.90,		.82,		.94,		.88,],
])


cvae_warm_fashion_acc = np.array([
    [.78,		.63,		.67,		.66,		.65,],
    [.77,		.67,		.71,		.69,		.70,],
    [.77,		.55,		.67,		.59,		.69,],
    [.79,		.58,		.68,		.64,		.68,],
])


cvae_warm_svhn_acc = np.array([
    [.68,		.54,		.43,		.56,		.58,],
    [.67,		.54,		.53,		.64,		.65,],
    [.70,		.53,		.46,		.56,		.52,],
    [.76,		.53,		.40,		.58,		.53,],
])


cvae_warm_cifar10_acc = np.array([
    [.54,		.44,		.34,		.42,		.36,],
    [.60,		.44,		.37,		.44,		.36,],
    [.55,		.43,		.36,		.47,		.37,],
    [.57,		.47,		.37,		.44,		.44,],
])


cvae_warm_emnist_acc = np.array([
    [.44,		.63,		.71,],
    [.44,		.66,		.74,],
    [.40,		.54,		.62,],
    [.40,		.54,		.65,],
])

#-------------------------------------------------------------


cvae_warm_mnist_forg = np.array([
    [.03,		.08,		.19,		.03,		.07,],
    [.03,		.08,		.19,		.02,		.05,],
    [.03,		.10,		.20,		.04,		.14,],
    [.03,		.12,		.20,		.03,		.09,],
])


cvae_warm_fashion_forg = np.array([
    [.34,		.44,		.41,		.18,		.27,],
    [.37,		.41,		.35,		.18,		.20,],
    [.36,		.55,		.41,		.21,		.35,],
    [.31,		.51,		.44,		.19,		.34,],
])


cvae_warm_svhn_forg = np.array([
    [.49,		.53,		.60,		.28,		.43,],
    [.52,		.52,		.48,		.24,		.36,],
    [.46,		.53,		.59,		.25,		.59,],
    [.57,		.55,		.65,		.29,		.47,],
])


cvae_warm_cifar10_forg = np.array([
    [.63,		.66,		.71,		.27,		.59,],
    [.51,		.66,		.70,		.27,		.57,],
    [.60,		.66,		.72,		.24,		.62,],
    [.57,		.62,		.74,		.27,		.56,],
])


cvae_warm_emnist_forg = np.array([
    [.34,		.36,		.45,],	
    [.29,		.34,		.39,],
    [.54,		.47,		.62,],
    [.38,		.47,		.59,],
])

################
# CVAE - RESET #
################


cvae_reset_mnist_acc = np.array([
    [.97,		.93,		.82,		.94,		.90,],
    [.97,		.93,		.86,		.95,		.90,],
    [.97,		.88,		.81,		.94,		.88,],
    [.97,		.90,		.83,		.93,		.88,],
])


cvae_reset_fashion_acc = np.array([
    [.78,		.62,		.65,		.65,		.67,],
    [.79,		.66,		.71,		.69,		.70,],
    [.78,		.56,		.68,		.64,		.65,],
    [.78,		.57,		.68,		.63,		.67,],
])


cvae_reset_svhn_acc = np.array([
    [.66,		.52,		.43,		.58,		.60,],
    [.70,		.57,		.54,		.65,		.66,],
    [.72,		.53,		.42,		.58,		.47,],
    [.72,		.54,		.39,		.56,		.54,],
])


cvae_reset_cifar10_acc = np.array([
    [.55,		.44,		.33,		.43,		.37,],
    [.57,		.44,		.35,		.46,		.38,],
    [.51,		.46,		.27,		.48,		.35,],
    [.57,		.48,		.28,		.48,		.36,],
])


cvae_reset_emnist_acc = np.array([
    [.39,		.70,		.61,],
    [.40,		.69,		.65,],
    [.39,		.66,		.57,],
    [.42,		.61,		.58,],
])

#-------------------------------------------------------------


cvae_reset_mnist_forg = np.array([
    [.04,		.08,		.19,		.03,		.08,],
    [.03,		.07,		.15,		.02,		.07,],
    [.03,		.14,		.22,		.03,		.14,],
    [.03,		.13,		.20,		.04,		.11,],
])


cvae_reset_fashion_forg = np.array([
    [.33,		.46,		.40,		.18,		.26,],
    [.30,		.41,		.36,		.18,		.20,],
    [.34,		.53,		.45,		.19,		.38,],
    [.32,		.53,		.43,		.20,		.36,],
])


cvae_reset_svhn_forg = np.array([
    [.55,		.53,		.59,		.30,		.42,],
    [.46,		.49,		.49,		.22,		.37,],
    [.43,		.53,		.65,		.28,		.60,],
    [.43,		.53,		.73,		.28,		.49,],
])


cvae_reset_cifar10_forg = np.array([
    [.61,		.67,		.73,		.29,		.58,],
    [.56,		.65,		.70,		.26,		.57,],
    [.70,		.65,		.81,		.26,		.63,],
    [.58,		.64,		.83,		.27,		.60,],
])


cvae_reset_emnist_forg = np.array([
    [.42,		.38,		.45,],
    [.29,		.41,		.41,],
    [.57,		.45,		.58,],
    [.41,		.50,		.51,],
])

#################
# WGANGP - WARM #
#################


wgangp_warm_mnist_acc = np.array([
    [.95,	.88,	.70,	.93,	.83,],
    [.94,	.92,	.80,	.94,	.91,],
    [.96,	.89,	.71,	.93,	.87,],
    [.95,	.90,	.70,	.92,	.86,],
])


wgangp_warm_fashion_acc = np.array([
    [.69,	.44,	.54,	.59,	.50,],
    [.69,	.45,	.51,	.59,	.52,],
    [.71,	.47,	.52,	.61,	.51,],
    [.68,	.45,	.53,	.59,	.47,],
])


wgangp_warm_svhn_acc = np.array([
    [.56,	.35,	.30,	.41,	.44,],
    [.54,	.40,	.32,	.43,	.39,],
    [.56,	.39,	.25,	.48,	.46,],
    [.54,	.36,	.29,	.41,	.46,],
])


wgangp_warm_cifar10_acc = np.array([
    [.44,	.39,	.28,	.36,	.32,],
    [.42,	.42,	.33,	.41,	.40,],
    [.44,	.38,	.30,	.42,	.37,],
    [.44,	.40,	.36,	.40,	.31,],
])


wgangp_warm_emnist_acc = np.array([
    [.22,	.59,	.55,],
    [.42,	.60,	.52,],
    [.24,	.61,	.56,],
    [.24,	.62,	.56,],
])

#-------------------------------------------------------------


wgangp_warm_mnist_forg = np.array([
    [.08,	.14,	.32,	.05,	.10,],
    [.09,	.09,	.21,	.04,	.06,],
    [.05,	.12,	.30,	.04,	.08,],
    [.06,	.12,	.32,	.05,	.08,],
])


wgangp_warm_fashion_forg = np.array([
    [.52,	.69,	.51,	.23,	.33,],
    [.51,	.68,	.56,	.23,	.35,],
    [.48,	.65,	.54,	.23,	.32,],
    [.55,	.68,	.55,	.23,	.35,],
])


wgangp_warm_svhn_forg = np.array([
    [.72,	.78,	.72,	.41,	.61,],
    [.77,	.70,	.71,	.38,	.79,],
    [.73,	.70,	.74,	.38,	.57,],
    [.76,	.71,	.73,	.36,	.59,],
])


wgangp_warm_cifar10_forg = np.array([
    [.79,	.79,	.78,	.34,	.67,],
    [.83,	.76,	.78,	.33,	.65,],
    [.80,	.78,	.72,	.33,	.55,],
    [.79,	.75,	.73,	.34,	.67,],
])


wgangp_warm_emnist_forg = np.array([
    [.37,	.47,	.52,],
    [.39,	.50,	.59,],
    [.36,	.46,	.50,],
    [.36,	.43,	.52,],
])

##################
# WGANGP - RESET #
##################


wgangp_reset_mnist_acc = np.array([
    [.95,		.89,		.66,		.92,		.82,],
    [.94,		.90,		.80,		.93,		.90,],
    [.95,		.88,		.63,		.91,		.84,],
    [.95,		.88,		.65,		.93,		.83,],
])


wgangp_reset_fashion_acc = np.array([
    [.70,		.42,		.46,		.61,		.49,],
    [.71,		.44,		.55,		.60,		.54,],
    [.68,		.45,		.49,		.62,		.53,],
    [.70,		.48,		.47,		.61,		.49,],
])


wgangp_reset_svhn_acc = np.array([
    [.60,		.33,		.26,		.39,		.41,],
    [.54,		.36,		.38,		.44,		.43,],
    [.56,		.37,		.25,		.52,		.47,],
    [.58,		.41,		.28,		.43,		.47,],
])


wgangp_reset_cifar10_acc = np.array([
    [.43,		.38,		.24,		.35,		.25,],
    [.43,		.41,		.33,		.38,		.36,],
    [.44,		.37,		.24,		.40,		.28,],
    [.44,		.39,		.28,		.39,		.27,],
])


wgangp_reset_emnist_acc = np.array([
    [.23,		.56,		.50,],
    [.40,		.60,		.51,],
    [.23,		.60,		.52,],
    [.24,		.60,		.51,],
])

#-------------------------------------------------------------


wgangp_reset_mnist_forg = np.array([
    [.07,		.13,		.36,		.05,		.12,],
    [.09,		.12,		.21,		.04,		.08,],
    [.07,		.14,		.39,		.06,		.10,],
    [.08,		.14,		.38,		.04,		.10,],
])


wgangp_reset_fashion_forg = np.array([
    [.49,		.72,		.60,		.22,		.37,],
    [.49,		.69,		.57,		.22,		.36,],
    [.53,		.68,		.57,		.24,		.29,],
    [.51,		.63,		.58,		.22,		.34,],
])


wgangp_reset_svhn_forg = np.array([
    [.66,		.80,		.75,		.43,		.69,],
    [.75,		.74,		.70,		.35,		.68,],
    [.71,		.75,		.75,		.37,		.55,],
    [.68,		.70,		.70,		.39,		.56,],
])


wgangp_reset_cifar10_forg = np.array([
    [.80,		.79,		.84,		.35,		.73,],
    [.81,		.77,		.85,		.36,		.71,],
    [.78,		.78,		.81,		.34,		.66,],
    [.79,		.80,		.79,		.33,		.66,],
])


wgangp_reset_emnist_forg = np.array([
    [.37,		.52,		.56,],
    [.44,		.50,		.62,],
    [.35,		.47,		.57,],
    [.36,		.47,		.56,],
])

###############
# CGAN - WARM #
###############


cgan_warm_mnist_acc = np.array([
    [.93,		.20,		.10,		.10,		.57,],
    [.93,		.20,		.10,		.11,		.37,],
    [.94,		.19,		.10,		.10,		.78,],
    [.78,		.20,		.10,		.10,		.31,],
])


cgan_warm_fashion_acc = np.array([
    [.64,		.48,		.20,		.55,		.51,],
    [.66,		.48,		.13,		.56,		.57,],
    [.64,		.48,		.19,		.55,		.48,],
    [.63,		.44,		.10,		.55,		.54,],
])


cgan_warm_svhn_acc = np.array([
    [.45,		.15,		.06,		.09,		.20,],
    [.45,		.09,		.06,		.09,		.20,],
    [.45,		.11,		.07,		.08,		.20,],
    [.45,		.13,		.06,		.08,		.20,],
])


cgan_warm_cifar10_acc = np.array([
    [.40,		.19,		.10,		.10,		.10,],
    [.40,		.20,		.10,		.10,		.10,],
    [.40,		.20,		.10,		.10,		.10,],
    [.41,		.19,		.10,		.10,		.10,],
])


cgan_warm_emnist_acc = np.array([
    [.21,	.17,	.55,],
    [.12,	.16,	.52,],
    [.31,	.17,	.54,],
    [.34,	.16,	.53,],
])

#-------------------------------------------------------------


cgan_warm_mnist_forg = np.array([
    [.10,		1.0,		1.0,		.49,		.47,],
    [.10,		1.0,		1.0,		.49,		.70,],
    [.10,		1.0,		1.0,		.50,		.21,],
    [.41,		1.0,		1.0,		.49,		.75,],
])


cgan_warm_fashion_forg = np.array([
    [.61,		.63,		.91,		.25,		.40,],
    [.59,		.64,		.95,		.24,		.43,],
    [.60,		.63,		.89,		.26,		.46,],
    [.63,		.68,		.96,		.24,		.38,],
])


cgan_warm_svhn_forg = np.array([
    [.94,		.98,		1.0,		.48,		.99,],
    [.94,		.79,		1.0,		.47,		.99,],
    [.93,		.92,		1.0,		.48,		.99,],
    [.93,		.98,		1.0,		.48,		.99,],
])


cgan_warm_cifar10_forg = np.array([
    [.87,		.93,		1.0,		.43,		.97,],
    [.87,		.93,		1.0,		.43,		.97,],
    [.87,		.94,		1.0,		.43,		.97,],
    [.86,		.93,		1.0,		.43,		.97,],
])


cgan_warm_emnist_forg = np.array([
    [.70,	.96,	.52,],
    [.90,	.97,	.58,],
    [.51,	.97,	.55,],
    [.51,	.97,	.57,],
])

################
# CGAN - RESET #
################


cgan_reset_mnist_acc = np.array([
    [.94,	.20,	.10,	.26,	.33,],
    [.93,	.20,	.10,	.37,	.38,],
    [.93,	.22,	.11,    .39,	.36,],
    [.94,	.21,	.10,	.51,	.34,],
])


cgan_reset_fashion_acc = np.array([
    [.65,	.44,	.55,	.52,	.49,],
    [.66,	.52,	.60,	.57,	.56,],
    [.67,	.48,	.54,	.55,	.52,],
    [.66,	.47,	.57,	.55,	.52,],
])


cgan_reset_svhn_acc = np.array([
    [.45,		.16,		.06,		.08,		.20,],	
    [.45,		.13,		.06,		.12,		.20,],
    [.45,		.13,		.10,		.08,		.20,],
    [.45,		.08,		.08,		.08,		.19,],
])


cgan_reset_cifar10_acc = np.array([
    [.40,		.19,		.10,		.10,		.10,],
    [.40,		.16,		.10,		.12,		.10,],
    [.41,		.19,		.10,		.10,		.10,],
    [.40,		.19,		.10,		.11,		.11,],
])


cgan_reset_emnist_acc = np.array([
    [.05,		.54,		.35,],
    [.29,		.60,		.48,],
    [.06,		.59,		.35,],
    [.04,		.59,		.28,],
])

#-------------------------------------------------------------


cgan_reset_mnist_forg = np.array([
    [.11,	1.0,	1.0,	.42,	.71,],
    [.12,	.99,	1.0,	.35,	.69,],
    [.10,	.97,	1.0,	.35,	.70,],
    [.10,	.98,	1.0,	.28,	.72,],
])


cgan_reset_fashion_forg = np.array([
    [.59,	.68,	.58,	.28,	.42,],
    [.58,	.60,	.52,	.25,	.43,],
    [.56,	.63,	.55,	.25,	.40,],
    [.59,	.64,	.55,	.25,	.40,],
])


cgan_reset_svhn_forg = np.array([
    [.93,		.95,		1.0,		.48,		.99,],
    [.93,		.90,		1.0,		.70,		.99,],
    [.94,		.95,		.96,		.48,		.99,],
    [.93,		.79,		.98,		.48,		.99,],
])


cgan_reset_cifar10_forg = np.array([
    [.87,		.93,		1.0,		.43,		.96,],
    [.88,		.91,		1.0,		.43,		.97,],
    [.87,		.93,		1.0,		.43,		.96,],
    [.87,		.93,		1.0,		.43,		.99,],
])


cgan_reset_emnist_forg = np.array([
    [.90,		.59,		.85,],
    [.59,		.55,		.64,],
    [.92,		.55,		.84,],
    [.93,		.55,		.96,],
])

# ACC: replay mods

er_acc_stacked = np.hstack([er_mnist_acc, er_fashion_acc, er_svhn_acc, er_cifar10_acc, er_emnist_acc])
cvae_warm_acc_stacked = np.hstack([cvae_warm_mnist_acc, cvae_warm_fashion_acc, cvae_warm_svhn_acc, cvae_warm_cifar10_acc, cvae_warm_emnist_acc])
wgangp_warm_acc_stacked = np.hstack([wgangp_warm_mnist_acc, wgangp_warm_fashion_acc, wgangp_warm_svhn_acc, wgangp_warm_cifar10_acc, wgangp_warm_emnist_acc])
cgan_warm_acc_stacked = np.hstack([cgan_warm_mnist_acc, cgan_warm_fashion_acc, cgan_warm_svhn_acc, cgan_warm_cifar10_acc, cgan_warm_emnist_acc])

acc1_stacked = np.vstack([er_acc_stacked, cvae_warm_acc_stacked, cgan_warm_acc_stacked, wgangp_warm_acc_stacked, ])
row_labels = er_row_labels + dgr_row_labels + dgr_row_labels + dgr_row_labels

create_heatmap(acc1_stacked, row_labels, all_col_header, name='mods_acc', cb_label='accuracy')

# FORG: replay mods

er_forg_stacked = np.hstack([er_mnist_forg, er_fashion_forg, er_svhn_forg, er_cifar10_forg, er_emnist_forg])
cvae_warm_forg_stacked = np.hstack([cvae_warm_mnist_forg, cvae_warm_fashion_forg, cvae_warm_svhn_forg, cvae_warm_cifar10_forg, cvae_warm_emnist_forg])
wgangp_warm_forg_stacked = np.hstack([wgangp_warm_mnist_forg, wgangp_warm_fashion_forg, wgangp_warm_svhn_forg, wgangp_warm_cifar10_forg, wgangp_warm_emnist_forg])
cgan_warm_forg_stacked = np.hstack([cgan_warm_mnist_forg, cgan_warm_fashion_forg, cgan_warm_svhn_forg, cgan_warm_cifar10_forg, cgan_warm_emnist_forg])

forg1_stacked = np.vstack([er_forg_stacked, cvae_warm_forg_stacked, cgan_warm_forg_stacked, wgangp_warm_forg_stacked])

create_heatmap(forg1_stacked, row_labels, all_col_header, name='mods_forg', cb_label='forgetting')

# ACC: generator

cvae_reset_acc_stacked = np.hstack([cvae_reset_mnist_acc, cvae_reset_fashion_acc, cvae_reset_svhn_acc, cvae_reset_cifar10_acc, cvae_reset_emnist_acc])
wgangp_reset_acc_stacked = np.hstack([wgangp_reset_mnist_acc, wgangp_reset_fashion_acc, wgangp_reset_svhn_acc, wgangp_reset_cifar10_acc, wgangp_reset_emnist_acc])
cgan_reset_acc_stacked = np.hstack([cgan_reset_mnist_acc, cgan_reset_fashion_acc, cgan_reset_svhn_acc, cgan_reset_cifar10_acc, cgan_reset_emnist_acc])

acc2_stacked_gen = np.vstack([er_acc_stacked, cvae_reset_acc_stacked, cgan_reset_acc_stacked, wgangp_reset_acc_stacked])

# FORG: generator

cvae_reset_forg_stacked = np.hstack([cvae_reset_mnist_forg, cvae_reset_fashion_forg, cvae_reset_svhn_forg, cvae_reset_cifar10_forg, cvae_reset_emnist_forg])
wgangp_reset_forg_stacked = np.hstack([wgangp_reset_mnist_forg, wgangp_reset_fashion_forg, wgangp_reset_svhn_forg, wgangp_reset_cifar10_forg, wgangp_reset_emnist_forg])
cgan_reset_forg_stacked = np.hstack([cgan_reset_mnist_forg, cgan_reset_fashion_forg, cgan_reset_svhn_forg, cgan_reset_cifar10_forg, cgan_reset_emnist_forg])

forg2_stacked_gen = np.vstack([er_forg_stacked, cvae_reset_forg_stacked, cgan_reset_forg_stacked, wgangp_reset_forg_stacked]) 

diff_acc = acc2_stacked_gen - acc1_stacked
diff_forg = forg2_stacked_gen - forg1_stacked

create_heatmap(acc2_stacked_gen, row_labels, all_col_header, name='gen_acc', cb_label='accuracy')
create_heatmap(forg2_stacked_gen, row_labels, all_col_header, name='gen_forg', cb_label='forgetting')

acc2_stacked_l1norm_col = np.sum(np.abs(acc2_stacked_gen), axis=0)
acc2_stacked_normalized = acc2_stacked_gen / acc2_stacked_l1norm_col

forg2_stacked_l1norm_col = np.sum(np.abs(forg2_stacked_gen), axis=0)
forg2_stacked_normalized = forg2_stacked_gen / forg2_stacked_l1norm_col

create_heatmap(acc2_stacked_normalized, row_labels, all_col_header, name='gen_acc_norm', cb_label='accuracy (L1 norm.)')
create_heatmap(forg2_stacked_normalized, row_labels, all_col_header, name='gen_forg_norm', cb_label='forgetting (L1 norm.)')

create_heatmap(diff_acc, row_labels, all_col_header, name='gen_diff_acc', cb_label='accuracy gain')
create_heatmap(diff_forg, row_labels, all_col_header, name='gen_diff_forg', cb_label='forgetting gain')

# NORM DATA: max. from column (L1-norm column)
#L1 norm acc
acc1_stacked_l1norm_col = np.sum(np.abs(acc1_stacked), axis=0)
acc1_stacked_normalized = acc1_stacked / acc1_stacked_l1norm_col
create_heatmap(acc1_stacked_normalized, row_labels, all_col_header, name='mods_acc_norm', cb_label='accuracy (L1 norm.)')

#L1 norm forgetting
forg1_stacked_l1norm_col = np.sum(np.abs(forg1_stacked), axis=0)
forg1_stacked_normalized = np.abs(forg1_stacked) / forg1_stacked_l1norm_col
create_heatmap(forg1_stacked_normalized, row_labels, all_col_header, name='mods_forg_norm', cb_label='forgetting (L1 norm.)')

#  acc & forgetting
diff_acc_l1norm_col = np.sum(np.abs(diff_acc), axis=0)
diff_acc_normalized = np.abs(diff_acc) / diff_acc_l1norm_col

print(diff_acc_normalized)

diff_forg_l1norm_col = np.sum(np.abs(diff_forg), axis=0)
diff_forg_normalized = np.abs(diff_forg) / diff_forg_l1norm_col

print(diff_forg_normalized)

create_heatmap(diff_acc_normalized, row_labels, all_col_header, name='gen_diff_acc_norm', cb_label='accuracy gain (L1 norm.)')
create_heatmap(diff_forg_normalized, row_labels, all_col_header, name='gen_diff_forg_norm', cb_label='forgetting gain (L1 norm.)')