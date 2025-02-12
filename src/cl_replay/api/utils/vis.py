"""
The definitive vis tools for GMM prototypes etc.
"""

import os
import sys, gzip, pickle
from pathlib import Path

import numpy              as np
import matplotlib         as mp
import matplotlib.pyplot  as plt, sys, math
import imageio.v2 as imageio

from matplotlib   import cm
from argparse   import ArgumentParser

mp.use("Agg") # so as to not try to invoke X on systems without it



def getIndexArray(w_in, h_in, c_in):
    indicesOneSample  = np.zeros([w_in*h_in*c_in],dtype=np.int32)
    tileX             = int(math.sqrt(c_in)) ;  tileY = tileX
    tilesX            = w_in ;                  tilesY = c_in
    sideX             = tileX * w_in ;          sideY = tileY * h_in

    for inIndex in range(0,w_in * h_in * c_in):
        tileIndexFlat = inIndex // c_in
        tileIndexX    = tileIndexFlat % tilesX
        tileIndexY    = tileIndexFlat // tilesX

        tilePosFlat   = inIndex % (tileX*tileY)
        tilePosY      = tilePosFlat // tileX
        tilePosX      = tilePosFlat % tileX

        posY          = tileIndexY * tileY + tilePosY
        posX          = tileIndexX * tileX + tilePosX
        outIndex      = sideX  * posY + posX

        indicesOneSample [outIndex] = inIndex

    return indicesOneSample


def vis_img_data(parser_args, prefix, prefix_path=None, suffix=0):
    """
    visualizes data saved by GMM Layers --> GMM_Layer.py
    Can vis, per component, depending on parameter "--what":
    - centroids (mus)
    - diagonal sigmas (sigmas)
    - loading matrix rows (loadingMatrix)
    - convGMM centroids arranged in visually intuitive way (organizedWeights)
    Over all visualizations the pi value of that comp. can be overlaid.
    """
    pifile    = FLAGS.pi_file
    mufile    = FLAGS.mu_file
    sigmafile = FLAGS.sigma_file
    channels  = FLAGS.channels

    print(f'IN: {prefix_path}')

    if FLAGS.what == 'loadingMatrix':
        sigmafile = 'gammas.npy'

    if FLAGS.vis_cons:
        train_confile   = 'train_connectivities.npy'
        test_confile    = 'eval_connectivities.npy'
        try:
            train_cons      = np.load(prefix_path + train_confile)
            vis_train_cons  = True
            component_count = train_cons.shape[1]
        except OSError: vis_train_cons = False
        try:
            eval_cons       = np.load(prefix_path + test_confile)
            vis_eval_cons   = True
            component_count = eval_cons.shape[1]
        except OSError: vis_eval_cons = False

    protos          = np.load(prefix_path + mufile)

    print(f'FILE(S) LOCATION: {prefix_path}')
    print(f'FILE(S) PREFIX: {prefix}')

    if FLAGS.img_sequence:
        seq_path        = FLAGS.data_path
        task_id         = seq_path[seq_path.find('_')+1:]
        save_path       = FLAGS.out + '/' + f'{prefix}{task_id}_E{suffix}' + '_mus.png'
        print(f'TASK ID: {task_id}')
    else:
        save_path = FLAGS.out + '/' + f'sampled.png'

    print(f'SAVE: {save_path}')

    print ("Raw protos have shape", protos.shape)

    pis = sigmas = None
    if len(protos.shape) == 4: # sampling file, single npy with dims N,d*d
        _n,_h,_w,_c = protos.shape
        _d          = _h*_w*_c ;
        pis         = np.zeros([_n,_d])
        sigmas      = np.zeros([_n,_d])
        protos      = protos.reshape(_n,_d)
    else:
        pis     = np.load(prefix_path + pifile)[0, FLAGS.y, FLAGS.x]
        sigmas  = np.load(prefix_path + sigmafile)[0,FLAGS.y, FLAGS.x]
        protos  = protos[0, FLAGS.y, FLAGS.x]

    n     = int(math.sqrt(protos.shape[0]))
    imgW  = int(math.sqrt(protos.shape[1] / channels))
    imgH  = int(math.sqrt(protos.shape[1] / channels))
    d_    = int(math.sqrt(protos.shape[1] / channels))

    #print (FLAGS.proto_size)
    if FLAGS.proto_size[0] != -1:
        imgH, imgW = FLAGS.proto_size
    #print (imgH,imgW, "!!!")
    #print ("ND=", n,d_)

    h_in, w_in, c_in    = FLAGS.prev_layer_hwc
    h_out, w_out, c_out = FLAGS.cur_layer_hwc
    pH,pW               = FLAGS.filter_size

    indices = None
    if h_in != -1:
        indices = getIndexArray(h_in, w_in, c_in, h_out, w_out, c_out, pH, pW)
        print (indices.shape, "INDICES")
        print (indices.min(), indices.max(), "INDICES")

    if FLAGS.what == "mus2D":
        f,ax = plt.subplots(1,1)
        data = None
        if FLAGS.dataset_file != "":
            with gzip.open(FLAGS.dataset_file) as f:
                data = pickle.load(f)["data_test"]
            #print ("Loaded data: ", data.shape)
        if data is not None: ax.scatter(data[:,0], data[:,1])
        ax.scatter(protos[:,0],protos[:,1])
        plt.tight_layout(pad=1, h_pad=.0, w_pad=-10)
        plt.savefig(FLAGS.out)
        sys.exit(0)

    f = axes = None
    #f, axes = plt.subplots(n, n, gridspec_kw={'wspace':0, 'hspace':0}) ;
    f, axes = plt.subplots(n, n)

    if n ==1:
        f = np.array([f])
        axes = np.array([axes])

    axes = axes.ravel()
    index = -1

    exp_pi = np.exp(pis)
    sm = exp_pi/exp_pi.sum()

    pad, w_pad, h_pad = 0., 0., 0.

    for (dir_, ax_, pi_, sig_) in zip(protos, axes, sm, sigmas):
        index += 1

        disp = dir_
        if FLAGS.what == "precs_diag":      disp = sig_
        if FLAGS.what == "loadingMatrix":   disp = sig_[:,FLAGS.l]
        #print (disp.shape);

        if FLAGS.what == "organizedWeights" and indices is not None:
            dispExp = (disp.reshape(h_in, w_in, c_in))
            #print (dispExp.shape)
            disp = dispExp
            disp = disp.ravel()[indices]
            #print ("Disp hape=",disp.shape)

        #print ("minmax=", disp.min(), disp.max())
        disp    = np.clip(disp,FLAGS.clip_range[0],FLAGS.clip_range[1])

        refmin  = disp.min() if FLAGS.disp_range[0] == -100 else FLAGS.disp_range[0]
        refmax  = disp.max() if FLAGS.disp_range[1] == +100 else FLAGS.disp_range[1]

        # This is interesting to see unconverged components
        #print(index, "minmax=", disp.min(), disp.max(), refmin,refmax, disp.shape, channels, imgH, imgW)

        if FLAGS.vis_cons:
            def vis_connections(comp_con):
                component_con = comp_con[:, index]
                #print(f'component activity (row), 0-10 classes for one component:\n{component_con}\n')
                top_K_indices   = np.argpartition(component_con, -1)[-3:]
                top_K_vals      = np.array(component_con[top_K_indices], dtype=np.float32)
                top_K_vals      /= top_K_vals.sum(dtype=np.float32)
                top_K_vals      = top_K_vals.round(2)

                #print('component TOP-K: ', top_K_indices, top_K_vals)

                colors = ['yellow', 'orange', 'red']
                if imgH > 20:
                    first = 6
                    increment = 8
                    x = 0

                if imgH < 20:
                    first = 2
                    increment = 1
                    x = 0

                if imgH < 14:
                    first = 1
                    increment = 1.5
                    x = 0

                if component_count <= 36:
                    for K in range(2, -1, -1):
                        ax_.text(x, first, 'M{:} {:.2f}'.format(top_K_indices[K], top_K_vals[K]), fontsize=8, c=colors[K], alpha=0.5,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontfamily='sans-serif', fontstretch='ultra-condensed', fontweight='bold')
                        #ax_.text.set(alpha=0.5, horizontalalignment='center', fontvariant='small-caps')
                        first += increment
                    w_pad   = .5
                    h_pad   = .5
                    pad     = .1

                else:
                    for K in range(2, -1, -1):
                        ax_.text(x, first, '[{:}] {:.2f}'.format(top_K_indices[K], top_K_vals[K]), fontsize=6, c=colors[K], alpha=0.5,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontfamily='sans-serif', fontstretch='ultra-condensed', fontweight='bold')
                        first += increment
                    w_pad   = .2
                    h_pad   = .2
                    pad     = .05

            if FLAGS.vis_cons_mode == 'train' and vis_train_cons:   vis_connections(train_cons)
            if FLAGS.vis_cons_mode == 'eval' and vis_eval_cons:     vis_connections(eval_cons)

        #if channels == 4: 
        #  disp = disp.reshape(imgW,imgH,channels)[:,:,[0,1,2]]; 
        #  channels = 3
        ax_.imshow(disp.reshape(imgH, imgW, channels) if channels != 1 else disp.reshape(imgH,imgW), vmin=refmin, vmax=refmax, cmap=cm.bone)

        if FLAGS.vis_pis == True:
            ax_.text(-5, 1, "%.03f" % (pi_), fontsize=5, c="black", bbox=dict(boxstyle="round", fc=(1, 1, 1), ec=(.5, .5, .5)))

        ax_.set_aspect('auto')
        ax_.tick_params( # disable labels and ticks
                axis        = 'both',
                which       = 'both',
                bottom      = False ,
                top         = False ,
                left        = False ,
                right       = False ,
                labelbottom = False ,
                labelleft   = False ,
        )

    #plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0.5,hspace=0.5)
    plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    plt.savefig(save_path, transparent=True)

    return save_path


if __name__ == "__main__":
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    ### python3 cl_replay/utils/vis.py --prefix "" --out $protos_t1_out --mu_file $protos_t1_mus_file --sigma_file $protos_t1_sigmas_file --pi_file $protos_t1_pis_file ###

    parser = ArgumentParser()

    parser.add_argument("--channels",       default = 1,               type=int,   help = "If u r visualizing centroids that come from color images (SVHN, Fruits), please specify 3 here!")
    parser.add_argument("--y",              default = 0,               type=int,   help = "PatchY index for convGMMs")
    parser.add_argument("--x",              default = 0,               type=int,   help = "PatchX index for convGMMs")
    parser.add_argument("--l",              default = 0,               type=int,   help = "row of MFA loading matrix")
    parser.add_argument("--what",           default = "mus",           type=str,   choices=["mus2D", "mus","precs_diag","organizedWeights","loadingMatrix"],  help="Visualize centroids or precisions°")
    parser.add_argument("--prefix",         default = "gmm_layer_",    type=str,   help="Prefix for file names")
    parser.add_argument("--vis_pis",        default = False,           type=eval,  help="True or False depending on whether you want the weights drawn on each component")
    parser.add_argument("--vis_cons",       default = True,            type=eval,  help="True or False depending on whether you want the top K component/classification connectivities drawn on each component")
    parser.add_argument("--vis_cons_mode",  default = "eval",          type=str,   help="visualize test/train component mapping")
    parser.add_argument("--cur_layer_hwc",  default = [-1, -1, -1],    type=int,   nargs = 3, help = "PatchX index for convGMMs")
    parser.add_argument("--prev_layer_hwc", default = [-1, -1, -1],    type=int,   nargs = 3, help = "PatchX index for convGMMs")
    parser.add_argument("--filter_size",    default = [-1, -1],        type=int,   nargs = 2, help = "PatchX index for convGMMs")
    parser.add_argument("--proto_size",     default = [-1, -1],        type=int,   nargs = 2, help = "PatchX index for convGMMs")
    parser.add_argument("--clip_range",     default = [-100., 100.],   type=float, nargs = 2, help = "clip display to this range")
    parser.add_argument("--disp_range",     default = [-100., 100.],   type=float, nargs = 2, help = "clip display to this range")

    parser.add_argument("--out",            default = "./vis.png",   type=str,   help="output file name")
    parser.add_argument("--dataset_file",   default = "",                       type=str,   help="for mus2d mode")
    parser.add_argument("--mu_file",        default = "mus.npy",                type=str,   help="usually keep default value, path and prefix are prepended")
    parser.add_argument("--sigma_file",     default = "sigmas.npy",             type=str,   help="duh")
    parser.add_argument("--pi_file",        default = "pis.npy",                type=str,   help="duh2")

    parser.add_argument("--img_sequence",   default = False,                     type=eval,  help="flag if visualizing an image sequence")
    parser.add_argument("--data_path",  required=True,    type=str,   help="path to sigma,mus and pi files")
    parser.add_argument("--only_last_epoch",default = False,                    type=eval,  help="only visualize last epoch protos & test connectivities")

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.data_path): os.makedirs(FLAGS.data_path)
    if not os.path.exists(FLAGS.out): os.makedirs(FLAGS.out)

    ### RUN WITH:
    ### python3 $HOME/scclv2/src/cl_replay/utils/vis.py --data_path "$HOME/scclv2/src/results/protos_T1" --prefix "vargen_gmm_8x8_maska_L2_GMM_CL_" --out "$HOME/scclv2/src/results/protos_out/vargen_exp_T1"

    if FLAGS.img_sequence:
        # VIS DATA
        if os.path.exists(FLAGS.data_path):                 # ./results/protos_T1/
            epoch_dirs = os.listdir(FLAGS.data_path)
            proto_img_list = [0] * (len(epoch_dirs)-1)
            for i in range(0, len(epoch_dirs)-1):               # E_0 ... E_N, where N is num of training epochs
                if FLAGS.only_last_epoch:
                    sub_dir_path = FLAGS.data_path + f'/E{(len(epoch_dirs)-2)}'
                    i = (len(epoch_dirs)-2)
                else:
                    sub_dir_path = FLAGS.data_path + f'/E{i}'
                if os.path.isdir(sub_dir_path) and os.path.exists(sub_dir_path):
                    prefix = sub_dir_path + '/' + FLAGS.prefix
                    proto_img_list[i] = vis_img_data(FLAGS, prefix_path=prefix, prefix=FLAGS.prefix, suffix=i)

                if FLAGS.only_last_epoch: break

        # IMAGEIO CREATE GIF
        if not FLAGS.only_last_epoch:
            images = list()
            for i in range(0, len(proto_img_list)):
                img_name = proto_img_list[i]
                if img_name.endswith('mus.png'):
                    images.append(imageio.imread(img_name))

            gif_path = Path(FLAGS.out + '/protos.gif')
            imageio.mimwrite(gif_path, images, duration=1)

    else:
        if os.path.exists(FLAGS.data_path):
            prefix = FLAGS.data_path + '/' + FLAGS.prefix
            vis_img_data(FLAGS, prefix_path=prefix, prefix=FLAGS.prefix)
