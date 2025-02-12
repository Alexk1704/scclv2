import math
import os
import numpy            as np
import pandas           as pd

from pathlib            import Path
from tensorflow         import keras
from keras.callbacks    import Callback

from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log



def rm_dir(path):
    dir_ = Path(path)
    for sub in dir_.iterdir():
        if sub.is_dir():
            rm_dir(sub)
        else:
            sub.unlink()
    dir_.rmdir()



class Log_Protos(Callback):
    ''' Save trainables as .npy files, gets called either on epoch or train end. '''

    def __init__(self, **kwargs):
        super(Log_Protos, self).__init__()

        parser = Kwarg_Parser(**kwargs)
        self.save_protos        = parser.add_argument('--save_protos', type=str, choices=['on_epoch', 'on_train_end'], default='on_train_end')
        self.log_each_n_protos  = parser.add_argument('--log_each_n_protos', type=int, default=1)
        self.vis_path           = parser.add_argument('--vis_path', type=str, required=True)
        if os.path.isabs(self.vis_path) == False: log.error("--vis_path must be absolute!")
        self.exp_id             = kwargs.get('exp_id', None)
        
        self.test_task, self.train_task = 0, int(kwargs.get('load_task', 0))
        self.current_epoch  = 0; self.test_batch     = 0

        self.saved_protos   = []


    def save(self, saved_protos):
        # info: need this for vis script, could get rid of task information here ofc!
        for t, e, vars in saved_protos:
            save_dir = self.vis_path + f'/{self.exp_id}_protos_T{t}/E{e}'
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            for vname, v in vars:
                fname = f'{save_dir}/{self.exp_id}_{vname}.npy'
                np.save(fname, v)
    

    #-------------------------------------------- START: CALLBACK FUNCTIONS
    #-------------------------- TRAIN
    def on_train_begin(self, logs=None):      
        self.train_task += 1
        self.current_epoch = 0
        self.test_batch = 0


    def on_train_end(self, logs=None):
        if self.save_protos != "on_train_end": return
        self.save_vars(self.current_epoch)


    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        if self.save_protos != "on_epoch": return
        if (epoch % self.log_each_n_protos) == 0:
            self.save_vars(epoch)

    #-------------------------- TEST
    def on_test_begin(self, logs=None):
        if self.model.test_task:
            self.test_task = self.model.test_task
        else:
            self.test_task += 1
    #-------------------------------------------- END: CALLBACK FUNCTIONS


    def save_vars(self, epoch=1):
        # accumulate protos over epochs, only dump to storage when training ends!
        layer_vars = []
        for layer in self.model.layers:
            if hasattr(layer, 'is_layer_type'):
                if layer.is_layer_type('GMM_Layer'):
                    for v in layer.trainable_variables:
                        vname = f'{layer.name}_{v.name}'
                        layer_vars.append((vname, v.numpy()))
        
        self.save([[self.train_task, epoch, layer_vars]])
