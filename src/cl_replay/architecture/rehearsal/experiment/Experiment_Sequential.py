import os
import sys
import math
import itertools
import numpy        as np
import tensorflow   as tf
import matplotlib.pyplot as plt

from importlib      import import_module
from importlib.util import find_spec

from cl_replay.api.utils                        import log, helper
from cl_replay.api.experiment                   import Experiment_Replay
from cl_replay.api.model                        import DNN
from cl_replay.api.parsing                      import Kwarg_Parser

from cl_replay.architecture.rehearsal.adaptor   import Rehearsal_Adaptor
from cl_replay.architecture.rehearsal.buffer    import Rehearsal_Buffer

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_Sequential(Experiment_Replay):
    """ Defines a basic sequential experiment. """

    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor = Rehearsal_Adaptor(**self.parser.kwargs)

        self.model_type             = self.parser.add_argument('--model_type',      type=str, default='dnn', choice=['dnn', 'cnn'], help='model architecture, dnn is configurable, cnn is fixed.')
        self.num_layers             = self.parser.add_argument('--num_layers',      type=int, default=2, help='number of dense layers for the dnn.')
        self.num_units              = self.parser.add_argument('--num_units',       type=int, default=128, help='number of units for the dense layers of a dnn.')
        self.add_dropout            = self.parser.add_argument('--add_dropout',     type=str, default='no', choice=['yes', 'no'], help='add dropout layers after hidden layers.')
        self.dropout_rate           = self.parser.add_argument('--dropout_rate',    type=float, default=0.3, help='dropout rate.')
        self.freeze_n_layers        = self.parser.add_argument('--freeze_n_layers', type=int, default=0, help='number of lower layers to freeze for fine-tuning.')

        self.extra_eval             = self.parser.add_argument('--extra_eval', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')

    def _init_variables(self):
        Experiment_Replay._init_variables(self)

    #-------------------------------------------- MODEL CREATION/LOADING/SAVING
    def create_model(self):
        model_inputs  = tf.keras.Input(self.get_input_shape())
        if self.model_type == 'dnn':
            l_ = tf.keras.layers.Flatten()(model_inputs)
            for i in range(0, self.num_layers):
                l_ = tf.keras.layers.Dense(self.num_units, activation="relu")(l_)
                if self.add_dropout == 'yes':
                    l_ = tf.keras.layers.Dropout(rate=self.dropout_rate)(l_)
        if self.model_type == 'cnn':
            # conv_1  = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), name="conv_1", padding="same", activation="relu")(model_inputs)
            # pool_1  = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_1)
            # conv_2  = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), name="conv_2", padding="same", activation="relu")(pool_1)  # INFO: alternative kernel_size (3, 3)
            # pool_2  = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_2)
            # ---------- LeNet
            conv_1  = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(2, 2), name="conv_1", padding="same", activation="relu")(model_inputs)
            pool_1  = tf.keras.layers.AveragePooling2D (pool_size=(2, 2), strides=(2, 2), padding="valid")(conv_1)
            conv_2  = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), name="conv_2", padding="valid", activation="relu")(pool_1)  # INFO: alternative kernel_size (3, 3)
            pool_2  = tf.keras.layers.AveragePooling2D (pool_size=(2, 2), strides=(2, 2), padding="valid")(conv_2)
            flat    = tf.keras.layers.Flatten()(pool_2)
            l_      = tf.keras.layers.Dense(120, activation="relu")(flat)
            if self.add_dropout == 'yes': l_ = tf.keras.layers.Dropout(rate=self.dropout_rate)(l_)
            l_      = tf.keras.layers.Dense(84, activation="relu")(l_)
            if self.add_dropout == 'yes': l_ = tf.keras.layers.Dropout(rate=self.dropout_rate)(l_)
        model_outputs  = tf.keras.layers.Dense(name="prediction", units=self.num_classes, activation="softmax")(l_)

        model = DNN(inputs=model_inputs, outputs=model_outputs, **self.flags)
        model.compile(run_eagerly=True, optimizer=None)
        model.summary()
        
        return model
    
    
    def load_model(self):
        """ executed before training """
        super().load_model() # load or create model

        self.adaptor.set_input_dims(self.h, self.w, self.c, self.num_classes)
        self.adaptor.set_model(self.model)

        # setting callbacks manually is only needed when we train in "batch mode" instead of using keras' model.fit()
        if self.train_method == 'batch':
            for cb in self.train_callbacks: cb.set_model(self.model)
            for cb in self.eval_callbacks:  cb.set_model(self.model)
        else: return
            #TODO: add wandb support via keras callback
            
    
    def get_input_shape(self):
        return self.h, self.w, self.c
    
    
    def feed_sampler(self, task, current_data):
        cur_xs, cur_ys = current_data
        
        self.sampler.add_subtask(xs=cur_xs, ys=cur_ys)
        self.sampler.set_proportions([1.])
        
        if self.ml_paradigm == 'supervised':
            _, self.class_freq = self.calc_class_freq(total_classes=self.DAll, targets=cur_ys, mode='ds')
            self.adaptor.set_class_freq(class_freq = self.class_freq)
        else: self.class_freq = None
        
        self.train_steps = self.get_task_iters()
        log.info(f'setting up "steps_per_epoch"... iterations for current task (generated samples): {self.train_steps},')
        log.info(f'\tadded generated data for deletion task t{task} to the replay_sampler...')


    def before_task(self, task, **kwargs):
        dnn = self.adaptor.model
        if task == 2:
            current_lr = dnn.optimizer.learning_rate
            dnn.optimizer.learning_rate.assign(current_lr*0.1)

            freezed_layers = 0
            for l in dnn.layers[1:]:
                if ('dense' in l.name or 'conv' in l.name) and (
                    freezed_layers < self.freeze_n_layers):
                    freezed_layers += 1
                    l.trainable = False     

        self.sampler.reset()
        self.feed_sampler(task, self.train_set)


    def _test(self, task):
        super()._test(task)
        if self.extra_eval != []:
            _, self.eeval_test, _, self.eeval_amount = self.dataset.get_dataset(self.extra_eval, task_info=None)
            self.model.test_task = f'EXTRA'  # test task identifier
            log.info(f'\t[TEST] -> {self.model.test_task}({np.unique(np.argmax(self.eeval_test[1], axis=-1))})')
            self.model.evaluate(x=self.eeval_test[0], y=self.eeval_test[1],
                                batch_size=self.test_batch_size,
                                steps=(self.eeval_amount//self.test_batch_size),
                                callbacks=self.eval_callbacks,
                                verbose=self.verbosity,
                                return_dict=True)


    def after_task(self, task, **kwargs):
        super().after_task(task, **kwargs)
        if self.model_type == 'cnn':
            self.visualize_filters(n_channels=3, n_filters=6, layer_idx=1)
            self.visualize_filters(n_channels=6, n_filters=16, layer_idx=3)
            # ----------- get a random image
            test_data   = self.test_sets[1][0]
            rnd_img     = test_data[np.random.randint(0, test_data.shape[0])]
            rnd_img     = np.expand_dims(rnd_img, axis=0)  # reshape to (1,H,W,C)
            
            fig, axes = plt.subplots(1, 1, figsize=(3, 3))
            im = axes.imshow(rnd_img[0])
            axes.set_xticks([])
            axes.set_yticks([])
            plt.savefig(f'{self.vis_path}/input_img.svg', 
                        dpi=300., pad_inches=0.1, facecolor='auto', format='svg')
            plt.close()
            # -----------
            self.visualize_fmaps(rnd_img, layer_idx=1, n_rows=2, n_cols=3, figsize=(9,6))  # C1 6 filter
            self.visualize_fmaps(rnd_img, layer_idx=3, n_rows=4, n_cols=4, figsize=(6,6))  # C2 16 filter
        
    
    def visualize_filters(self, n_channels, n_filters, layer_idx):
        """ Visualize N CNN filters. """
        sel_layer = self.model.layers[layer_idx]
        filters, _ = sel_layer.get_weights()
        filter_min, filter_max = filters.min(), filters.max()
        # kernel_height, kernel_widt, input_channels, output_channels
        filters = (filters - filter_min) / (filter_max - filter_min) # normalize [0,1]

        log.debug(f'visualizing CNN filters for layer: {sel_layer.name}, filter shape: {filters.shape}.')
        fig, axes = plt.subplots(n_channels, n_filters, figsize=(n_filters, n_channels))
        for i in range(n_channels):
            for j in range(n_filters):
                f = filters[:, :, :, j] # get j-th filter
                # always visualize 0-th channel
                axes[i,j].imshow(f[:, :, i], cmap='gray') # visualize i-th channel
                
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
                axes[i,j].axis('off')
                axes[i,j].set_aspect('equal')

        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(
            f'{self.vis_path}/L{layer_idx}_cnn-filters.svg', 
            dpi=300, pad_inches=0.1, bbox_inches='tight', facecolor='auto', edgecolor='auto', format='svg'
        )
        plt.close()


    def visualize_fmaps(self, img, layer_idx, n_rows, n_cols, figsize):
        """ Visualize CNN feature maps."""
        layer_name  = self.model.layers[layer_idx].name
        layer_out   = self.model.get_layer(layer_name).output
        sub_model   = DNN(inputs=self.model.input, outputs=layer_out, **self.flags)
        fmap        = sub_model.predict(img)
        print(fmap.shape)

        ix = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        for i in range(n_rows):
            for j in range(n_cols):
                axes[i,j].imshow(fmap[0, :, :, ix-1], cmap='gray')
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
                axes[i,j].axis('off')
                axes[i,j].set_aspect('equal')
                ix += 1

        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(
            f'{self.vis_path}/L{layer_idx}_cnn-fmap.svg', 
            dpi=300, 
            #pad_inches=0.1,
            bbox_inches='tight', facecolor='auto', edgecolor='auto', format='svg'
        )
        plt.close()


if __name__ == '__main__':
    Experiment_Sequential().run_experiment()