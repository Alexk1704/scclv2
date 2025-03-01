import os
import sys
import math
import itertools
import numpy        as np
import tensorflow   as tf

from importlib      import import_module
from importlib.util import find_spec
from matplotlib     import plt as plt

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
            conv_1  = tf.keras.layers.Conv2D(32, (3, 3), (2, 2), name="conv_1", padding="same", activation="relu")(model_inputs)
            pool_1  = tf.keras.layers.MaxPool2D((2, 2))(conv_1)
            conv_2  = tf.keras.layers.Conv2D(64, (3, 3), (2, 2), name="conv_2", padding="same", activation="relu")(pool_1)  # INFO: alternative kernel_size (3, 3)
            pool_2  = tf.keras.layers.MaxPool2D((2, 2))(conv_2)
            flat    = tf.keras.layers.Flatten()(pool_2)
            l_      = tf.keras.layers.Dense(512, activation="relu")(flat)
            if self.add_dropout == 'yes': l_ = tf.keras.layers.Dropout(rate=self.dropout_rate)(l_)
            l_      = tf.keras.layers.Dense(256, activation="relu")(l_)
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
        
        print("HI")


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
            self.visualize_filters(32, 1)
        
    
    def visualize_filters(self, n_filters, layer_idx):
        """ Visualize first N filters of i-th conv-layer (credits to MrNouman). """
        filters, _ = self.model.layers[layer_idx].get_weights()
        filter_min, filter_max = filters.min(), filters.max()
        filters = (filters - filter_min) / (filter_max - filter_min) # normalize [0,1]
        ix = 1
        for i in range(n_filters):
            f = filters[:, :, :, i]
            for j in range(self.c):
                ax = plt.subplot(n_filters, self.c, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(f[:, :, j])
                ix += 1
        plt.show()


    def visualize_fmaps(self, base_model, block_indices):
        """ Visualize feature maps derived from conv-blocks (credits to MrNouman).
            Arguments need to be integers of last conv-layer per visualized block.
        """
        outputs = [base_model.layers[i].output for i in block_indices]
    
        rnd_img = self.raw_tr_xs[np.random.randint(0, self.raw_tr_xs.shape[0])]
        rnd_img = np.expand_dims(rnd_img, axis=0) # reshape to (1,H,W,C)

        feature_maps = self.model.predict(rnd_img)

        sqr = 8
        for fmap in feature_maps:
            ix = 1
            for _ in range(sqr):
                for _ in range(sqr):
                    ax = plt.subplot(sqr, sqr, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.imshow(fmap[0, :, :, ix-1]) #cmap='gray'
                    ix += 1
            plt.show()
    
if __name__ == '__main__':
    Experiment_Sequential().run_experiment()