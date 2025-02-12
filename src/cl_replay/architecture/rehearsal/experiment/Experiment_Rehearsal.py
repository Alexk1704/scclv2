import os
import sys
import math
import itertools
import numpy        as np
import tensorflow   as tf

from importlib      import import_module
from importlib.util import find_spec

from cl_replay.api.data.Dataset                 import visualize_data
from cl_replay.api.utils                        import log, helper
from cl_replay.api.experiment                   import Experiment_Replay
from cl_replay.api.model                        import DNN
from cl_replay.api.parsing                      import Kwarg_Parser

from cl_replay.architecture.rehearsal.adaptor   import Rehearsal_Adaptor
from cl_replay.architecture.rehearsal.buffer    import Rehearsal_Buffer

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_Rehearsal(Experiment_Replay):
    """ Defines a basic experience replay experiment utilizing a buffer structure. """

    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor = Rehearsal_Adaptor(**self.parser.kwargs)

        self.model_type             = self.parser.add_argument('--model_type', type=str, default='dnn', choice=['dnn', 'cnn'], help='model architecture, dnn is configurable, cnn is fixed.')
        self.num_layers             = self.parser.add_argument('--num_layers', type=int, default=2, help='number of dense layers for the dnn.')
        self.num_units              = self.parser.add_argument('--num_units', type=int, default=128, help='number of units for the dense layers of a dnn.')
        self.add_dropout            = self.parser.add_argument('--add_dropout',     type=str, default='no', choice=['yes', 'no'], help='add dropout layers after hidden layers.')
        self.dropout_rate           = self.parser.add_argument('--dropout_rate',    type=float, default=0.3, help='dropout rate.')

        self.extra_eval             = self.parser.add_argument('--extra_eval', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')
        self.forgetting_mode        = self.parser.add_argument('--forgetting_mode', type=str, default='separate', choices=['separate', 'mixed'], help='switch between forgetting modes.')
        self.forg_sample_topdown    = self.parser.add_argument('--forg_sample_topdown', type=str, default='no', choices=['no', 'yes'], help='turn on/off conditional sampling for forgetting phase.')
        self.forg_sample_variants   = self.parser.add_argument('--forg_sample_variants', type=str, default='no', choices=['no', 'yes'], help='turn on/off variant sampling for forgetting phase.')
        self.forgetting_tasks       = self.parser.add_argument('--forgetting_tasks', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define forgetting tasks.')
        
        self.del_dict = {}
        for i, cls in enumerate(self.task_list):
            if (i+1) in self.forgetting_tasks:
                self.del_dict.update({(i+1) : cls})
        log.debug(f'setting up deletion dict: {self.del_dict}')
                
        if self.forgetting_mode == 'mixed' and self.num_tasks in self.del_dict:
            log.error('mixed forgetting task can not be the last task in the training sequence, add a learning task to the end!')
            sys.exit(0)

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
        self.adaptor.set_generator(Rehearsal_Buffer(data_dims=self.adaptor.get_input_dims()))
        # setting callbacks manually is only needed when we train in "batch mode" instead of using keras' model.fit()
        if self.train_method == 'batch':
            for cb in self.train_callbacks: cb.set_model(self.model)
            for cb in self.eval_callbacks:  cb.set_model(self.model)
        else: return
            #TODO: add wandb support via keras callback


    def get_input_shape(self):
        return self.h, self.w, self.c

    #-------------------------------------------- DATA HANDLING
    def generate(self, task, data, gen_classes, real_classes, **kwargs):
        if data is None: xs = None
        else: xs, _ = data
        if self.forgetting_mode == 'separate' and task in self.del_dict: xs = None
        return self.adaptor.sample(task, xs, gen_classes, real_classes, **kwargs)


    def replace_subtask_data(self, buffer_samples):
        """ replace subtask data (shapes have to coincide) of the sampler """
        x_buf, y_buf = buffer_samples
        self.sampler.replace_subtask_data(subtask_index=-1, x=x_buf, y=y_buf)
        log.debug(f'replaced subtask data with buffer samples: {np.unique(np.argmax(y_buf, axis=1), return_counts=True)}')

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        if task > 1:
            if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip this phase
                log.debug(f'skipping this training phase since forgetting_mode == "mixed"')
                return
            #--------------------------------------- FORGETTING PREP  
            self.past_tasks += 1
            self.past_classes, self.real_classes = [], self.tasks[task]         
            past_classes = set()
            for p_class in self.tasks[1:task]:
                past_classes.update(p_class)
            
            if self.del_dict != {}:
                invalid_tasks = set() 
                for f_task, f_class in self.del_dict.items():
                    if f_task <= task:
                        invalid_tasks.update(f_class)
                self.past_classes = [x for x in past_classes if x not in invalid_tasks]             
            else:
                self.past_classes = [x for x in past_classes]    
            log.debug(f"past classes to keep: {self.past_classes}")
            #--------------------------------------- LOSS BALANCING
            if self.loss_coef == 'class_balanced':
                past_cls = len(self.past_classes)
                current_cls = len(self.tasks[task])
                r_s_c = float(current_cls / (past_cls + current_cls)) 
                g_s_c = 1. - r_s_c
            elif self.loss_coef == 'task_balanced':
                r_s_c = float((1. / self.past_tasks))
                g_s_c = 1. - r_s_c
            elif self.loss_coef == 'off':
                r_s_c = 1.; g_s_c = 1.
            # ----------
            self.sampler.reset()
        
            if self.forgetting_mode == 'mixed' and (task-1) in self.del_dict:
                self.adaptor.forget(self.del_dict[task-1])
            if self.forgetting_mode == 'separate' and task in self.del_dict:
                self.adaptor.forget(self.del_dict[task])
            
            self.generated_dataset = self.generate(task, data=self.train_set, gen_classes=self.past_classes, real_classes=self.real_classes)
            
            visualize_data(self.generated_dataset[0][:64], self.generated_dataset[1][:64], self.model.vis_path, f'gen_T{task}')
            
            log.debug(f'using the following sample weights: gen. data -> {g_s_c} / real data -> {r_s_c}')
            self.sampler.real_sample_coef = r_s_c
            self.sampler.gen_sample_coef = g_s_c

        self.feed_sampler(task, self.train_set)
        self.adaptor.before_subtask(
            task, 
            total_samples=self.samples_train_D_ALL
        )


    def after_task(self, task, **kwargs):
        """ select M samples from the task data and saves them. """
        super().after_task(task, **kwargs)
        
        # skip storage of samples for designated forgetting phases
        if (self.forgetting_mode == 'separate' and task in self.del_dict) or (
            self.forgetting_mode == 'mixed' and task in self.del_dict
        ): return

        self.adaptor.after_subtask(
            task, 
            task_classes=self.tasks[task],
            task_data=self.train_set
        )

   #--------------------------------------- OVERWRITE

    def feed_sampler(self, task, current_data):
        if self.forgetting_mode == 'separate' and task in self.del_dict:
            # only feed the sampler with generated data in case deletion task shall be handled separately
            gen_xs, gen_ys = self.generated_dataset
            
            self.sampler.add_subtask(xs=gen_xs, ys=gen_ys)
            self.sampler.set_proportions([1.])
            
            if self.ml_paradigm == 'supervised':
                _, self.class_freq = self.calc_class_freq(total_classes=self.DAll, targets=gen_ys, mode='ds')
                self.adaptor.set_class_freq(class_freq = self.class_freq)
            else: self.class_freq = None
            
            self.train_steps = gen_xs.shape[0] // self.batch_size
            
            log.info(f'setting up "steps_per_epoch"... iterations for current task (generated samples): {self.train_steps},')
            log.info(f'\tadded generated data for deletion task t{task} to the replay_sampler...')
        elif self.forgetting_mode == 'mixed' and task in self.del_dict:
            return
        else:
            super().feed_sampler(task, current_data)           


    def train_on_task(self, task):
        if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip train
            for t_cb in self.train_callbacks:
                t_cb.on_train_begin()
                t_cb.on_train_end()
        else:
            super().train_on_task(task)


    def _test(self, task):
        # if self.forgetting_mode == 'mixed' and task in self.del_dict: return  # skip test
        
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


if __name__ == '__main__':
    Experiment_Rehearsal().run_experiment()
