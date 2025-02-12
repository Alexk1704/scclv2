import sys
import numpy       as np
import tensorflow  as tf
import math 

from cl_replay.api.experiment    import Experiment, Experiment_Replay
from cl_replay.api.parsing       import Kwarg_Parser
from cl_replay.api.utils         import log

from cl_replay.architecture.ewc.model.EWC  import EWC

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_EWC(Experiment_Replay):


    def _init_parser(self, **kwargs):
        Experiment._init_parser(self, **kwargs)
        self.flags = kwargs

        self.mode              = self.parser.add_argument('--mode',                 type=str,   default='ewc',              choices=['ewc', 'mean_imm', 'mode_imm']   , help='what type of imm. [default="ewc", "mean_imm"]')
        self.imm_transfer_type = self.parser.add_argument('--imm_transfer_type',    type=str,   default='weight_transfer',  choices=['L2_transfer', 'weight_transfer'], help='what type of imm?')
        self.imm_alpha         = self.parser.add_argument('--imm_alpha',            type=float, default=0.5,                help='balancing parameter')
        
        self.model_type        = self.parser.add_argument('--model_type',           type=str,   default='dnn',              choices=['dnn', 'cnn'], help='class to load form module "model"')
        self.num_layers        = self.parser.add_argument('--num_layers',           type=int,   default=2,                  help='number of dense layers for the dnn.')
        self.num_units         = self.parser.add_argument('--num_units',            type=int,   default=128,                help='number of units for the dense layers of a dnn.')
        
        self.extra_eval        = self.parser.add_argument('--extra_eval',           post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')
        self.forgetting_tasks  = self.parser.add_argument('--forgetting_tasks',     post_process=Kwarg_Parser.make_list, type=int, default=[], help='define forgetting tasks.')
        self.del_dict = {}
        for i, cls in enumerate(self.task_list):
            if (i+1) in self.forgetting_tasks:
                self.del_dict.update({(i+1) : cls})
        log.debug(f'setting up deletion dict: {self.del_dict}')
        self.prev_tasks = []


    def create_model(self):
        # TODO: see Experiment_DGR.py, allow layer creation via bash file...
        model_inputs  = tf.keras.Input(self.get_input_shape())
        if self.model_type == 'dnn':
            l_ = tf.keras.layers.Flatten()(model_inputs)
            for i in range(0, self.num_layers):
                l_ = tf.keras.layers.Dense(self.num_units, activation="relu")(l_)
                # l_ = tf.keras.layers.Dropout(rate=0.3)(l_)
        if self.model_type == 'cnn':
            conv_1  = tf.keras.layers.Conv2D(32, (3, 3), (2, 2), padding="same", activation="relu")(model_inputs)
            pool_1  = tf.keras.layers.MaxPool2D((2, 2))(conv_1)
            conv_2  = tf.keras.layers.Conv2D(64, (3, 3), (2, 2), padding="same", activation="relu")(pool_1)  # INFO: alternative kernel_size (3, 3)
            pool_2  = tf.keras.layers.MaxPool2D((2, 2))(conv_2)
            flat    = tf.keras.layers.Flatten()(pool_2)
            dense_1 = tf.keras.layers.Dense(512, activation="relu")(flat)
            drop_1  = tf.keras.layers.Dropout(rate=0.3)(dense_1)
            dense_2 = tf.keras.layers.Dense(256, activation="relu")(drop_1)
            l_      = tf.keras.layers.Dropout(rate=0.3)(dense_2)
        model_outputs = tf.keras.layers.Dense(name="prediction", units=self.num_classes)(l_)

        self.model = EWC(inputs=model_inputs, outputs=model_outputs, **self.flags)
        self.model.compile()
        self.model.summary()

        self.model.set_parameters(**{'tasks' : self.tasks})

        return self.model


    def get_input_shape(self):
        return self.h, self.w, self.c
    

    def feed_sampler(self, task, current_data):
        cur_xs, cur_ys = current_data
        self.sampler.add_subtask(xs=cur_xs, ys=cur_ys)
        self.sampler.set_proportions([1.])
        
        self.train_steps = self.get_task_iters()
        
        log.info(f'setting up "steps_per_epoch"... iterations for current task T{task}: {self.train_steps},')
    
    
    def prepare_forgetting(self, task):
        if self.del_dict != {}:
            self.forget_classes = []
            for _, del_cls in self.del_dict.items():
                self.forget_classes.extend(del_cls)
    
    
    def before_task(self, task, **kwargs):
        """ generates datasets of past tasks if necessary and resets model layers """
        if task in self.del_dict: return
        
        if self.mode in ['mean_imm', 'mode_imm'] and self.imm_transfer_type == 'L2_transfer': # for weight-transfer, we keep previous weights
            self.model.randomize_weights()
        
        if task > 1:
            # NOTE: disable FIM for specific classes
            for i, f_task in enumerate(self.forgetting_tasks):
                if f_task <= task: # task to forget was learned in the past, so forget it ASAP
                    for p_cls in self.tasks[f_task]:
                        log.info(f'disabling FIM and stored params for id: {p_cls}.')
                        self.model.coeffs[p_cls].assign(tf.constant(0.0))
                    del self.forgetting_tasks[i]
                    
        model_upd_kwargs = {'current_task' : task}
        self.model.set_parameters(**model_upd_kwargs)

        current_train_set = self.train_set
        self.sampler.reset()
        self.feed_sampler(task, current_train_set)
    
    
    def train_on_task(self, task):
        if task in self.del_dict:  # skip train
            for t_cb in self.train_callbacks:
                t_cb.on_train_begin()
                t_cb.on_train_end()
        else:
            super().train_on_task(task)


    def _test(self, task):
        # if task in self.del_dict: return  # skip test for forg. task
        
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
        if task == len(self.tasks)-1: return
        
        super().after_task(task, **kwargs)
        self.prepare_forgetting(task+1)  # look-ahead if there are any classes to delete!
        
        self.prev_tasks.append(task)
        
        if task in self.del_dict: return
        
        # ----- FIM calculation
        self.model.init_fim_struct()
        
        if self.del_dict != {}:
            xs, ys = self.train_set
            for fim_class in self.tasks[task]:
                ys_mask = np.isin(ys.argmax(axis=-1), fim_class)
                xs_ = xs[ys_mask]
                ys_ = ys[ys_mask]
                if xs_.shape[0] > 0:
                    ds = tf.data.Dataset.from_tensor_slices((xs_,ys_)).batch(self.batch_size, drop_remainder=True)

                    iters = 0
                    for x_, _ in ds:
                        iters += 1
                        self.model.compute_fim(x_)
                        # if iters % 10 == 0: print(iters)
                    # print("iters: ", iters)
                    self.model.finalize_fim(iters, self.batch_size, id=fim_class)
        else:
            pass
            # TODO: add (default) task-based EWC

if __name__ == '__main__':
    Experiment_EWC().run_experiment()
