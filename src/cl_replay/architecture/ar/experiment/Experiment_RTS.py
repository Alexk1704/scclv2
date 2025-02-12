import os
import sys
import math
import itertools
import numpy       as np

from importlib import import_module
from importlib.util import find_spec

from cl_replay.api.data.Dataset             import visualize_data
from cl_replay.api.parsing                  import Kwarg_Parser
from cl_replay.api.utils                    import log, helper
from cl_replay.api.experiment               import Experiment_Replay
from cl_replay.architecture.ar.experiment   import Experiment_GMM
from cl_replay.architecture.ar.adaptor      import AR_Supervised
from cl_replay.architecture.ar.generator    import AR_Generator


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class Experiment_RTS(Experiment_Replay):
    ''' Defines an experiment for AR. '''

    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.extra_eval             = self.parser.add_argument('--extra_eval', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define classes for extra eval at the end of training.')
        self.forgetting_mode        = self.parser.add_argument('--forgetting_mode', type=str, default='separate', choices=['separate', 'mixed'], help='switch between forgetting modes.')
        self.forg_sample_topdown    = self.parser.add_argument('--forg_sample_topdown', type=str, default='no', choices=['no', 'yes'], help='turn on/off conditional sampling for forgetting phase.')
        self.forg_sample_variants   = self.parser.add_argument('--forg_sample_variants', type=str, default='no', choices=['no', 'yes'], help='turn on/off variant sampling for forgetting phase.')
        self.forgetting_tasks       = self.parser.add_argument('--forgetting_tasks', post_process=Kwarg_Parser.make_list, type=int, default=[], help='define forgetting tasks.')
        self.forg_buffer            = self.parser.add_argument('--forg_buffer', type=int, default=100, help="how many samples per tasks are saved for VarGen + separate.")  
        
        self.del_dict = {}
        self.data_buffer = {}
        for i, cls in enumerate(self.task_list):
            if (i+1) in self.forgetting_tasks:
                self.del_dict.update({(i+1) : cls})
                    
        log.debug(f'setting up deletion dict: {self.del_dict}')
                
        if self.forgetting_mode == 'mixed' and self.num_tasks in self.del_dict:
            log.error('mixed forgetting task can not be the last task in the training sequence, add a learning task to the end!')
            sys.exit(0)

        self.adaptor = AR_Supervised(**self.parser.kwargs)


    def _init_variables(self):
        Experiment_Replay._init_variables(self)
        self.adaptor.model_outputs = self.model_outputs

    #-------------------------------------------- MODEL CREATION & LOADING
    def create_model(self):
        return Experiment_GMM.Experiment_GMM.create_model(self)


    def load_model(self):
        ''' Loads a model state and lets all needed dependancies. '''
        super().load_model()
        self.adaptor.set_input_dims(self.h, self.w, self.c, self.num_classes)
        self.adaptor.set_model(self.model)
        self.adaptor.set_generator(AR_Generator(model=self.model, data_dims=self.adaptor.get_input_dims()))
        # setting callbacks manually is only needed when we train in "batch mode" instead of using keras' model.fit()
        if self.train_method == 'batch':
            for cb in self.train_callbacks: cb.set_model(self.model)
            for cb in self.eval_callbacks:  cb.set_model(self.model)
        else: return


    def get_input_shape(self):
        return self.h, self.w, self.c

    #-------------------------------------------- DATA HANDLING
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
    
    
    def generate(self, task, data, gen_classes, real_classes, **kwargs):
        if data is None: xs = None
        else: xs, _ = data
        if self.forgetting_mode == 'separate' and task in self.del_dict: xs = None
        generate_labels = False
        if self.ml_paradigm == 'supervised': generate_labels = True
        
        return self.adaptor.generate(task, xs, gen_classes, real_classes, generate_labels, **kwargs)


    def replace_subtask_data(self, variants):
        ''' Replace the subtask data of the sampler. '''
        x_vars, y_vars = variants
        self.sampler.replace_subtask_data(subtask_index=-1, x=x_vars, y=y_vars)
        log.debug(f'replaced subtask data with variants: {np.unique(np.argmax(y_vars, axis=1), return_counts=True)}')


    def prepare_forgetting(self, task): 
        if self.del_dict != {}:
            self.del_classes = []
            for task_id, del_cls in self.del_dict.items():
                if task >= task_id:
                    self.del_classes.extend(del_cls)
                    
        self.past_tasks += 1
        self.past_classes, self.real_classes = [], self.tasks[task]
        past_classes = set()
        for p_class in self.tasks[1:task]: past_classes.update(p_class)
        
        if self.del_dict != {}:
            invalid_tasks = set() 
            for f_task, f_class in self.del_dict.items():
                if f_task <= task:
                    invalid_tasks.update(f_class)
            self.past_classes = [x for x in past_classes if x not in invalid_tasks]             
        else:
            self.past_classes = [x for x in past_classes]    
        log.debug(f"past classes to keep: {self.past_classes}")


    def calc_loss_coefs(self, task):        
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
            
        return r_s_c, g_s_c

    #-------------------------------------------- TRAINING/TESTING
    def before_task(self, task, **kwargs):
        self.adaptor.before_subtask(task)

        self.adaptor.model.current_task = task
        current_train_set = self.train_set

        if task > 1:
            if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip this phase
                log.debug(f'skipping this training phase since forgetting_mode == "mixed"')
                return
            
            # ---- FORGETTING PREP
            self.prepare_forgetting(task)
            
            # ---- LOSS BALANCING
            r_s_c, g_s_c = self.calc_loss_coefs(task)

            # ---- DATA GENERATION
            # -- GMM SETTINGS
            add_kwargs = {}
            forget = False          
            if self.forgetting_mode == 'mixed':
                if (task not in self.del_dict) and ((task-1) in self.del_dict):  # prev. task was a deletion task, use forg. sampling
                    forget = True
                    log.debug(f'mixed forgetting active for task T{task}, since task T{task-1} was an f-task.')
                    add_kwargs.update({
                        'sample_variants' : self.forg_sample_variants,
                        'sample_topdown' : self.forg_sample_topdown
                    })
                    if self.forg_sample_variants == 'no':  # use class-balanced loss weighting for cyclic and class-conditional sampling
                        if self.loss_coef == 'class_balanced':
                            past_cls = len(self.past_classes)
                            current_cls = len(self.tasks[task])
                            r_s_c = float(current_cls / (past_cls + current_cls)) 
                            g_s_c = 1. - r_s_c
                        else:
                            r_s_c = 1.; g_s_c = 1.
                else:  # loss balancing is off for standard replay since we use AR's variant gen.
                    r_s_c = 1.; g_s_c = 1.  
            if self.forgetting_mode == 'separate':
                if task in self.del_dict:
                    forget = True
                    log.debug(f'separate forgetting active for f-task T{task}.')
                    add_kwargs.update({
                        'sample_variants' : self.forg_sample_variants, 
                        'sample_topdown' : self.forg_sample_topdown,
                        'forget_classes' : self.tasks[task],
                        'forgetting' : True
                    })
           
            # NOTE: if using topdown/cyclic sampling for forgetting, this resets the sampling parameters when performing variant generation training for "non-forgetting tasks"
            if forget:  # change sampling parameters between default replay & forgetting
                if self.forg_sample_topdown == 'yes':  # class-conditional sampling
                    self.adaptor.change_sampling_params(sampling_I=-1 , sampling_S=3, somSigma_sampling='no')
                if self.forg_sample_topdown == 'no' and self.forg_sample_variants == 'no':  # cyclic sampling
                    self.adaptor.change_sampling_params(sampling_I=-2 , sampling_S=0, somSigma_sampling='no')
            else:
                self.adaptor.change_sampling_params(restore=True)

            # -- 
            self.generated_dataset = self.generate(
                task, data=current_train_set, gen_classes=self.past_classes, real_classes=self.real_classes, **add_kwargs)

            self.sampler.reset()
            
            # ---- FORGETTING
            def filter_gen_samples():
                log.debug(f'filtering generated classes for forgetting since no top_down sampling was used!')
                gen_xs, gen_ys  = self.generated_dataset
                pred_ys         = np.argmax(gen_ys, axis=1)
                pred_mask       = ~np.isin(pred_ys, self.del_classes)  # mask out classes to delete                              
                filtered_gen_xs = gen_xs[pred_mask]; filtered_gen_ys = gen_ys[pred_mask]
                diff_to_gen     = gen_xs.shape[0] - filtered_gen_xs.shape[0] 
                pad_indices     = np.random.choice(filtered_gen_xs.shape[0], size=diff_to_gen, replace=True)
                pad_xs          = filtered_gen_xs[pad_indices]; pad_ys = filtered_gen_ys[pad_indices] 
                gen_xs          = np.concatenate((filtered_gen_xs, pad_xs)); gen_ys = np.concatenate((filtered_gen_ys, pad_ys))
                self.generated_dataset  = gen_xs, gen_ys
                # visualize_data(pad_xs, None, f'{self.model.vis_path}/gen', f'filtered_T{task}_I{gen_it}')
                # ---- print class dist.
                classes, counts = np.unique(gen_ys.argmax(axis=1), return_counts=True)
                total_cls       = np.zeros(self.num_classes, dtype=np.float32)
                total_cls[classes] = counts
                log.debug(f'filtered data: {gen_ys.shape}: {total_cls}')
            
            if (self.forgetting_mode == 'mixed' and (task-1) in self.del_dict) or (
                self.forgetting_mode == 'separate' and task in self.del_dict
                ):
                if self.forg_sample_variants == 'yes' or self.forg_sample_topdown == 'no':  # filter after data generation
                    filter_gen_samples()
                    
            if self.forgetting_mode == 'separate' and task not in self.del_dict:
                if self.adaptor.sample_topdown == 'no':
                    filter_gen_samples()

            log.debug(f'using the following sample weights: gen. data -> {g_s_c} / real data -> {r_s_c}')
            self.sampler.real_sample_coef = r_s_c
            self.sampler.gen_sample_coef = g_s_c
        self.feed_sampler(task, current_train_set)


    def train_on_task(self, task):
        if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip train
            for t_cb in self.train_callbacks:
                t_cb.on_train_begin()
                t_cb.on_train_end()
        else:
            super().train_on_task(task)


    def _test(self, current_task, current_step=None):
        # if self.forgetting_mode == 'mixed' and task in self.del_dict: return  # skip test    
        
        super()._test(current_task, current_step=current_step)

        if self.op_mode == 'test' and self.extra_eval != []:
            _, self.eeval_test, _, self.eeval_amount = self.dataset.get_dataset(self.extra_eval, task_info=None)
            self.model.test_task = f'EXTRA'  # test task identifier
            log.info(f'[TEST] -> {self.model.test_task}({np.unique(np.argmax(self.eeval_test[1], axis=-1))})')
            self.model.evaluate(x=self.eeval_test[0], y=self.eeval_test[1],
                                batch_size=self.test_batch_size,
                                steps=(self.eeval_amount//self.test_batch_size),
                                callbacks=self.eval_callbacks,
                                verbose=self.verbosity,
                                return_dict=True)


    def after_task(self, task, **kwargs):
        super().after_task(task, **kwargs)
        self.adaptor.after_subtask(task)
        
        # visualize and classify protos after last task
        if task == len(self.tasks)-1 and self.adaptor.vis_gen == 'yes':
            topdown_logits, one_hot = self.adaptor.model.construct_topdown_for_classifier(
                self.num_classes, 0.95, self.del_classes)
            gen_xs = self.model.sample_one_batch(   topdown=topdown_logits,
                                                    last_layer_index=-1,
                                                    sampling_bs=self.adaptor.model.sampling_batch_size)
            
            visualize_data(gen_xs, one_hot, f'{self.adaptor.model.vis_path}/gen', f'gen_forg_T{task}')


if __name__ == '__main__':
    Experiment_RTS().run_experiment()
