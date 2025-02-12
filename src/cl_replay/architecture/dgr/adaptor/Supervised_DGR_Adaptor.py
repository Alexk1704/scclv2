import numpy as np
import tensorflow as tf

from cl_replay.architecture.ar.adaptor  import AR_Supervised
from cl_replay.api.utils                import log


class Supervised_DGR_Adaptor(AR_Supervised):


    def __init__(self, **kwargs):
        AR_Supervised.__init__(self, **kwargs)

        self.samples_to_generate    = self.parser.add_argument('--samples_to_generate', type=float, default=1., required=False, help='total amount or factor of generated samples for each replay sub-task')
        self.drop_solver 	        = self.parser.add_argument('--drop_solver', 		type=str,   default='no', choices=['no', 'yes'], help='should the model be dropped after every task?')
        self.drop_generator         = self.parser.add_argument('--drop_generator', 	    type=str,   default='no', choices=['no', 'yes'], help='should the generator be dropped after every task?')

        self.vis_gen                = kwargs.get('vis_gen', 'no')
        self.amnesiac               = kwargs.get('amnesiac', 'no')


    def generate(self, task, data=None, gen_classes=None, real_classes=None, **kwargs):
        if data is None:
            stg = kwargs.get('samples_to_generate', 10000)   
        else:
            if self.samples_to_generate == -1.: stg = data.shape[0]     # const
            else: stg = data.shape[0] * len(gen_classes)                # balanced

        return self.generator.generate_data(
            task=task, xs=None, gen_classes=gen_classes,
            stg=stg, sbs=self.sampling_batch_size, vis_gen=self.vis_gen)


    def before_subtask(self, task, total_samples=None, **kwargs):
        if task == 1: 
            self.initial_model_weights = self.model.get_model_weights()
        if task > 1:
            if self.drop_generator == 'yes':
                self.model.reset_generator(self.initial_model_weights)
            if self.drop_solver == 'yes':
                self.model.reset_solver(self.initial_model_weights)
        self.model.set_train_generator_flag(True)


    def after_subtask(self, task, task_classes, task_data, **kwargs):
        if self.vis_gen == 'yes':
            self.model.generator.visualize_samples(self.model.batch_size, task_classes, f'gen_T{task}')
        if self.amnesiac == 'yes':
            fim_samples     = kwargs.get('fim_samples', 10000)
            past_classes    = kwargs.get('past_classes', [])
            prev_tasks      = kwargs.get('prev_tasks', [])
            
            self.model.generator.set_parameters(**{'current_task' : task, 'prev_tasks' : prev_tasks})
            self.model.generator.init_fim_struct()
            self.model.generator.compute_fim(fim_samples, past_classes)
            self.model.generator.finalize_fim(fim_samples, 1)
    

    def set_class_freq(self, class_freq):
        self.class_freq = class_freq  # compat
