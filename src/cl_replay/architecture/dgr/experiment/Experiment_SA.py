import sys
import itertools

import numpy as np
import tensorflow as tf

from importlib                  import import_module

from cl_replay.api.utils        import helper, log
from cl_replay.api.model        import Func_Model, DNN
from cl_replay.api.parsing      import Kwarg_Parser

from cl_replay.architecture.dgr.model           import DGR
from cl_replay.architecture.dgr.model.dgr_gen   import VAE
from cl_replay.architecture.dgr.adaptor         import Supervised_DGR_Adaptor
from cl_replay.architecture.dgr.generator       import DGR_Generator
from cl_replay.architecture.dgr.experiment      import Experiment_DGR


class Experiment_SA(Experiment_DGR):


    def _init_parser(self, **kwargs):
        Experiment_DGR._init_parser(self, **kwargs)
        
        # ---- Forgetting with Selective Amnesia
        self.amnesiac           = self.parser.add_argument('--amnesiac', type=str, default='no', choices = ['no', 'yes'],  help='activate selective amnesia.')
        self.sa_forg_iters      = self.parser.add_argument('--sa_forg_iters', type=int, default=10000, help='number of iterations performed to forget classes.')
        self.sa_fim_samples     = self.parser.add_argument('--sa_fim_samples', type=int, default=50000, help='number of samples generated to compute FIM.')

    #-------------------------------------------- MODEL CREATION & LOADING
    def create_model(self):
        ''' 
        Instantiate a functional keras DGR dual-architecture, builds layers from imported modules specified via bash file parameters "--XX_".
            - Layer and model string are meant to be modules, like a.b.c.Layer or originate from the api itself (cl_replay.api.layer.keras). 
            - DGR uses 3 networks, as of such, the single models are defined by using their distinct prefix.
                - EX_ : encoder network (VAE)
                - GX_ : generator network (GAN)
                - DX_ : decoder/discriminator network (VAE/GAN)
                - SX_ : solver network
        '''
        log.debug(f'instantiating model of type "{self.model_type}"')
        
        if self.model_type == 'DGR-VAE':
            for net in ['E', 'D', 'S']:
                sub_model = self.create_submodel(prefix=net)
                # each sub_model defines a "functional block"
                if net == 'E': vae_encoder = sub_model
                if net == 'D': vae_decoder = sub_model
                if net == 'S': dgr_solver  = sub_model
            self.flags.update({'encoder': vae_encoder, 'decoder': vae_decoder, 'solver': dgr_solver})
            if self.amnesiac == 'yes':
                sub_model = self.create_submodel(prefix='D')
                self.flags.update({'decoder_copy': sub_model})
        if self.model_type == 'DGR-GAN':
            log.error('SA is only implemented for the DGR-VAE model type.')
            sys.exit(0)
            
        dgr_model = DGR(**self.flags)
        return dgr_model


    #-------------------------------------------- TRAINING/TESTING
    def train_on_task(self, task):
        if self.amnesiac == 'yes':
            if self.forgetting_mode == 'mixed':
                log.error(f'mixed forgetting mode is not possible for selective amnesia.')
                sys.exit(0)
            if self.forgetting_mode == 'separate':
                if task in self.del_dict:  # detected forgetting task
                    for t_cb in self.train_callbacks: t_cb.on_train_begin()
                    self.adaptor.model.generator.forget_training(
                        num_iters=self.sa_forg_iters,
                        batch_size=None,
                        forget_classes=self.forget_classes,
                        preserved_classes=self.past_classes
                    )
                    for t_cb in self.train_callbacks: t_cb.on_train_end()
                    
                    if self.adaptor.vis_gen == 'yes':
                        self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.past_classes, f'gen_pres_T{task}')
                        # self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.forget_classes, f'gen_forg_T{task}')

                    # ---- train solver on generated data from the re-trained VAE
                    if self.adaptor.drop_solver == 'yes':
                        self.adaptor.model.reset_solver(self.adaptor.initial_model_weights)
                        log.debug(f'resetting solver before proceeding with the next task!')
                    else:
                        # ---- retrain solver on gen. data (5k per class)
                        samples_to_generate = 5000 * len(self.past_classes)
                        self.generated_dataset = self.generate(
                            task, data=None, gen_classes=self.past_classes, real_classes=self.real_classes, samples_to_generate=samples_to_generate)

                        self.sampler.reset()
                        self.sampler.real_sample_coef = 1
                        self.sampler.gen_sample_coef = 1
                        self.feed_sampler(task, self.generated_dataset) 
                        
                        log.info('{:20s}'.format(f' [START] solver training on task: {task} total epochs: {self.epochs} ').center(64, '~'))
                        _ = self.adaptor.model.solver.fit(self.sampler(),
                                                          epochs=self.adaptor.model.solver_epochs,
                                                          batch_size=self.batch_size,
                                                          steps_per_epoch=self.train_steps,
                                                          callbacks=self.train_callbacks,
                                                          verbose=self.verbosity)
                    return
        if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip train
            for t_cb in self.train_callbacks:
                t_cb.on_train_begin()
                t_cb.on_train_end()
        else:
            super().train_on_task(task)


    def after_task(self, task, **kwargs):
        
        if self.adaptor.vis_gen == 'yes':
            if task in self.del_dict:
                self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.past_classes, f'gen_pres_T{task}')
            if task == len(self.tasks)-1:
                self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.forget_classes, f'gen_forg_T{task}')
        
        if task == len(self.tasks)-1:
            return
        
        self.prepare_forgetting(task+1)
    
        if self.amnesiac == 'yes' and task in self.del_dict:  # skip exec of adaptor code for forgetting phase
            return

        self.prev_tasks.append(task)
        self.adaptor.after_subtask(
            task, 
            task_classes=self.tasks[task],
            task_data=None,  # self.training_sets[task],
            past_classes=self.past_classes,
            fim_samples=self.sa_fim_samples,
            prev_tasks=self.prev_tasks
        )


if __name__ == '__main__':
    Experiment_SA().run_experiment()
