import sys
import itertools

import numpy as np
import tensorflow as tf

from importlib                  import import_module

from cl_replay.api.utils        import helper, log
from cl_replay.api.experiment   import Experiment_Replay
from cl_replay.api.model        import Func_Model, DNN
from cl_replay.api.parsing      import Kwarg_Parser

from cl_replay.architecture.dgr.model           import DGR
from cl_replay.architecture.dgr.model.dgr_gen   import VAE
from cl_replay.architecture.dgr.adaptor         import Supervised_DGR_Adaptor
from cl_replay.architecture.dgr.generator       import DGR_Generator


class Experiment_DGR(Experiment_Replay):


    def _init_parser(self, **kwargs):
        Experiment_Replay._init_parser(self, **kwargs)
        self.adaptor        = Supervised_DGR_Adaptor(**self.parser.kwargs)
        self.model_type 	= self.parser.add_argument('--model_type', type=str, default='DGR-VAE', choices=['DGR-VAE', 'DGR-GAN'], help='Which architecture to use for DGR?')

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
        self.prev_tasks = []
        
        if self.forgetting_mode == 'mixed' and self.num_tasks in self.del_dict:
            log.error('mixed forgetting task can not be the last task in the training sequence, add a learning task to the end!')
            sys.exit(0)

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
        if self.model_type == 'DGR-GAN':
            for net in ['G', 'D', 'S']:
                sub_model = self.create_submodel(prefix=net)
                if net == 'G': gan_generator        = sub_model
                if net == 'D': gan_discriminator    = sub_model
                if net == 'S': gan_solver           = sub_model
            self.flags.update({'generator': gan_generator, 'discriminator': gan_discriminator, 'solver': gan_solver})
        
        dgr_model = DGR(**self.flags)
        return dgr_model
    

    def create_submodel(self, prefix):
        model_layers = dict()
        model_input_index = self.parser.add_argument(f'--{prefix}_model_inputs', type=int, default=0, help="layer index of model inputs")
        if type(model_input_index) == type(1): model_input_index = [model_input_index]
        model_output_index = self.parser.add_argument(f'--{prefix}_model_outputs', type=int, default=-1, help="layer index of model outputs")
        #-------------------------------------------- INIT LAYERS
        for i in itertools.count(start=0):  # instantiate model layers
            layer_prefix        = f'{prefix}{i}_'
            layer_type          = self.parser.add_argument(f'--{prefix}{i}', type=str, default=None, help="Layer type")
            if layer_type is None: break    # stop if type undefined
            layer_input         = self.parser.add_argument('--input_layer', type=int, prefix=layer_prefix, default = 10000, help="Layer indices of input layers")
            log.debug(f'\tcreating layer of type "{layer_type}", input coming from "{layer_input}"...')
                
            try:  # functional model layer creation
                target = helper.target_ref(targets=layer_input, model_layers=model_layers)
                if target is not None: # not input layer
                    layer_class_name=layer_type.split(".")[-1]
                    layer_obj = getattr(import_module(layer_type), layer_class_name)(name=f"{prefix}{i}", prefix=layer_prefix, **self.flags)(target)
                else: # input Layer
                    layer_obj = getattr(import_module('cl_replay.api.layer.keras'), layer_type)(name=f"{prefix}{i}", prefix=layer_prefix, **self.flags)
                    if hasattr(layer_obj, 'create_obj'):  # if a layer exposes a tensor (e.g. Input), we create a layer object after instantiating the layer module
                        layer_obj = layer_obj.create_obj()
                # fallback
                last_layer_ref = layer_obj  
                last_layer_ref_index = i
                
                model_layers.update({i: layer_obj})
            except Exception as ex:
                import traceback
                log.error(traceback.format_exc())
                log.error(f'error while loading layer item with prefix "{layer_prefix}": {ex}.')

        model_inputs = helper.target_ref(model_input_index, model_layers)
        if model_output_index == -1: model_output_index = last_layer_ref_index
        model_outputs = helper.target_ref(model_output_index, model_layers)

        #-------------------------------------------- INSTANTIATE AND INIT MODEL
        if prefix == 'E' or prefix == 'G' or prefix == 'D':
            model_prefix = 'VAE-' if self.model_type == 'DGR-VAE' else 'GAN-'
            model = Func_Model(inputs=model_inputs, outputs=model_outputs, name=f'{model_prefix}{prefix}', **self.flags)
        if prefix == 'S':
            model = DNN(inputs=model_inputs, outputs=model_outputs, name=f'DGR-solver', **self.flags)
    
        model.compile(run_eagerly=True)
        model.summary()
        
        return model


    def load_model(self):
        Experiment_Replay.load_model(self)

        self.adaptor.set_model(self.model)
        data_dims = self.model.input_size
        self.adaptor.set_input_dims(data_dims[0], data_dims[1], data_dims[2], self.model.num_classes)
        self.adaptor.set_generator(DGR_Generator(model=self.model, data_dims=self.adaptor.get_input_dims()))

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
        return self.adaptor.generate(task, xs, gen_classes, real_classes, **kwargs)


    def prepare_forgetting(self, task): 
        if self.del_dict != {}:
            self.forget_classes = []
            for task_id, del_cls in self.del_dict.items():
                if task >= task_id:
                    self.forget_classes.extend(del_cls)
        
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
        self.adaptor.model.current_task = task  # info about task (SL) for e.g. logging callbacks
        
        if task > 1:
            if self.forgetting_mode == 'mixed' and task in self.del_dict:  # skip this phase
                log.debug(f'skipping this training phase since operating in "mixed" forgetting_mode.')
                return

            # ---- LOSS BALANCING
            r_s_c, g_s_c = self.calc_loss_coefs(task)

            # ---- DATA GENERATION
            self.generated_dataset = self.generate(
                task, data=self.train_set, gen_classes=self.past_classes, real_classes=self.real_classes)

            self.sampler.reset()
            
            log.debug(f'using the following sample weights: gen. data -> {g_s_c} / real data -> {r_s_c}')
            self.sampler.real_sample_coef = r_s_c
            self.sampler.gen_sample_coef = g_s_c
    
        self.feed_sampler(task, self.train_set)
        self.adaptor.before_subtask(task)


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


    def after_task(self, task, **kwargs):
        if self.adaptor.vis_gen == 'yes':
            if self.forgetting_mode == 'mixed' and task-1 in self.del_dict or (
                self.forgetting_mode == 'separate' and task in self.del_dict):
                self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.past_classes, f'gen_pres_T{task}')
            if self.del_dict != {} and task == len(self.tasks)-1:
                self.adaptor.model.generator.visualize_samples(self.model.batch_size, self.forget_classes, f'gen_forg_T{task}')
        
        if task == len(self.tasks)-1:
            return
        
        super().after_task(task, **kwargs)
        self.prepare_forgetting(task+1)
        self.prev_tasks.append(task)
        self.adaptor.after_subtask(
            task, 
            task_classes=self.tasks[task],
            task_data=self.train_set,
        )


if __name__ == '__main__':
    Experiment_DGR().run_experiment()
