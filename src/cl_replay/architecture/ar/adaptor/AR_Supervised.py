from cl_replay.api.experiment.adaptor import Supervised_Replay_Adaptor
from cl_replay.api.utils              import log



class AR_Supervised(Supervised_Replay_Adaptor):
    ''' 
        Adaptor for Adiabatic Replay in a supervised classification scenario.

        Attributes
        ----------
        sample_topdown : bool, default=False
            - Turn on/off conditional top-down sampling
        sample_variants : bool default=False
            - Turn on/off variant sampling.
        sampling_layer : int, optional, default=-1
            - The layer index of sampling layer
        sampling_clip_range : int, optional, default=[0.,1.]
            - Clips the generated samples to a range of [min - max].

        ...
    '''
    def __init__(self, **kwargs):
        Supervised_Replay_Adaptor.__init__(self, **kwargs)
        self.sample_topdown         = self.parser.add_argument('--sample_topdown',      type=str,   default='no', choices=['no', 'yes'], help='turn on/off conditional sampling')
        self.sample_variants        = self.parser.add_argument('--sample_variants',     type=str,   default='no', choices=['no', 'yes'], help='turn on/off variant sampling')
        self.sampling_layer         = self.parser.add_argument('--sampling_layer',      type=int,   default=-1, required=False, help='layer index of sampling layer')
        self.sampling_clip_range    = self.parser.add_argument('--sampling_clip_range', type=float, default=[0., 1.], help='clip generated samples to this range')
        
        if len(self.sampling_clip_range) != 2: self.sampling_clip_range = [0., 1.]

        self.vis_batch  = kwargs.get('vis_batch', 'no')
        self.vis_gen    = kwargs.get('vis_gen', 'no')


    def generate(self, task=-1, xs=None, gen_classes=None, real_classes=None, generate_labels=True, **kwargs):
        if kwargs.get('sample_variants', None): sample_variants = kwargs.get('sample_variants')
        else: sample_variants = self.sample_variants
        if kwargs.get('sample_topdown', None): sample_topdown = kwargs.get('sample_topdown')
        else: sample_topdown = self.sample_topdown  
        
        if xs is None:
            stg = kwargs.get('samples_to_generate', 10000)   
        else:
            stg = xs.shape[0]  # const

        return self.generator.generate_data(task=task, xs=xs, gen_classes=gen_classes,
                                            stg=stg, sbs=self.sampling_batch_size,
                                            sampling_layer=self.sampling_layer, sampling_clip_range=self.sampling_clip_range,
                                            top_down=sample_topdown, variants=sample_variants, generate_labels=generate_labels,
                                            vis_batch=self.vis_batch, vis_gen=self.vis_gen)


    def before_subtask(self, task):
        if task == 1: self.store_sampling_params()
        if task > 1: self.model.reset()


    def after_subtask(self, task, **kwargs):
        pass


    def set_class_freq(self, class_freq):
        self.model.set_parameters(class_freq=class_freq)


    def change_sampling_params(self, restore=False, sampling_I=-1, sampling_S=1, somSigma_sampling='yes'):
        i = 0
        for layer in self.model.layers[1:]:
            if hasattr(layer, 'is_layer_type'):
                if layer.is_layer_type('GMM_Layer'):
                    if restore:
                        log.debug(f'restoring old sampling params for {layer.name}!')
                        sampling_I, sampling_S, somSigma_sampling = self.model_sampling_params[i]
                    log.debug(f'changing sampling params of {layer.name} to: I={sampling_I}, S={sampling_S}!')
                    layer.sampling_I = sampling_I
                    layer.sampling_S = sampling_S
                    layer.somSigma_sampling = somSigma_sampling

                    i += 1

    def store_sampling_params(self):
        self.model_sampling_params = []
        for layer in self.model.layers[1:]:
            if hasattr(layer, 'is_layer_type'):
                if layer.is_layer_type('GMM_Layer'):
                    self.model_sampling_params.append([layer.sampling_I, layer.sampling_S, layer.somSigma_sampling])