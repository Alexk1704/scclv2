import math
import numpy                as np
import tensorflow           as tf

from cl_replay.api.layer    import Custom_Layer
from cl_replay.api.utils    import log


# TODO: implement
class NearestMean_Layer(Custom_Layer):


    def __init__(self, **kwargs):
        super(NearestMean_Layer, self).__init__(**kwargs)
        self.kwargs                 = kwargs

        self.input_layer            = self.parser.add_argument('--input_layer',         type=int,   default=[None],             help='a list of prefixes of this layer inputs')
        #-------------------------- SAMPLING
        self.num_classes            = self.parser.add_argument('--num_classes',         type=int,   default=10,                 help='number of output classes')
        self.sampling_batch_size    = self.parser.add_argument('--sampling_batch_size', type=int,   default=100,                help='sampling batch size')
        #-------------------------- LEARNING
        self.batch_size             = self.parser.add_argument('--batch_size',          type=int,   default=100,                help='bs')
        # each K is a vote, class with minimum distance to the evaluated sample is the forecast for the instance ("classification by majority vote").
        self.K                      = self.parser.add_argument('--K',                   type=int,   default=10,                 help='top K measured distnaces')


    def build(self, input_shape):
        self.input_shape    = input_shape
        self.channels_in    = np.prod(input_shape[1:])
        self.channels_out   = self.num_classes

        self.fwd, self.return_loss, self.raw_return_loss    = None, None, None
        self.resp_mask                                      = None

        self.build_layer_metrics()


    def call(self, inputs, training=None, *args, **kwargs):
        self.fwd = self.forward(input_tensor=inputs)

        return self.fwd


    # @tf.function(autograph=False)
    def forward(self, input_tensor):
        """ #NOTE: some ideas regarding the function of this layer
        * Remember, 1st task we don't have trained Gaussians yet... 
        * So how do we determine the classes for T1? Take real samples from training set.
        * Later on, used trained mixture model to compute mean for each component and use these means as the neirest neighbors.
        * Requires that each Gaussian gets assigned a class label (via training?)

        * Other option: always use real data samples, i.e., build a reservoir with N samples for each class K.
        
        * Calculate distance between input_tensor and means from reservoir/mixture model.
        * Sort distances
        * Chose first k distances
        
        * Distance measure:
            # 1. euclidean: d(A,B) = \sqrt{\sum (A_i - B_i)**2}
            # 2. L1 distance: d(A,B) = \sum |A_i - B_i|
            # 3. cosine similarity: d(A,B) = 1 - \frac{A \cdot B}{||A|| ||B||}
        
        * prediction -> tf.argmin(distance)
        
        * Layer is not "trained"! Only fwd(), however samples need to be chosen (rebalanced later?) accordingly
        
        * GMM prototype(s) with highest resp(s), maybe pick top 3 activations? -> f/e sample we calc. the distance to the reservoir of samples?
        """
        # return self.logits
        return


    # def get_fwd_result(self):       return self.fwd
    # def get_output_result(self):    return self.logits


    def pre_train_step(self):
        return


    def reset_layer(self, **kwargs):
        return


    def compute_output_shape(self, input_shape):
        ''' Returns a tuple containing the output shape of this layers computation. '''
        return self.batch_size, self.channels_out


    def set_parameters(self, **kwargs):
        return


    def get_layer_opt(self):
        ''' Returns the optimizer instance attached to this layer. '''
        return None


    def build_layer_metrics(self):
        self.layer_metrics = [
            # tf.keras.metrics.CategoricalAccuracy(name=f'{self.prefix}acc') # uses one-hot
        ]


    def get_layer_metrics(self):
        return self.layer_metrics


    def get_logging_params(self):
        return {}
