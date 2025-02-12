import time
import numpy as np
import tensorflow as tf

from cl_replay.api.model            import Func_Model
from cl_replay.api.parsing          import Kwarg_Parser
from cl_replay.api.utils            import log, change_loglevel


class DNN(Func_Model):
    
    
    def __init__(self, inputs, outputs, name="DNN", **kwargs):
        super(DNN, self).__init__(inputs, outputs, name, **kwargs)
        self.kwargs             = kwargs

        self.opt                = self.parser.add_argument('--opt',             type=str,   default='sgd', choices=['sgd', 'adam'], help='Choice of optimizer.')

        self.sgd_epsilon        = self.parser.add_argument('--sgd_epsilon',     type=float, default=1e-4, help='SGD learning rate.')
        self.sgd_momentum       = self.parser.add_argument('--sgd_momentum',    type=float, default=0.0, help='SGD momentum.')
        self.sgd_wdecay         = self.parser.add_argument('--sgd_wdecay',      type=float, default=None, help='SGD weight decay.')
        
        self.adam_epsilon       = self.parser.add_argument('--adam_epsilon',    type=float, default=1e-3, help='Optimizer learning rate.')
        self.adam_beta1         = self.parser.add_argument('--adam_beta1',      type=float, default=0.9, help='ADAM beta1')
        self.adam_beta2         = self.parser.add_argument('--adam_beta2',      type=float, default=0.999, help='ADAM beta2')
        
        self.vis_path           = self.parser.add_argument('--vis_path',        type=str, required=True)
        self.log_level          = self.parser.add_argument('--log_level',       type=str, default='DEBUG', choices=['DEBUG', 'INFO'], help='determine level for console logging.')
        change_loglevel(self.log_level)

        self.dtype_np_float = np.float32
        self.dtype_tf_float = tf.float32


    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=True, steps_per_execution=1, **kwargs):
        if not loss:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        if not optimizer:
            if self.opt == 'sgd':
                if type(self.sgd_wdecay) != type(0.0):
                    self.sgd_wdecay = None
                optimizer = tf.keras.optimizers.SGD(learning_rate=self.sgd_epsilon, momentum=self.sgd_momentum, weight_decay=self.sgd_wdecay)
            if self.opt == 'adam':
                optimizer = tf.keras.optimizers.Adam(self.adam_epsilon, self.adam_beta1, self.adam_beta2)
        
        if not metrics:
            metrics = [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
                tf.keras.metrics.Mean(name='loss'),
                tf.keras.metrics.Mean(name='step_time')
            ]
            self.custom_metrics = metrics

        self.model_params = {}

        self.supports_chkpt = False  # TODO: enable checkpointing
        self.current_task, self.test_task = 'T?', 'T?'  # placeholder

        super(Func_Model, self).compile(
            optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, 
            weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, steps_per_execution=steps_per_execution, 
            **kwargs
        )

    @tf.function
    def train_step(self, data, **kwargs):
        xs, ys, sw = data[0], data[1], data[2]
        
        t1 = tf.timestamp(name='t1')

        with tf.GradientTape(persistent=True) as tape:
            logits = self(inputs=xs, training=True)
            loss = self.compute_loss(xs, ys, logits, sample_weight=sw)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        del tape
        
        t2 = tf.timestamp(name='t2')
        tdelta = tf.subtract(t2, t1)
        
        self.custom_metrics[0].update_state(ys, logits)
        self.custom_metrics[1].update_state(loss)
        self.custom_metrics[-1].update_state(tdelta)

        return {m.name: m.result() for m in self.custom_metrics}


    def test_step(self, data, **kwargs):
        xs, ys = data[0], data[1]
        
        t1 = tf.timestamp(name='t1')
        
        logits = self(inputs=xs, training=False)
        
        t2 = tf.timestamp(name='t2')
        tdelta = tf.subtract(t2, t1)

        self.custom_metrics[0].update_state(ys, logits)
        self.custom_metrics[-1].update_state(tdelta)
        
        return {m.name: m.result() for m in self.custom_metrics}


    @property
    def metrics(self):
        return self.custom_metrics


    def get_model_params(self):
        ''' Return a dictionary of model parameters to be tracked for an experimental evaluation via W&B. '''
        return {}