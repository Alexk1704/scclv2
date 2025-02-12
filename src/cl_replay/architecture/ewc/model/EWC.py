import time
import tensorflow      as tf
import numpy           as np

from collections       import defaultdict
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients

from cl_replay.api.model.DNN    import DNN
from cl_replay.api.parsing      import Kwarg_Parser
from cl_replay.api.utils        import log


class EWC(DNN):
    
    
    def __init__(self, inputs, outputs, **kwargs):
        super(EWC, self).__init__(inputs, outputs, **kwargs)
        self.parser        = Kwarg_Parser(**kwargs)
        self.lambda_       = self.parser.add_argument('--lambda', type=float, default=100., help='EWC lambda')
        self.num_tasks     = self.parser.add_argument('--num_tasks', type=int, default=1, help='specify the number of total tasks, in case of unsupervised: num_tasks splits given dataset into equal proportions!')
        self.num_classes   = self.parser.add_argument('--num_classes', type=int, default=1, help='specify the total number of classes.')
        
        self.init_tf_variables()
        
        self.current_task = -1
        

    def randomize_weights(self): 
        for var in self.trainable_variables:
            var.assign(tf.random.truncated_normal(var.shape, 0, 0.1, dtype=self.dtype_tf))


    def init_tf_variables(self):
        # NOTE: using self.num_classes for FIMs instead of tasks!
        self.ewc_storage    = {i: [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables] for i in range(0, self.num_classes)}
        self.fims           = {i: [tf.Variable(initial_value=var*0., trainable=False) for var in self.trainable_variables] for i in range(0, self.num_classes)}
        self.coeffs         = {i: tf.Variable(0.0, trainable=False) for i in range(0, self.num_classes)}
        self.fim_acc        = [tf.Variable(initial_value=var*0., trainable=False, name=f"acc{i}") for i, var in enumerate(self.trainable_variables)]
        self.lambda_        = tf.Variable(self.lambda_, trainable=False)


    def compute_loss(self, xs, ys, logits, sample_weight=None):
        ewc_loss = tf.constant(0.0)
        dnn_loss = tf.reduce_mean(self.loss(ys, logits))
        
        ewc_loss += self.compute_ewc_penalty(ewc_loss)
        
        self.custom_metrics[0].update_state(ys, logits)
        self.custom_metrics[1].update_state(dnn_loss+ewc_loss)
        
        # tf.print("DNN L: ", dnn_loss, "EWC L: ", ewc_loss)
        return dnn_loss + ewc_loss


    def compute_ewc_penalty(self, l):
        # NOTE: currently calculating FIM per-class!
        # for task_id in range(1, self.num_tasks):  # calculates the EWC loss penalty
        for class_id in range(0, self.num_classes):
            for var, var_prev, fim_var_prev in zip(self.trainable_variables, self.ewc_storage[class_id], self.fims[class_id]):
                # tf.print(var.shape, var_prev.shape, fim_var_prev.shape)
                # tf.print(self.coeffs[class_id], tf.reduce_mean(var), tf.reduce_mean(var_prev), tf.reduce_mean(fim_var_prev))
                l = l + self.coeffs[class_id] * tf.reduce_sum(fim_var_prev * (var - var_prev)**2)
        return l * self.lambda_


    def init_fim_struct(self):
        for fim_var in self.fim_acc: fim_var.assign(fim_var * 0.)  # reset accumulators


    def compute_fim(self, xs):
        loss = None
        with tf.GradientTape() as g:
            model_out   = self(inputs=xs)
            loss        = tf.nn.log_softmax(model_out)     
        gradients = g.gradient(loss, self.trainable_variables)
        '''
        We accumulate the second-order partial derivates, squaring the first-order partial derivate is sufficient.
        Using a common alternative to the fisher matrix def.: evaluate it from the negative expected value of the Hessian of the log-likelihood:
        $$ \mathcal{F}(\theta) = -\mathbb{E} \left[ \frac{\partial^2}{\partial \theta^2} \log L(X; \theta) \right $$
        This relates to the curvature of the log-likelihood function around the maximum likelihood estimate.
        '''
        for grad, acc in zip(gradients, self.fim_acc): acc.assign(acc + grad**2)  # accumulate over MB
        

    def finalize_fim(self, num_batches, batch_size, id):
        for i, acc in enumerate(self.fim_acc):
            acc.assign(acc / num_batches / batch_size)
            # print("i: ", i, acc.shape, acc.numpy().min(), acc.numpy().max())
            
        # update variables
        for fim, acc in zip(self.fims[id], self.fim_acc): fim.assign(acc)
        for var, storage in zip(self.trainable_variables, self.ewc_storage[id]): storage.assign(var)
        self.coeffs[id].assign(tf.constant(1.0))


    def set_parameters(self, **kwargs):
        self.current_task = kwargs.get('current_task', None)


    def apply_imm_after_task(self, mode, imm_transfer_type, imm_alpha, current_task):
        if mode == 'ewc'    : return
        if current_task == 0: return

        prev_task_weight = 1. - imm_alpha
        cur_task_weight  = imm_alpha
        if mode == 'mean_imm':
            for var, prev_var in zip(self.trainable_variables, self.ewc_storage[current_task - 1]):
                var.assign(prev_task_weight * prev_var + cur_task_weight * var)

        if mode == 'mode_imm':
            for var, prev_var, oldfim_var, fim_var, fim_b in zip(
                self.trainable_variables, self.ewc_storage[current_task - 1], self.fims[current_task - 1], self.fims[current_task]):
                common_var = prev_task_weight * oldfim_var + cur_task_weight * fim_var + 1e-30

        new_var = (prev_task_weight * prev_var * oldfim_var + cur_task_weight * var * fim_var) / common_var
        var.assign(new_var)
