import sys
import numpy as np
import tensorflow as tf

from cl_replay.api.data                     import Replay_Sampler, Buffer
from cl_replay.api.experiment               import Experiment_Replay

from cl_replay.architecture.ar.experiment   import Experiment_GMM, Experiment_AR
#from cl_replay.architecture.ar.adaptor      import AR_Unsupervised


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)



class Experiment_AR_OD(Experiment_AR):
    """ TODO: adapt AR to unsupervised learning with outlier detection & buffer structures """

    def _init_parser(self, **kwargs):
        Experiment_AR._init_parser(self, **kwargs)
        #self.adaptor = AR_Unsupervised(**self.parser.kwargs)
        self.buffer_threshold = self.parser.add_argument('--buffer_sample_threshold', type=int, default=5000,
                                                         help='max. sample size of buffer partitions')


    def _init_variables(self):
        self.allow_training = False  # control var to allow training in buffer mode

        if self.replay_task == 'supervised':
            self.replay_task_iters = len(self.DAll) * [
                None]  # saves the training iterations (steps per epoch) for each incremental replay task
            self.classes = list()  # holds the list of current ds classes
            # TODO: make this dynamic -> based on class count
            self.load_test_ds()  # this loads single class test data to evaluate on each task separately

        elif self.replay_task == 'unsupervised':
            self.buffer_partitioning_mode = 'anomaly'

        else: raise Exception('Failed to init buffer... please specify a valid replay_task mode, e.g. supervised/unsupervised...')

        if self.adaptor.data_mode == 'buffer':  # inits the buffer structure
            self.buffer = Buffer(shape=self.get_input_shape(),
                                 num_classes=self.num_classes,
                                 sample_threshold=self.buffer_threshold,
                                 d_type='np',
                                 partitioning_mode=self.buffer_partitioning_mode)


    def before_task(self, task, **kwargs):
        """ generates datasets of past tasks if necessary and resets model layers """
        Experiment_AR.before_task(self, task, **kwargs)

        ### SELECTION OF REPLAY STRATEGY ###
        if self.replay_mode == 'var_gen':   pass
        elif self.replay_mode == 'vanilla_replay': pass
        else: raise Exception('\tTHIS EXPERIMENT FILE ONLY SUPPORTS "vanilla_replay" or "var_gen" FOR "--replay_mode"...')

        ### SELECTION OF CL SETTING ###
        if self.replay_task == 'supervised':
            #TODO: iterations_task_all only known when supervised and dataset can be accessed beforehand
            # SOL: take buffer max fill size as reference and calculate task iterations / epochs based on mini-batch size and buffer
            if self.replay_mode == 'var_gen': pass
            else: pass
        elif self.replay_task == 'unsupervised':
            pass
        elif self.replay_task =='RL':
            #TODO: see above, think about what structures are needed for embedding in RL scenario
            pass

        ### REPLAY SPECIFIC SETTINGS ###
        if task > 1:
            self.allow_training = False     # sets flag after each training task, set to true after buffer merged

            # TODO adjust to allow for layer specific reset_factors
            self.model.reset_factor = self.reset_factor
            self.model.reset(reset_factor = self.reset_factor)



    def before_train(self, task):
        """
        Loop to fill buffer structures with samples, trigger buffer merge/flush & allows/deny training for an experiment

        1) draw a batch from the replay generator containing only the currently processed dataset, we do not care how many samples we get
        this shall simulate streaming data coming in
            -> kind off hacky right now because we are working with TF dataset & replay_gen structure, but it will proof the point
        2) perform outlier detection -> gives us a mask in the case of a supervised task
        3) prepare replay, e.g. by building variants from misclassified samples
        4) save the data into the assigned buffer partitions (0, 1, (2) if variant buffer is used aswell)
        5) evaluate buffer status periodically
        6) break out if buffer status responds with OK (enough samples collected for meaningful training)
            -> this gives us the GO to merge buffer data into replay data generator & start the training procedure
            -> training is of sequential nature
        """
        while True:
            ##################################
            # PERFORM OUTLIER DETECTION HERE #
            ##################################
            #  TODO: check how sampling from GMM components behaves when we do not specify classes to sample from/no topdown, need cyclic!!!
            #   * No matter which method we chose, we constantly have to keep track about new data coming in and clearly separate novel
            #   * sample instances, OD should save away these in a buffer and perform sanity checks when its time to empty buffer and start
            #   * (re-)training for unseen data
            #   * When data is task-splitted -> NP
            #   * When data comes in a mixed fashion, e.g. novel and known classes mixed -> split samples to buffer partitions
            data = self.replay_gen.next_batch()  # current batch - xs, ys
            # TODO: before_task() performs endless loop; fills buffers
            #  * while buffer is not giving the training signal, we block training via bool flag
            #  * whenever the buffer is filled, we start a merge with the replay data generator
            #  * allow training afterwards by exiting loop and setting bool flag to True

            # TODO: add for unsupervised case (inliers, outliers)

            # TODO: have to use fixed size mini-batches for model sampling, we have 2 options here
            #  * Opt 1: Only possible if we get exact batch size with each iteration: obtain bool mask, gen. variants for all samples, mask variants
            #  * Opt 2: Pool missed samples until needed size is reached (generic even for streaming data)

            correct, miss = self.detect_outliers(data, supervised=True)  # returns masks of current batch (if supervised)
            print(correct.shape, miss.shape)

            ##########################
            ### REPLAY PREPARATION ###
            ##########################
            # TODO: generate variants and split data to according buffers, check buffers periodically
            variants = self.adaptor.generate(task, xs=correct, classes=self.classes)
            self.add_variants_to_buffer(variants, task)

            # generate variants from wrong classifications
            # fill hit/miss buffers
            # fill variant buffer
            # check for buffer status, print info
            break


    def train_batch_mode(self, current_task):

        # TODO: buffer training vs. fixed dataset
        if not self.allow_training:
            self.before_train(current_task)

        if self.allow_training:
            super().train_batch_mode(current_task)


    def add_variants_to_buffer(self, variants, task):
        """
        Separate buffers for correct classifications/inliers and misclassifications/outliers
        buffers will grow at a different pace

        """
        # TODO: separate buffer for variants
        # TODO: only create subpartition once
        self.buffer.add_subpartition(pid=2)                                 # generate a subpartition
        self.buffer.add_to_partition(data=variants, pid=2, sub_part=task)   # adds variant data to sub_partition


    def merge_buffer_with_generator(self):
        """ gets triggered by before_train() and merges the buffer into the Replay_Data_Generator instance for training """
        # TODO: finish
        self.replay_gen = Replay_Sampler(batch_size=self.batch_size,
                                         dtype_np=self.model.dtype_np_float,
                                         dtype_tf=self.model.dtype_tf_float)
        pass


    def calc_anomaly_threshold(self, historic_data, method='ses', N=None, ses_alpha=.2):
        """ calculates a threshold for determination of anomalies, uses historic log data to perform different averaging methods """
        # trivial average
        def avg(data):
            return np.mean(data)

        # consider data points back N in the past as "noise" and filter them out
        def last_n_avg(data, n):
            return np.mean(data[-n:])

        # moving average, create a series of avgs. of different subsets from data points
        def moving_avg(data, n):
            sum_ = np.cumsum(data) # cumulative sum
            sum_[n:] = sum_[n:] - sum_[:-n]

            return sum_[n - 1:] / n

        # ses_ = lambda x, a, n: (a * ((1 - a) ** n)) * x         # short lambda fn of ses eqn.

        def ses(data, alpha):
            """
            * SES - simple exponential smoothing, used for data forecasts without clear trends or seasonality patterns
            * recursive... smallest weights are assigned to observations from the past, weights decrease exponentially
            * as we go back in time -> recent points matter more!
            * https://otexts.com/fpp2/ses.html
            * https://grisha.org/blog/2016/01/29/triple-exponential-smoothing-forecasting/
                * y_{T+1|T} =  a*y_T + ses(y_{T-1}, a, 1) + ses(y_{T-2}, a, 2} + ...
                * forecast eqn.:    y^{hat}_{t+h|t) = l_{t}
                * smoothing eqn.:   l_{t] = alpha * y_{t} + (1-alpha)*l_{t-1}
            * alpha is smoothing param: [0., 1.], higher alpha -> less importance of past points
            * we should not deal with cyclic patterns/seasonality if we only log losses from correctly classified samples
            * only record losses on training-samples (streaming data scenario)
            * consider if we build the average over whole training period OR reset it with each time replay training is triggered
            """
            results = np.zeros_like(data)

            results[0] = data[0]                # first data point, o history to learn from
            for t in range(1, data.shape[0]):   # loop over time steps t, ex. first data point since no history available
                results[t] = alpha * data[t] + (1 - alpha) * results[t - 1]

            return results

        if method == 'ses':     return ses(historic_data, ses_alpha);
        elif method == 'mvg':   return moving_avg(historic_data, n=N)
        elif method == 'navg':  return last_n_avg(historic_data, n=N)
        elif method == 'avg' :  return avg(historic_data)
        else: raise Exception('anomaly detection method unknown...')


    def detect_outliers(self, data, supervised=True):
        # TODO: outlier detection should eliminate the need for known/fixed subtask boundaries.
        #  * Knowledge about the existence of classes per sub-task (count of classes & mixing proportions)
        #  * Knowledge about the number of samples per sub-task permitted (sample size & arival of unknown data)
        #  * Definition of a sub-task (is it just a training period where novel instances where found?)
        #  * Can we treat the initial training differently? Is there a clear distinction between "initial training" & "replay training"
        # Scenario:
        # We only allow knowledge about the existence of an initial training task, and assume that we know how many classes/samples it contains.
        #  * We have a well-defined initial task, after which new data might arrive at an unknown timing & unknown size
        #  * new data can potentially be mixed with old data from previous tasks
        #  * Knowledge about past & future data is strictly forbidden, no information should be hold by the model itself
        #  * Data drift has to be recognized & appropriately dealt with in an automated manner

        if len(data) == 2:          # check if we have labels, else no mask computation/classification possible
            xs, ys = data[0], data[1]
        else:
            raise Exception('Only labeled (supervised) outlier detection is supported...')

        self.model.test_step(data)  # perform fwd, loss calc on data batch ONLY SUPERVISED for now

        # RL            -> bad rewards (but outlier can yield positive reward, so it does not matter)
        # supervised    -> false classification
        # unsuspervised -> outlier instances

        if supervised:
            # detect if classified correctly or misclassified
            bool_mask           = self.model.compute_mask_(labels=ys)                   # calc mask, gives us indices for correct/misclassified
            true                = xs[bool_mask]
            false               = xs[np.logical_not(bool_mask)]

            return true, false # returns bool mask

        outlier_layer_id    = self.model.outlier_layer                                  # grab which layer is responsible for detecting outliers
        per_sample_losses   = self.model.layers[outlier_layer_id].get_raw_layer_loss()  # per sample losses

        if not supervised:
            historic_loss_data  = self.model.get_historic_loss_data() #TODO: set limit for historic data, so structure does not grow linear
            kappa               = .8 # TODO: scaling factor for threshold value??? Define as console arg
            thresh_value        = kappa * self.calc_anomaly_threshold(historic_loss_data, method='avg')

            outliers_mask       = per_sample_losses <= thresh_value
            inliers_mask        = np.logical_not(outliers_mask)

            outliers            = per_sample_losses[outliers_mask]
            inliers             = per_sample_losses[inliers_mask]

        # TODO:
        #  * Save indices below threshold to outlier buffer, save indices above threshold to inlier buffers
        #  * Merge buffer input to Replay_Data_Generator as a sub task, inliers get added immediately
        #  * How do we assign samples to a specific sub task in case of unknown labels?
        # TODO: Log_Losses callback -> track training loss for OD


if __name__ == '__main__':
    Experiment_AR_OD().train()