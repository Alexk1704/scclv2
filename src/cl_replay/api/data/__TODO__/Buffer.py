import numpy as np


# NOTE: 
"""
I have started to implement this generic buffer structure to make it generically suitable for many different cases including: 
    supervised learning, unsupervised learning, anomaly detectio, etc.
This code is incomplete, however, the general idea behind this and the required method stubs do exist.
"""


class Partition_Mode:
    CLASSES         = 'classification'  # supervised classification (ys known)
    ANOMALY         = 'anomaly'         # unsupervised/regression tasks (no label data)


class Partition_Type:
    INLIERS         = 'inliers'         # GROWING partitions; split into sub-task partitions; inlier instances
    VARIANTS        = 'variants'        # GROWING partitions, split into sub-task partitions; generate variants for misclassifications/anomalies
    MISSED_CLASSES  = 'misclassified'   # GROWING partitions: split into sub-task partitions; false classified; only supervised
    ANOMALIES       = 'anomalies'       # FIXED partition; mixed data; outlier instances
    SUB_PART        = 'sub_partition'   # FIXED sub-partition for a growing data buffer


class Dataset_Type:
    NUMPY           = 'np'
    TENSORFLOW      = 'tf'



class Buffer:
    '''
        A buffer instance has a partition mapping to grant access to different subsets of the data.
        It supports adding/drawing of: anomalies, inliers, variants, and misclassified samples from past sub-tasks (if the partition grows dynamically).
        Growing partitions for inliers, variants & misclassified samples based on sub-task splits (potentially holding up to K classes per sub-task).

        Attributes
        ----------
        shape : tuple
            - Specify sample & (label) data shapes, shapes are matching across all samples/partitions.

        num_classes: int
            - If supervised classification, set to 0 if unsupervised/regression.

        sample_threshold : int
            - Number of samples the inlier/outlier/variant buffers can hold until it needs to be cleared/merged, if a
            buffer partition has dynamically growing sub-partitions for sub-tasks, the max size refers to the max size
            each sub-partition can have.

        partitioning_mode : Partition_Mode
            - Supervised or Unsupervised buffer setup, this inits needed structures.

        d_type : Dataset_Type
            - Either 'np' for numPy arrays, or 'tf' for TensorFlow constants.
    '''
    def __init__(self, shape, num_classes=0, sample_threshold=5000, partitioning_mode='classification', d_type='np', variants=False):
        
        if d_type == 'np':
            d_type = Dataset_Type.NUMPY
        elif d_type == 'tf':
            d_type = Dataset_Type.TENSORFLOW
        else: raise Exception("specified dataset type is not supported, please use 'np' or 'tf'....")

        self.buffer             = [None] * 3    # data buffer
        self.partition_mapping  = {}            # bookkeeping struct
        self.partition_count    = 0             # number of partitions (2 without variants, 3 with variants)

        # setup & init buffer partitions
        if partitioning_mode == Partition_Mode.CLASSES:
            labeled = True
            self.init_partition_mapping(0,          # SUPERVISED
                                        p_type      = Partition_Type.MISSED_CLASSES,
                                        sub_parts   = list(),
                                        shape       = shape,
                                        d_type      = d_type,
                                        max_size    = sample_threshold,
                                        labeled     = True,
                                        num_classes = num_classes)
        elif partitioning_mode == Partition_Mode.ANOMALY:
            labeled = False
            self.init_partition_mapping(0,          # UNSUPERVISED
                                        p_type      = Partition_Type.ANOMALY,
                                        sub_parts   = None,
                                        shape       = shape,
                                        d_type      = d_type,
                                        max_size    = sample_threshold,
                                        labeled     = False,
                                        num_classes = num_classes)
        else: raise Exception('Partitioning mode not supported, please see Partition_Mode class for valid choices...')
        self.create_partition(0); self.partition_count += 1
        # partition for inliers/correct classifications; static buffer - only representing current sub-task
        self.init_partition_mapping(1,          # INLIERS
                                    p_type      = Partition_Type.INLIERS,
                                    sub_parts   = None,
                                    shape       = shape,
                                    d_type      = d_type,
                                    max_size    = sample_threshold,
                                    labeled     = labeled,
                                    num_classes = num_classes)
        self.create_partition(1); self.partition_count += 1
        if variants:
            self.init_partition_mapping(2,          # VARIANTS
                                        p_type      = Partition_Type.VARIANTS,
                                        sub_parts   = list(),
                                        shape       = shape,
                                        d_type      = d_type,
                                        max_size    = sample_threshold,
                                        labeled     = labeled,
                                        num_classes = num_classes)
            self.create_partition(2); self.partition_count += 1


    def init_partition_mapping(self, pid: object, p_type: object, shape: tuple, d_type: object = Dataset_Type.NUMPY,
                               max_size: int = 5000, labeled: bool = True, num_classes: int = 0, sub_parts: list = None):
        '''
        Creates and saves the config for a partition/sub-partition in the partition mapping bnokkeeping structure.

        Args
        ----------
        pid : int
            Specification of the partition identifier.

        p_type: Partition_Type
            specify if partition is: INLIERS, VARIANTS, MISSED_CLASSES, ANOMALY, SUB_PART.

        shape : tuple
            Specify sample & (label) data shapes, shapes are matching across all samples/partitions.

        d_type : Dataset_Type
            Either 'np' for numPy arrays, or 'tf' for TensorFlow constants.

        max_size : int
            Number of samples the buffer can hold until it needs to be cleared/merged.

        labeled : bool
            If data contains labels/targets (ys).

        num_classes : int
            Total number of dataset classes (e.g., for MNIST = 10)

        sub_parts : list
            Holds sub-partitions for each sub-task encountered during training, and is None if buffer is static.
            Contains a list with corresponding indices for each sub-task encountered in order of task appearance.

        '''
        entry = { pid :
            {
                'type': p_type,
                'dataset_type': d_type,
                'sub_parts' : sub_parts, # contains a list if partition grows with num of sub-tasks, is None if no sub-partitioning
                'dataset_shape': shape,
                'sample_size': 0,
                'labeled': labeled,
                'num_classes': num_classes,
                'max_partition_size': max_size
            }
        }
        self.partition_mapping.update(entry)


    def create_partition(self, pid, sub_part=False):
        '''
        Creates a new buffer partition based on the given config from self.partition_mapping.

        Args
        ----------
        pid : int
            Specification of the partition identifier.

        sub_part : bool
            If set to true, returns a "super-partition" representing a list of sub-partitions

        Returns
        -------
        partition: list
            Initialized (empty) partition based on the partition config.
        '''
        config = self.partition_mapping.get(pid)
        partition = []

        dims_xs, dims_ys = None, None

        if sub_part:
            return partition

        if config['dataset_type'] == 'np':
            # construct shape tuple(s)
            h, w, c         = config['dataset_shape'][0], config['dataset_shape'][1], config['dataset_shape'][2]
            dims_xs         = (config['max_partition_size'], h, w, c)
            xs_struct       = np.empty(shape=dims_xs, dtype=np.float32)
            partition.append(xs_struct)

            if config['labeled']:
                num_classes = config['num_classes'] # assume labels are one-hot encoded
                dims_ys     = (config['max_partition_size'], num_classes)
                ys_struct   = np.empty(shape=dims_ys, dtype=np.float32)
                partition.append(ys_struct)
            else:
                partition.append([None])

        print(f'created a partition for {config["type"]}.... with dims: {dims_xs}, {dims_ys}')

        if config['dataset_type'] == 'tf':
            #TODO: TF dataset_type handling
            pass

        return partition


    def add_subpartition(self, pid):
        '''
        Creates a new sub partition for a "super partition", and appends it to the structure.

        Args
        ----------
        pid : int
            Specification of the partition identifier.
        '''
        config = self.partition_mapping.get(pid)
        if config['sub_parts'] == list():  # this means that partition config allows sub_partitions
            sub_partition = self.create_partition(pid)
            if sub_partition:
                config['sub_parts'].append(sub_partition)


    def get_partition(self, pid, sub_part=-1):
        '''
        Returns the buffer structure (possibly containing data) for a pid.
        If sub_part >= 0, returns the sub-partition if available.

        Args
        ----------
        pid : int
            Specification of the partition identifier.

        sub_part : int
            If sub_part == -1, do not query for sub_part, if sub_part >= 0, query sub_partition of partition.
        '''
        buffer = None
        sub_buffer = None

        if self.buffer[pid]: buffer = self.buffer[pid]
        else: raise Exception(f'can not query partition {pid}, it does not exist...')

        if sub_part >= 0:
            if self.buffer[pid][sub_part]:
                if sub_part < len(self.buffer[pid][sub_part]):
                    sub_buffer = self.buffer[pid][sub_part]
                else: raise Exception(f'can not query sub_partition of partition {pid} with sub-index {sub_part}, out of bounds...')
            else: raise Exception(f'can not query sub_partition of partition {pid} with sub-index {sub_part}, it does not exist...')

        if sub_buffer: return sub_buffer
        else: return buffer


    def get_pid_for_partition_type(self, p_type='none'):
        '''
        Returns the pid for a given partition type

        Args
        ----------
        p_type : Partition_Type
            String or Partition_Type to search for e.g. 'inliers'
        '''
        for pid in self.partition_mapping.values():
            if self.partition_mapping[pid]['type'] == p_type: return pid


    def add_to_partition(self, data, pid, sub_part=-1):
        '''
        Adds the arriving data to a partition

        Args
        ----------
        pid : int

        data : (numpy.ndarray|tensorflow.Constant)
            We assume data arrives as a tuple of dims ([sample_index, H, W, C], [sample_index, num_classes])

        sub_part : int
            If sub_part == -1, partition found via pid is targeted.
            If sub_part >= 0, check if sub_partition exists and add data to it.

        '''
        #TODO: data handling -> check if data matches with dataset_type if not: convert or just reject???
        #  self.subtask_data[subtask_index][0][:]        = x



    def draw_from_partition(self, pid, N, sub_part=-1):
        '''
        Draws N samples from a single partition.

        Parameters
        ----------
        pid : int
            Specification of the partition identifier.

        N : int
            Number of samples to be drawn fromm the specified partitions.

        sub_part : int
            If sub_part == -1, partition found via pid is targeted.
            If sub_part >= 0, check if sub_partition exists and draw data from it.

        Returns
        -------
        draw: np.Array or tf.Constant
            returns np.Array or tf.Constant - a tuple of (xs,ys) with consistent dimensions from the merged data of
            all partitions, if a single partition has specified labeled=False -> ys is None
        '''
        #TODO draw N from a normal distribution np.random.standard_normal()
        draw = None

        return draw


    def draw_from_partitions(self, pids, N, proportions):
        '''
        Draws N samples from multiple partitions.

        Parameters
        ----------
        pids : list [int]
            Specification of partition identifiers data is drawn from, e.g. [0, 1].
        N : int
            Number of samples to be drawn from all specified partitions.
        proportions : list [int]
            Sample distribution from specified partitions, e.g. [50,50] if drawn from 2 partitions, has to match
            with the count of partitions specified in pids.

        Returns
        -------
        draw: np.Array or tf.Constant
            returns merged np.Array or tf.Constant consisting of a tuple of (xs,ys) with consistent dimensions,
            if partitions have specified labeled=False -> ys is None
        '''
        # TODO: how to draw from multiple partitions/sub-partitions, think about a mapping
        draw = None

        return draw


    def flush_partition(self, pid, sub_part=-1):
        '''
        Flushes the partition after the data has been merged with Data_Generator.

        Parameters
        ----------
        pid : int
            Specification of the partition identifier.

        sub_part : int
            If sub_part == -1, partition found via pid is targeted.
            If sub_part >= 0, check if sub_partition exists and flush it instead of super partition.
        '''


    def check_buffers(self):
        '''
        Called when we get a signal that a buffer has reached its maximum volume (threshold).
        We check all buffers for their sample_size and flush buffers after merging them into the Data_Generator.
        '''


    def merge_partitions(self, pid_from, pid_to):
        ''' Merge data from one partition with another partition'''
        #TODO: no need for now, maybe later
        pass


    def remove_partition(self, partition_key):
        ''' remove partition from the buffer '''
        #TODO: no need for now, maybe later
        pass
