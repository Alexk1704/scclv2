import os
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.callbacks import Callback
 
from cl_replay.api.parsing  import Kwarg_Parser
from cl_replay.api.utils    import log


class Log_Metrics(Callback):
    ''' Implements a metric logging callback to save the evaluation/test data into .csv files utilizing pandas dataframes. ''' 

    def __init__(self, **kwargs):
        super(Log_Metrics, self).__init__()

        parser = Kwarg_Parser(**kwargs)
        self.log_path           = parser.add_argument('--log_path',         type=str,       required=True)
        self.log_training       = 'no'
        # self.log_training       = parser.add_argument('--log_training',     type=str,       choices=['yes', 'no'], default='no')
        # self.dump_after_train   = parser.add_argument('--dump_after_train', type=str,       choices=['yes', 'no'], default='no')
        # self.dump_after_test    = parser.add_argument('--dump_after_test',  type=str,       choices=['yes', 'no'], default='no')
        if os.path.isabs(self.log_path) == False: log.error("--log_path must be absolute!")
        self.log_path           = os.path.join(self.log_path, "metrics")
        if not os.path.exists(self.log_path): os.makedirs(self.log_path)
        log.debug(f'metrics logging path: {self.log_path}')
        self.exp_id             = kwargs.get('exp_id', None)
        self.DAll               = kwargs.get('DAll', None)
        self.extra_eval         = kwargs.get('extra_eval', [])
        self.single_class_test  = kwargs.get('single_class_test', 'no')
        self.num_tasks          = kwargs.get('num_tasks', None)
        self.forgetting_tasks   = kwargs.get('forgetting_tasks', None)
        self.forgetting_mode    = kwargs.get('forgetting_mode', None)
        if type(self.num_tasks) is str: self.num_tasks = int(self.num_tasks)
        self.eval_during_train  = kwargs.get('eval_during_train', 'no')
        self.full_eval          = kwargs.get('full_eval', 'no')
        
        self.current_task       = int(kwargs.get('load_task', 0))
        self.batch_ctr          = 0
        
        self.test_metric_names, self.test_metric_values = [], []
        self.train_metric_names, self.train_metric_values = [], []
        
        self.append_once        = False
        self.train              = False
        self.run_ok             = False
        

    def __del__(self):
        if self.run_ok == False: return
        # self.dump_to_csv()
        self.create_pkl_matrices()


    def dump_to_csv(self, mode='test'): 
        if mode == 'test':
            # if not self.append_once:
            #     self.test_metric_names.extend(['num_tasks', 'num_metrics'])
            #     self.test_metric_values.extend([self.current_task, len(self.model.metrics)])
            #     self.append_once = True
            data = [np.array(self.test_metric_values)]
            cols = self.test_metric_names
        else:
            data = [np.array(self.train_metric_values)]
            cols = self.train_metric_names
        
        df_new = pd.DataFrame(columns=cols, data=data)

        self.fname = os.path.join(self.log_path, f"{self.exp_id}_{mode}.csv")

        if os.path.exists(self.fname): # join data
            df_exist = pd.read_csv(self.fname, index_col=0)
            df_concat = pd.concat([df_exist, df_new], axis=1)
            df_concat.to_csv(self.fname)
        else: # write new
            df_new.to_csv(self.fname)

    
    def create_pkl_matrices(self):
        """
        Acc. Matrix:
            T1 T2 .. TN
        T1  
        T2
        ..
        TN        
        - each row represents a training task
        - each col represents the accuracy on that task
        
        Forgetting Matrix: 
        F_ij = max_{i \in 1...T-1} a_ij - a_Tj    forall j < T
        - change in performance of task i after learning task j.
        """
        if self.full_eval == 'no' or self.single_class_test == 'yes' or self.num_tasks < 2: return
        
        labels  = np.array(self.test_metric_names)
        vals    = np.array(self.test_metric_values)
        
        # if self.forgetting_mode == 'mixed' and self.forgetting_tasks != []:
        #     exclude_list = []
        #     for task_id in self.forgetting_tasks: exclude_list.append(f'T{task_id}') 
            
        #     exclude_indices = []
        #     for i, test_label in enumerate(labels):
        #         exclude_entry = False
        #         for exclude_str in exclude_list:
        #             if exclude_str in test_label: exclude_entry = True
        #         if not exclude_entry: exclude_indices.append(i)
            
        #     labels_ = labels[exclude_indices]
        #     vals_ = vals[exclude_indices]

        #     self.num_tasks -= len(self.forgetting_tasks)
        # else:
        
        labels_ = labels
        vals_ = vals    

        num_metrics = len(self.model.metrics)
        if 'step_time' in self.model.metrics[-1].name:
            num_metrics -= 1
            
        start_index = 0
        for i, metric_name in enumerate(labels[:num_metrics+1], start=0):
            if 'acc' in metric_name: start_index = i; break
        
        acc = vals_[start_index::num_metrics]
        acc_l = labels_[start_index::num_metrics]

        # --- ACC / LOSS matrices
        acc_mat = None
        # loss_mat = None
        
        i = 0
        step = self.num_tasks+2 if self.extra_eval != [] else self.num_tasks+1 
        max = acc_l.shape[0]    
        while(i < max):
            acc_row = acc[(i+1):(i+step)]
            # loss_row = loss[(i+1):(i+step)]
            
            if type(acc_mat) is type(None): 
                acc_mat = acc_row
                # loss_mat = loss_row
            else: 
                acc_mat = np.vstack((acc_mat, acc_row))
                # loss_mat = np.vstack((loss_mat, loss_row))
            i += step

        # --- FORG matrix
        forg_mat = np.zeros(shape=(self.num_tasks, self.num_tasks), dtype=np.float32)
        for i in range(0, self.num_tasks):
            for j in range(0, self.num_tasks):
                if i != j:
                    forg_mat[j,i] = acc_mat[i,i] - acc_mat[j,i]
        print(acc_mat, '\n', forg_mat)
        
        acc_fname = os.path.join(self.log_path, f"{self.exp_id}_accmat.npy")
        forg_fname = os.path.join(self.log_path, f"{self.exp_id}_forgmat.npy")

        np.save(acc_fname, acc_mat, allow_pickle=False)
        np.save(forg_fname, forg_mat, allow_pickle=False)
    
    
    def save_train_logs(self):
        # for m_name, m_val in self.train_metrics.items():
        #     print(m_name)
        #     print(m_val)    
        
        if self.forgetting_tasks != []:
            if str(self.current_task) in self.forgetting_tasks: 
                return
        m_names = np.array(list(self.train_metrics.keys()), dtype='U12')
        m_vals = np.array(list(self.train_metrics.values()), dtype=np.float32)
    
        save_names = f'{self.exp_id}_{self.current_task}_treval_names.npy'
        save_vals = f'{self.exp_id}_{self.current_task}_treval_vals.npy'
        
        np.save(os.path.join(self.log_path, save_names), m_names, allow_pickle=False)
        np.save(os.path.join(self.log_path, save_vals), m_vals, allow_pickle=False)


    def on_train_begin(self, logs=None):
        # ---- reset
        self.current_task += 1
        self.batch_ctr = 0
        self.train = True
        
        if self.eval_during_train == 'yes': self.train_metrics = {}


    def on_batch_end(self, batch, logs=None):
        self.batch_ctr += 1


    def on_epoch_end(self, epoch, logs=None):
        if self.log_training == 'no': return
        
        # FIXME: bad practice; find a better way for generic metrics...
        if 'step_time' in self.model.metrics[-1].name:
            avg_step_time = self.model.metrics[-1].result().numpy()
            epoch_duration = avg_step_time * self.batch_ctr
            self.train_metric_names.extend(
                [f"train_T{self.current_task}-E{epoch}_{self.model.name}_duration"]
            )
            self.train_metric_values.extend(
                [epoch_duration]
            )
            all_metrics = self.model.metrics[:-1]
        else:
            all_metrics = self.model.metrics

        self.train_metric_names.extend([f"train_T{self.current_task}-E{epoch}_{self.model.name}_" + m.name for m in all_metrics])
        self.train_metric_values.extend([m.result().numpy() for m in all_metrics])
        
        self.batch_ctr = 0


    def on_train_end(self, logs=None):
        self.run_ok = True
        self.train = False
        # ---- saves eval data gathered during training
        if self.eval_during_train == 'yes':
            # training done after task TX, serialize results!
            self.save_train_logs()
            # clear
            self.train_metrics = {}
            self.train_metric_names, self.train_metric_values = [], []
 
        # if self.dump_after_train == 'yes': self.dump_to_csv(mode='train')

 
    def on_test_begin(self, logs=None): 
        self.test_batch_ctr = 0
        self.meaned_test_metrics = {}


    def on_test_batch_end(self, batch, logs=None):
        self.test_batch_ctr += 1.
        
        if 'step_time' in self.model.metrics[-1].name:
            all_metrics = self.model.metrics[:-1]
        else:
            all_metrics = self.model.metrics
        
        for m in all_metrics:
            m_k = f"T{self.current_task}-{self.model.test_task}-{m.name}"
            m_v = m.result().numpy()
            if m_k in self.meaned_test_metrics:
                self.meaned_test_metrics[m_k] += m_v
            else:
                self.meaned_test_metrics.update({m_k : m_v})


    def on_test_end(self, logs=None):
        self.run_ok = True
        
        for m_k, m_v in self.meaned_test_metrics.items():
            m_v /= self.test_batch_ctr
            if self.train:
                self.train_metric_names.extend([m_k])
                self.train_metric_values.extend([m_v])
            else:
                self.test_metric_names.extend([m_k])
                self.test_metric_values.extend([m_v])
                
            if self.eval_during_train == 'yes':
                if not m_k in self.train_metrics:
                    self.train_metrics.update({ m_k : [m_v] })
                else:
                    self.train_metrics.update({ m_k : self.train_metrics[m_k] + [m_v]})
        
        # if self.dump_after_test == 'yes': self.dump_to_csv()
        