import numpy as np



class DERpp_Buffer:
    
    
    def __init__(self):
        self.dtype_np_float = np.float32
        self.dtype_np_int = np.int32
        self.H, self.W, self.C, self.num_classes = data_dims

        self.storage_budget = 0
        
    
    def __init_buffers__(self, storage_budget):
        indices = np.arange(0, storage_budget, dtype=self.dtype_np_int)
        samples = np.zeros(shape=(storage_budget, self.H, self.W, self.C), dtype=np.float32)
        labels  = np.zeros(shape=(storage_budget, self.num_classes), dtype=np.float32)
        stats   = np.zeros(shape=(storage_budget, 2), dtype=self.dtype_np_float)
        
        self.buffer = dict()
        
        for i in indices:
            self.buffer.update({ i: { 'x': samples[i], 'y': labels[i], 'stats': stats[i] }})
        print(self.buffer)
        
        
    def get_buffer_stats(self):
        ''' return stats about buffer population. '''
        #TODO: empty space -> check how many samples are zero
        #TODO: stats -> min./max./avg. loss, misses & hits total
        #TODO: classes -> how many samples of each class are saved currently
        
    
    
    def save_to_buffer(self, data):
        ''' expects a mini-batch of data (x,y,stats), whereas stats contains a 2d numpy array holding loss and acc. '''
        
        # TODO: capacity available: save data
        
        # TODO: set a threshold for inclusion: samples above inclusion threshold are always kept since they represent challenging cases
        # Q: How to pick threshold? Static threshold makes no sense... Better to keep it dynamic (e.g. 10% of samples with highest loss)
        
        # TODO: if capacity full: dynamic reallocation based on classes
        
        # TODO: store samples with high loss and/or misses with a higher probability! (prioritized population of buffer)
        
        
    def sample_from_buffer(self, amount, der_ratio, der_strategy):
        ''' samples from buffer based on the der_ratio (%) and strategy ("acc" or "loss"), other classes are sampled randomly. 
        e.g. 80% common experiences from all classes, 20% dark experiences. '''
        # TODO: always take a fixed portion of high loss or misclassified samples