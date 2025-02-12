import tensorflow as tf

from .                      import Regularizer
from cl_replay.api.utils    import log



class NewReg(Regularizer):
    ''' A very simple regularizer instance.
    Strategy:
        - accumulate the avg. loss values over so-called "loops" of length 2/alpha iterations
        - compute slope of averaged loss w.r.t. averaged loss at the end of the last loop
        - if slope is < delta * < first slope after most recent sigma update > --> update sigma
    '''
    def __init__(self, tf_eps, tf_somSigma, eps_0, somSigma_0, somSigma_inf, eps_inf, **kwargs):
        super().__init__(**kwargs)
        # set the internal values coming from the gmm layer
        self.tf_eps         = tf_eps
        self.tf_somSigma    = tf_somSigma
        self.eps_0          = eps_0
        self.somSigma_0     = somSigma_0
        self.somSigma_inf   = somSigma_inf
        self.eps_inf        = eps_inf
        # regularizer settings
        self.alpha          = self.parser.add_argument('--alpha',       type=float, default=self.eps_inf,        help='reaction speed (higher is slower)')
        self.gamma          = self.parser.add_argument('--gamma',       type=float, default=0.9,                 help='reduction factor of somSigma')
        self.delta          = self.parser.add_argument('--delta',       type=float, default=0.5 ,                help='stationarity detection threshold')
        self.reset_sigma    = self.parser.add_argument('--reset_sigma', type=float, default=self.somSigma_inf,   help='reset value for sigma')  # default, reset to somSigma_inf, even if not yet completely reduced
        self.reset_eps      = self.parser.add_argument('--reset_eps',   type=float, default=self.eps_inf,        help='reset value for eps')  # default, reset to
        # tracking vars
        self.avg_loss       = None
        self.last_avg       = None
        self.ref_avg        = None
        self.ref_loop       =  -1
        # global iteration counter, NOT reset after each loop
        self.iteration      = 0
        # episode/loop length = inverse of learning rate = iterations between two regularizer check events
        self.W              = int(1. / self.alpha)
        self.limit          = None
        self.current_sigma  = self.somSigma_0


    def add(self, loss):
        ''' Register a convertible-to-python float object. '''
        it_in_loop      = self.iteration % self.W
        if it_in_loop   == 0: self.avg_loss = 0.0
        # incrementally compute average loss in loop. At last iteration in loop, self.avg_loss will contain exact average
        self.avg_loss   *= it_in_loop / (it_in_loop + 1.)
        self.avg_loss   += tf.reduce_mean(loss) / (it_in_loop + 1.)
        self.iteration  += 1
        # log.debug(f"added iter [{self.iteration}] ... checking each {self.W} steps - loss (cur.): {loss}  loss (avg.): {self.avg_loss} loss (last avg.): {self.last_avg}...\n")

        return self


    def set(self, eps=None, sigma=None):
        ''' Assign new or default values to eps and sigma. '''
        reset_eps      = eps   if eps      else self.reset_eps
        reset_sigma    = sigma if sigma    else self.reset_sigma

        reset_eps      = reset_eps    if reset_eps      > self.eps_inf       else self.eps_inf
        reset_sigma    = reset_sigma  if reset_sigma    > self.somSigma_inf  else self.somSigma_inf
        
        log.debug(f"set sigma of {self.name} to: {reset_sigma}...")

        self.tf_somSigma.assign(reset_sigma)
        self.tf_eps.assign(reset_eps)


    """
    self.avg_loss - current loss average for current loop
    terms: iteration - 1 mini-batch update
           loop - fixed interval of self.W iterations
           cycle - variable interval between two somsigma updates
    _check is called after each iteration
    one cycle is at least 2 loops long: before loop 0, self.l0 is taken from prevous cycle. loop 0 for ref.l1, and loop >0 for free evaluation of slopes
    """
    def _check(self):
        if self.iteration % self.W != self.W - 1:  return  # we never do anything except at the last iter. of a loop
        self.last_avg = self.avg_loss ;

        loop_index = (self.iteration // self.W)

        # at the end of the first ever loop, we initialize self.l0
        if loop_index == 0:
            self.start_cycle_loop_index = 0
            self.avgs = [self.avg_loss]
            return

        loop_in_cycle = loop_index - self.start_cycle_loop_index

        self.avgs.append(self.avg_loss)
 
        # loop #0 is to build up avg: first meaningful avg @ end of loop 0
        if loop_in_cycle >= 2:  # 3rd or more loop after update
            l0 = self.avgs[0] 
            l1 = self.avgs[loop_in_cycle // 2]
            delta_right = (l1 - self.avg_loss)
            delta_left = (l0 - l1)
            
            log.debug(f"checking slope... last avg. loss: {self.avg_loss}, L0={l0}, L1={l1}, (slope/ref_slope) {delta_right / delta_left} < delta: {self.delta}...?")
            if (delta_right / delta_left) < self.delta: # TODO self.delta
                self.start_cycle_loop_index = loop_index # reset cycle!!
                self.avgs = [self.avg_loss]

                currentEps, self.current_sigma = [self.tf_eps.numpy(), self.tf_somSigma.numpy()]
                currentEps          *= self.gamma
                self.current_sigma  *= self.gamma
                self.set(eps=currentEps, sigma=self.current_sigma)
