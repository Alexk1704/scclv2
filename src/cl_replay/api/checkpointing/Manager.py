import os, sys
import tensorflow as tf

from cl_replay.api.utils    import log
from cl_replay.api.parsing  import Kwarg_Parser


class Manager:
    '''  A manager supporting the saving/loading of training progress (model vars/weights) to the file system. '''

    def __init__(self, **kwargs):
        parser          = Kwarg_Parser(**kwargs)

        self.exp_id     = kwargs.get('exp_id', None)
        self.model_type = kwargs.get('model_type', None)
        default_path    = os.path.expanduser('~')
        self.ckpt_dir   = parser.add_argument('--ckpt_dir', type=str, default=f'{default_path}/chkpt', help='directory for checkpoint files')
        self.load_ckpt_from = parser.add_argument('--load_ckpt_from', type=str, help='provide custom checkpoint file path (omit .ckpt).')
        if os.path.isabs(self.ckpt_dir) == False:
            log.error("--chkpt_dir must be an absolute path!")
            sys.exit(0)
        self.ckpt_dir  = os.path.join(self.ckpt_dir, "checkpoints")
        if not os.path.exists(self.ckpt_dir): os.makedirs(self.ckpt_dir)
        self.filename   = os.path.join(self.ckpt_dir, f'{self.exp_id}-{self.model_type.split(".")[-1].lower()}-{{}}.weights.h5')

        self.load_task  = kwargs.get('load_task', 0)
        self.save_All   = kwargs.get('save_All', 'yes')

    def load_checkpoint(self, model, task = None, **kwargs):
        ''' Load a model configuration via the checkpoint manager. '''
        if task is None: task = int(self.load_task)
 
        if task <= 0           : return 0, model

        if self.load_ckpt_from:
            ckpt_file = self.load_ckpt_from + f'-{{}}.weights.h5'
            ckpt_file = ckpt_file.format(task)
        else:
            ckpt_file = self.filename.format(task)
        try:
            model.load_weights(ckpt_file)
            log.info(f'restored model: {model.name} from checkpoint file "{ckpt_file}"...')
        except Exception as ex:
            log.error(f'a problem was encountered loading the model: {model.name} from checkpoint file "{ckpt_file}": {ex}')
            self.load_task = 0
            raise ex

        return task, model

    def save_checkpoint(self, model, current_task, **kwargs):
        ''' Saves the current session state to the file system. '''
        if self.save_All == False: return

        try:
            chkpt_filename = self.filename.format(current_task)
            model.save_weights(chkpt_filename)
            self.model_name = model.name
            log.info(f'saved model weights of "{self.model_name}" after task T{current_task} to file "{chkpt_filename}"')
        except Exception as ex:
            log.error(f'a problem was encountered saving the checkpoint file for model: {self.model_name} after task T{current_task}...')
            raise ex
