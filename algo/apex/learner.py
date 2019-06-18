import os
import numpy as np
import threading
import tensorflow as tf
import ray

from utility import tf_utils
from utility.utils import pwc

def get_learner(BaseClass, *args, **kwargs):

    @ray.remote(num_gpus=1, num_cpus=2)
    class Learner(BaseClass):
        """ Interface """
        def __init__(self, 
                    name, 
                    args, 
                    env_args,
                    buffer_args,
                    sess_config=None, 
                    save=True, 
                    log_tensorboard=True, 
                    log_params=True,
                    log_stats=True,
                    device=None):
                    
            super().__init__(name, 
                            args, 
                            env_args,
                            buffer_args,
                            sess_config=sess_config,
                            save=save,
                            log_tensorboard=log_tensorboard,
                            log_params=log_params,
                            log_stats=log_stats,
                            device=device)
            
            self.learning_thread = threading.Thread(target=self.background_learning, daemon=True)
            self.learning_thread.start()
            
            pwc('Learner has been constructed.', color='green')
            
        def get_weights(self):
            weights = self.variables.get_flat()

            return weights

        def merge_buffer(self, local_buffer, length):
            self.buffer.merge(local_buffer, length)
            pwc(f'Buffer merged: {len(self.buffer)}')

        def record_stats(self, kwargs):
            assert isinstance(kwargs, dict)
            super()._record_stats_impl(kwargs)

    return Learner.remote(*args, **kwargs)
