import tensorflow as tf

from basic_model.model import Module


class Base(Module):
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.variable_scope = f'{scope_prefix}/{name}'
        
        super().__init__(name, args, graph, log_tensorboard=log_tensorboard, log_params=log_params)
