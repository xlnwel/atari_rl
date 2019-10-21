import numpy as np
import tensorflow as tf

from utility.tf_utils import norm_activation, wrap_layer, get_norm
from basic_model.model import Module


class Network(Module):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 graph,
                 data,
                 n_actions, 
                 scope_prefix='',
                 log_tensorboard=False,
                 log_params=False):
        self.obs = data['obs']
        self.action = data['action']
        self.next_obs = data['next_obs']
        self.reward = data['reward']
        self.n_actions = n_actions
        self.batch_size = args['batch_size']
        self.conv_norm = get_norm(args['conv_norm'])
        self.dense_norm = get_norm(args['dense_norm'])

        self.fixup = args['fixup']
        self.use_noisy = args['noisy']

        super().__init__(name, 
                         args, 
                         graph, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    @property
    def main_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/main')
    
    @property
    def target_variables(self):
        return self.graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope + '/target')

    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError

    def _conv_net(self, x, name=None):
        return self.impala_cnn(x, fixup=self.fixup, conv_norm=self.conv_norm, dense_norm=self.dense_norm, name=name)

    def _head_net(self, x, out_dim, name=None):
        def net(x, out_dim):
            layer = self.noisy if self.use_noisy else self.dense
            name_fn = lambda i: f'noisy_{i}' if self.use_noisy else f'dense_{i}'
            x = layer(x, 512, name=name_fn(1))
            x = norm_activation(x, norm=self.dense_norm, activation=tf.nn.relu)
            x = layer(x, out_dim, name=name_fn(2))
            return x
        
        return wrap_layer(name, lambda: net(x, out_dim))


def select_action(Qs, name):
    with tf.name_scope(name):
        action = tf.argmax(Qs, axis=1, name='best_action')
    
    return action