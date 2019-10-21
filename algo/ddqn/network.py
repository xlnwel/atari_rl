import numpy as np
import tensorflow as tf

from algo.ddqn.base_net import Network, select_action


class DuelDQN(Network):
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
        super().__init__(name, 
                         args, 
                         graph, 
                         data,
                         n_actions, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):
        net_fn = lambda obs, name, reuse=False: self._net(obs, self.n_actions, name=name, reuse=reuse)
    
        Qs = net_fn(self.obs, 'main')
        Qs_next = net_fn(self.next_obs, 'main', reuse=True)
        Qs_next_target = net_fn(self.next_obs, 'target')

        self.best_action = select_action(Qs, name='best_action')
        next_action = select_action(Qs_next, name='next_action')

        with tf.name_scope('q_value'):
            self.Q = tf.reduce_sum(tf.one_hot(self.action, self.n_actions) * Qs, axis=1, keepdims=True)
            self.Q_next_target = tf.reduce_sum(tf.one_hot(next_action, self.n_actions) 
                                                * Qs_next_target, axis=1, keepdims=True)

    def _net(self, x, out_dim, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            x = self._conv_net(x, 'conv_net')
            v = self._head_net(x, 1, 'value_net')
            a = self._head_net(x, out_dim, 'adv_net')

            with tf.name_scope('q'):
                q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return q