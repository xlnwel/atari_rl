import numpy as np
import tensorflow as tf

from algo.ddqn.base_net import Network, select_action


class Rainbow(Network):
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
        self.n_atoms = args['n_atoms']
        self.v_min = float(args['v_min'])
        self.v_max = float(args['v_max'])
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        self.z_support = tf.linspace(self.v_min, self.v_max, self.n_atoms)

        super().__init__(name, 
                         args, 
                         graph, 
                         data,
                         n_actions, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    def _build_graph(self):    
        net_fn = lambda obs, name, reuse=False: self._c51_net(obs, self.n_actions, name=name, reuse=reuse)
        # [B, A, N], [B, A]
        logits, dist, Qs = net_fn(self.obs, 'main')
        _, _, Qs_next = net_fn(self.next_obs, 'main', reuse=True)
        _, dist_next_target, _ = net_fn(self.next_obs, 'target')

        self.best_action = select_action(Qs, 'best_action')
        next_action = select_action(Qs_next, 'next_action')

        # [B, N], [B, N]
        self.logits = self._c51_action_value(self.action, logits, 'logits')
        self.dist_next_target = self._c51_action_value(next_action, dist_next_target, 'dist_next_target')
        # for tensorboard bookkeeping
        self.dist = self._c51_action_value(self.best_action, dist, 'best_prob')
        self.Q = self._c51_action_value(self.best_action, Qs, 'best_Q') 

    def _c51_net(self, x, out_dim, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            x = self._conv_net(x, 'conv_net')
            v_logits = self._c51_head(x, 1, 'value_net')
            a_logits = self._c51_head(x, out_dim, 'adv_net')

            with tf.name_scope('q_logits'):
                q_logits = v_logits + a_logits - tf.reduce_mean(a_logits, axis=1, keepdims=True)
            
            with tf.name_scope('q'):
                q_dist = tf.nn.softmax(q_logits, axis=-1)               # [B, A, N]
                q = tf.reduce_sum(self.z_support * q_dist, axis=-1)     # [B, A]
            
        return q_logits, q_dist, q

    def _c51_head(self, x, out_dim, name):
        with tf.variable_scope(name):
            x = self._head_net(x, out_dim*self.n_atoms)
            logits = tf.reshape(x, (-1, out_dim, self.n_atoms))         # [B, A, N]

        return logits

    def _c51_action_value(self, action, values, name):
        with tf.name_scope(name):
            action = action[..., None]
            value = tf.gather_nd(values, action, batch_dims=1)

        return value
