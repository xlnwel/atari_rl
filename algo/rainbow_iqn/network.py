import numpy as np
import tensorflow as tf

from algo.ddqn.base_net import Network, select_action


class RainbowIQN(Network):
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
        self.N = args['N']                          # N in paper, num of quantiles for online quantile network
        self.N_prime = args['N_prime']              # N' in paper, num of quantiles for target quantile network
        self.K = args['K']                          # K in paper, num of quantiles for action selection
        self.delta = args['delta']                  # 𝜅 in paper, used in huber loss
        
        super().__init__(name, 
                         args, 
                         graph, 
                         data,
                         n_actions, 
                         scope_prefix=scope_prefix,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params)

    """ Implementation """
    def _build_graph(self):
        net_fn = (lambda obs, n_quantiles, batch_size, name, reuse=False: 
                        self._iqn_net(obs, 
                                    n_quantiles, 
                                    batch_size, 
                                    self.n_actions,
                                    self._psi_net,
                                    self._phi_net,
                                    self._f_net,
                                    name=name, 
                                    reuse=reuse))
        # online IQN network                            # [B, N, 1], [N, B, A], [B, A]
        quantiles, quantile_values, Qs = net_fn(self.obs, self.N, self.batch_size, 'main')
        # Qs for online action selection                # [1, K, 1], [K, 1, A], [1, A]
        _, _, Qs_online = net_fn(self.obs, self.K, 1, 'main', reuse=True)
        # target IQN network                            # [B, N', 1], [N', B, A], [B, A]
        _, quantile_values_next_target, Qs_next_target = net_fn(self.next_obs, self.N_prime, self.batch_size, 'target')
        # next online Qs for double Q action selection  # [B, K, 1], [K, B, A], [B, A]
        _, _, Qs_next = net_fn(self.next_obs, self.K, self.batch_size, 'main', reuse=True)
        
        self.quantiles = quantiles                                              # [B, N, 1]

        self.best_action = select_action(Qs_online, 'best_action')              # [1]
        next_action = select_action(Qs_next, 'next_action')                     # [B]

        # quantile_values for regression loss
        # Q for priority required by PER
        self.quantile_values, self.Q = self._iqn_values(self.action, self.N, quantile_values, Qs)
        self.quantile_values_next_target, self.Q_next_target = self._iqn_values(next_action, 
                                                                                self.N_prime,
                                                                                quantile_values_next_target, 
                                                                                Qs_next_target)

    def _iqn_net(self, x, n_quantiles, batch_size, out_dim, 
                psi_net, phi_net, f_net, name, reuse=None):
        quantile_embedding_dim = self.args['quantile_embedding_dim']

        with tf.variable_scope(name, reuse=reuse):
            # 𝜓 function in the paper
            x_tiled = psi_net(x, n_quantiles)
            
            h_dim = x_tiled.shape.as_list()[1]

            # 𝜙 function in the paper
            quantiles, x_quantiles = phi_net(n_quantiles, batch_size, quantile_embedding_dim, h_dim)
            # Combine outputs of psi and phi
            y = x_tiled * x_quantiles
            # f function in the paper
            v_qv, v = f_net(y, 1, n_quantiles, batch_size, name='value_net')
            a_qv, a = f_net(y, out_dim, n_quantiles, batch_size, name='adv_net')

            with tf.name_scope('q'):
                quantile_values = v_qv + a_qv - tf.reduce_mean(a_qv, axis=2, keepdims=True)
                q = v + a - tf.reduce_mean(a, axis=1, keepdims=True)

        return quantiles, quantile_values, q

    def _psi_net(self, x, n_quantiles):
        with tf.variable_scope('psi_net'):
            x = self._conv_net(x)
            x_tiled = tf.tile(x, [n_quantiles, 1])
        
        return x_tiled

    def _phi_net(self, n_quantiles, batch_size, quantile_embedding_dim, h_dim):
        with tf.name_scope('quantiles'):
            quantile_shape = [n_quantiles * batch_size, 1]
            quantiles = tf.random.uniform(quantile_shape, minval=0, maxval=1)       # [N*B, 1]
            quantiles_tiled = tf.tile(quantiles, [1, quantile_embedding_dim])       # [N*B, D]
            # returned quantiles for computing quantile regression loss
            quantiles_reformed = tf.transpose(tf.reshape(quantiles, [n_quantiles, batch_size, 1]), [1, 0, 2])
            
        with tf.variable_scope('phi_net'):
            pi = tf.constant(np.pi)
            degrees = tf.cast(tf.range(quantile_embedding_dim), tf.float32) * pi * quantiles_tiled
            x_quantiles = tf.cos(degrees)
            x_quantiles = tf.layers.dense(x_quantiles, h_dim)
            x_quantiles = tf.nn.relu(x_quantiles)

        return quantiles_reformed, x_quantiles

    def _f_net(self, x, out_dim, n_quantiles, batch_size, name=None):
        name = f'{name}_f_net' if name else 'f_net'
        with tf.variable_scope(name):
            quantile_values = self._head_net(x, out_dim)
            quantile_values = tf.reshape(quantile_values, (n_quantiles, batch_size, out_dim))
            q = tf.reduce_mean(quantile_values, axis=0)

        return quantile_values, q
        
    def _iqn_values(self, action, n_quantiles, quantile_values, Qs):
        with tf.name_scope('action_values'):
            action_tiled = tf.reshape(tf.tile(action, [n_quantiles]), 
                                        [n_quantiles, -1])
            quantile_values = tf.reduce_sum(tf.one_hot(action_tiled, self.n_actions)
                                            * quantile_values, axis=2, keepdims=True)
            q = tf.reduce_sum(tf.one_hot(action, self.n_actions)
                              * Qs, axis=1, keepdims=True)

        return quantile_values, q
