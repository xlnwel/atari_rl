import tensorflow as tf
# from ray.experimental.tf_utils import TensorFlowVariables

from algo.rainbow_iqn.network import RainbowIQN
from algo.ddqn.agent import Agent as DDQN
from utility.losses import huber_loss
from utility.utils import pwc
from utility.tf_utils import n_step_target, stats_summary


class Agent(DDQN):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=False, 
                 log=False,
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None,
                 reuse=None):
        super().__init__(name, args, 
                         env_args, buffer_args,
                         sess_config=sess_config, 
                         save=save, 
                         log=log,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device,
                         reuse=reuse)

    def _create_nets(self):
        self.args['Qnets']['batch_size'] = self.args['batch_size']
        Qnets = RainbowIQN('Nets', 
                            self.args['Qnets'], 
                            self.graph, 
                            self.data, 
                            self.n_actions,    # n_actions
                            scope_prefix=self.name,
                            log_tensorboard=self.log_tensorboard,
                            log_params=self.log_params)
        
        return Qnets

    def _loss(self):
        def tiled_n_step_target():
            n_quantiles = self.args['Qnets']['N_prime']
            
            reward = self.data['reward'][None, ...]
            done = self.data['done'][None, ...]
            steps = self.data['steps'][None, ...]
            return n_step_target(reward, done, 
                                 self.Qnets.quantile_values_next_target,
                                 self.gamma, steps)
            
        def quantile_regression_loss(u):
            # [B, N, N']
            abs_part = tf.abs(self.Qnets.quantiles - tf.where(u < 0, tf.ones_like(u), tf.zeros_like(u)))
            
            huber = huber_loss(u, delta=self.args['Qnets']['delta'])
            
            qr_loss = tf.reduce_sum(tf.reduce_mean(abs_part * huber, axis=2), axis=1)   # [B]
            loss = tf.reduce_mean(self.data['IS_ratio'] * qr_loss)

            return loss

        _, priority = self._compute_priority(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])

        with tf.name_scope('loss'):
            quantile_values_target = tiled_n_step_target()                              # [N', B, 1]
            quantile_values_target = tf.transpose(quantile_values_target, [1, 2, 0])    # [B, 1, N']
            quantile_values = tf.transpose(self.Qnets.quantile_values, [1, 0, 2])       # [B, N, 1]
            quantile_error = quantile_values_target - quantile_values                   # [B, N, N']

            loss = quantile_regression_loss(quantile_error)

        return priority, loss

    def _log_info(self):
        with tf.device('/CPU:0'):
            if self.log_tensorboard:
                with tf.name_scope('info'):
                    if self.buffer_type == 'proportional':
                        stats_summary('priority', self.priority, max=True, std=True)
                    stats_summary('Q', self.Qnets.Q, max=True)
                    tf.summary.scalar('loss_', self.loss)
