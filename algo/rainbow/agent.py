import tensorflow as tf
# from ray.experimental.tf_utils import TensorFlowVariables

from algo.rainbow.network import Rainbow
from algo.ddqn.agent import Agent as DDQN
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
        Qnets = Rainbow('Nets', 
                        self.args['Qnets'], 
                        self.graph, 
                        self.data, 
                        self.n_actions,    # n_actions
                        scope_prefix=self.name,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)
        

        return Qnets

    def _loss(self):
        with tf.name_scope('loss'):
            # Eq.7 in c51 paper
            v_min, v_max = self.Qnets.v_min, self.Qnets.v_max
            N = self.Qnets.n_atoms
            z_support = self.Qnets.z_support
            delta_z = self.Qnets.delta_z
            
            Tz = n_step_target(self.data['reward'], self.data['done'], 
                                z_support[None, :], self.gamma, self.data['steps'])     # [B, N]
            Tz = tf.clip_by_value(Tz, v_min, v_max)[:, None, :]                         # [B, 1, N]
            z_original = z_support[None, :, None]                                       # [1, N, 1]

            weight = tf.clip_by_value(1. - tf.abs(Tz - z_original) / delta_z, 0, 1)     # [B, N, N]
            dist_target = tf.reduce_sum(weight * self.Qnets.dist_next_target, axis=2)   # [B, N]
            dist_target = tf.stop_gradient(dist_target)

            kl_loss = tf.nn.softmax_cross_entropy_with_logits(labels=dist_target, logits=self.Qnets.logits)
            loss = tf.reduce_mean(kl_loss, name='loss')

        with tf.name_scope('priority'):
            priority = self._rescale(kl_loss)

        return priority, loss
        
    def _log_info(self):
        with tf.device('/CPU:0'):
            if self.log_tensorboard:
                with tf.name_scope('info'):
                    if self.buffer_type == 'proportional':
                        stats_summary('priority', self.priority, max=True, std=True)
                    stats_summary('Q', self.Qnets.Q, max=True)
                    stats_summary('prob', self.Qnets.dist, max=True)
                    tf.summary.scalar('loss_', self.loss)