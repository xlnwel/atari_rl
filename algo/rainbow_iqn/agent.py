import time
from collections import deque
import numpy as np
import tensorflow as tf
# from ray.experimental.tf_utils import TensorFlowVariables

from basic_model.model import Model
from algo.rainbow_iqn.networks import Networks
from utility.losses import huber_loss
from utility.debug_tools import timeit
from utility.utils import pwc, timeformat
from utility.schedule import PiecewiseSchedule
from utility.tf_utils import n_step_target, stats_summary
from env.gym_env import GymEnv, GymEnvVec
# from algo.apex.buffer import LocalBuffer
from replay.basic_replay import Replay
from replay.proportional_replay import ProportionalPrioritizedReplay
from replay.uniform_replay import UniformReplay


class Agent(Model):
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=False, 
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99
        self.update_freq = args['update_freq']
        self.critic_loss_type = args['loss_type']
        self.target_update_freq = args['target_update_freq']
        self.update_step = 0
        self.algo = args['algorithm']

        # environment info
        self.env = (GymEnvVec(env_args) if 'n_envs' in env_args and env_args['n_envs'] > 1
                    else GymEnv(env_args))
        h, w, c = self.env.obs_space
        self.obs_space = (h, w, args['frame_history_len'] * c)
        self.n_actions = self.env.action_dim
        
        # replay buffer
        buffer_args['n_steps'] = args['n_steps']
        buffer_args['gamma'] = args['gamma']
        buffer_args['batch_size'] = args['batch_size']
        buffer_args['frame_history_len'] = args['frame_history_len']
        self.buffer_type = buffer_args['type']
        if self.buffer_type == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.env.obs_space)
        elif self.buffer_type == 'uniform':
            self.buffer = UniformReplay(buffer_args, self.env.obs_space)
        # elif self.buffer_type == 'local':
        #     buffer_args['local_capacity'] = self.env.max_episode_steps
        #     self.buffer = LocalBuffer(buffer_args, self.env.obs_space)
        else:
            raise NotImplementedError

        # arguments for prioritized replay
        self.prio_alpha = float(buffer_args['alpha'])
        self.prio_epsilon = float(buffer_args['epsilon'])

        super().__init__(name, args, 
                         sess_config=sess_config, 
                         save=save, 
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

        # learing rate schedule
        n_iterations = float(self.args['max_steps']) / 4.
        lr = self.args['Qnets']['learning_rate']
        self.lr_schedule = PiecewiseSchedule([(0, lr), (n_iterations / 10, lr), (n_iterations / 2,  5e-5)],
                                             outside_value=5e-5)
        if not self.Qnets.use_noisy:
            # epsilon greedy schedulear if Q network does not use noisy layers
            self.exploration_schedule = PiecewiseSchedule([(0, 1.0), (1e6, 0.1), (n_iterations / 2, 0.01)], 
                                                            outside_value=0.01)

        self._initialize_target_net()

        # with self.graph.as_default():
        #     self.variables = TensorFlowVariables(self.loss, self.sess)

    @property
    def max_path_length(self):
        return self.env.max_episode_steps

    def train(self, render, log_steps):
        max_steps = float(self.args['max_steps'])
        score_best = -float('inf')

        itrtimes = deque(maxlen=1000)
        
        obs = self.env.reset()
        t = 0
        while t <= max_steps:
            duration, obs = timeit(self.run, (obs, t, log_steps, False))
            t += log_steps

            # bookkeeping
            itrtimes.append(duration)
            episode_scores = self.env.get_episode_rewards()
            episode_lengths = self.env.get_episode_lengths()
            if not episode_scores:
                continue

            score = episode_scores[-1]
            score_mean = np.mean(episode_scores[-100:])
            score_std = np.std(episode_scores[-100:])
            score_best = max(score_best, np.max(episode_scores))
            epslen_mean = np.mean(episode_lengths[-100:])
            epslen_std = np.mean(episode_lengths[-100:])

            if hasattr(self, 'stats'):
                self.record_stats(global_step=t, score=score, score_mean=score_mean, score_std=score_std, score_best=score_best,
                                epslen_mean=epslen_mean, epslen_std=epslen_std)

            log_info = {
                'ModelName': f'{self.args["algorithm"]}-{self.model_name}',
                'Timestep': f'{(t//1000):3d}k',
                'Iteration': len(episode_scores),
                'IterationTime': timeformat(np.mean(itrtimes)) + 's',
                'Score': score,
                'ScoreMean': score_mean,
                'ScoreStd': score_mean,
                'ScoreBest': score_best,
                'EpsLenMean': epslen_mean,
                'EpsLenStd': epslen_std,
                'LearningRate': self.lr_schedule.value(t),
            }
            if hasattr(self, 'exploration_schedule'):
                log_info['Exploration'] = self.exploration_schedule.value(t)

            [self.log_tabular(k, v) for k, v in log_info.items()]
            self.dump_tabular()

    def learn(self, t, lr):
        if not self.Qnets.args['schedule_lr']:
            lr = None
        # update only update_freq steps
        if t % self.update_freq == 0:
            self._learn(lr)
        else:
            return

    def run(self, obs, start, steps, render):
        for i in range(steps):
            if render:
                self.env.render()
            random_act = (not self.buffer.good_to_learn if self.Qnets.use_noisy 
                          else not self.buffer.good_to_learn 
                            or np.random.sample() < self.exploration_schedule.value(start+i))
            action = self.act(obs, random_act=random_act)

            next_obs, reward, done, _ = self.env.step(action)
            self.add_data(obs, action, reward, done)
            if self.buffer.good_to_learn:
                self.learn(i, self.lr_schedule.value(start+i))

            obs = self.env.reset() if done else next_obs

        if hasattr(self, 'saver'):
            self.save()
            
        return obs

    def act(self, obs, random_act=False, return_q=False):
        obs = self.buffer.encode_recent_obs(obs)
        obs = obs.reshape((-1, *self.obs_space))
        feed_dict = {self.data['obs']: obs}
        if random_act and return_q:
            action = self.env.random_action()
            q = self.sess.run(self.Qnets.Q, feed_dict=feed_dict)
            return action, np.squeeze(q)
        elif not random_act and return_q:
            action, q = self.sess.run([self.action, self.Qnets.Q], 
                                        feed_dict=feed_dict)
            return np.squeeze(action), np.squeeze(q)
        elif random_act:
            action = self.env.random_action()
            return action
        else:
            action = self.sess.run(self.action, feed_dict=feed_dict)
            return np.squeeze(action)

    def add_data(self, obs, action, reward, done):
        self.buffer.add(obs, action, reward, done)

    """ Implementation """
    def _build_graph(self):
        self.data = self._prepare_data()

        self.Qnets = self._create_nets()
        self.action = self.Qnets.best_action

        self.priority, self.loss = self._loss()

        _, self.learning_rate, self.opt_step, _, self.opt_op = self.Qnets._optimization_op(
                                                                            self.loss, 
                                                                            tvars=self.Qnets.main_variables,
                                                                            opt_step=True, schedule_lr=self.Qnets.args['schedule_lr'])

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_loss()

    def _prepare_data(self):
        if self.buffer_type == 'uniform' or self.buffer_type == 'local':
            return self._prepare_data_uniform()
        elif self.buffer_type == 'proportional':
            return self._prepare_data_per()
        else:
            raise NotImplementedError

    def _prepare_data_uniform(self):
        with tf.name_scope('data'):
            sample_types = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
            sample_shapes = (
                (None, *self.obs_space),
                (None, ),
                (None, 1),
                (None, *self.obs_space),
                (None, 1),
                (None, 1)
            )
            ds = tf.data.Dataset.from_generator(self.buffer, sample_types, sample_shapes)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = ds.make_one_shot_iterator()
            samples = iterator.get_next(name='samples')

        obs, action, reward, next_obs, done, steps = samples

        obs /= 255.
        next_obs /= 255.
        
        data = {}
        data['obs'] = obs
        data['action'] = action
        data['reward'] = reward
        data['next_obs'] = next_obs
        data['done'] = done
        data['steps'] = steps

        return data

    def _prepare_data_per(self):
        with tf.name_scope('data'):
            exp_type = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
            sample_types = (tf.float32, tf.int32, exp_type)
            sample_shapes =((None), (None), (
                (None, *self.obs_space),
                (None, ),
                (None, 1),
                (None, *self.obs_space),
                (None, 1),
                (None, 1)
            ))
            ds = tf.data.Dataset.from_generator(self.buffer, sample_types, sample_shapes)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            iterator = ds.make_one_shot_iterator()
            samples = iterator.get_next(name='samples')

        # prepare data
        IS_ratio, saved_mem_idxs, (obs, action, reward, next_obs, done, steps) = samples

        obs /= 255.
        next_obs /= 255.
        
        data = {}
        data['IS_ratio'] = IS_ratio                 # Importance sampling ratio for PER
        # saved indexes used to index the experience in the buffer when updating priorities
        data['saved_mem_idxs'] = saved_mem_idxs     
        data['obs'] = obs
        data['action'] = action
        data['reward'] = reward
        data['next_obs'] = next_obs
        data['done'] = done
        data['steps'] = steps

        return data

    def _create_nets(self):
        scope_prefix = self.name
        self.args['Qnets']['batch_size'] = self.args['batch_size']
        self.args['Qnets']['algo'] = self.algo
        Qnets = Networks('Nets', 
                        self.args['Qnets'], 
                        self.graph, 
                        self.data, 
                        self.n_actions,    # n_actions
                        scope_prefix=scope_prefix,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)

        return Qnets

    def _loss(self):
        if self.algo == 'iqn':
            return self._iqn_loss()
        elif self.algo == 'c51' or self.algo == 'rainbow':
            return self._c51_loss()
        else:
            return self._q_loss()

    def _iqn_loss(self):
        def tiled_n_step_target():
            n_quantiles = self.args['Qnets']['N_prime']
            
            reward_tiled = tf.reshape(tf.tile(self.data['reward'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            done_tiled = tf.reshape(tf.tile(self.data['done'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            steps_tiled = tf.reshape(tf.tile(self.data['steps'], [n_quantiles, 1]),
                                      [n_quantiles, -1, 1])
            return n_step_target(reward_tiled, done_tiled, 
                                 self.Qnets.quantile_values_next_target,
                                 self.gamma, steps_tiled)

        def quantile_regression_loss(u):
            abs_part = tf.abs(self.Qnets.quantiles - tf.where(u < 0, tf.ones_like(u), tf.zeros_like(u)))
            huber = huber_loss(u, delta=self.args['Qnets']['delta'])
            
            qr_loss = tf.reduce_sum(tf.reduce_mean(abs_part * huber, axis=2), axis=1)   # [B]
            loss = tf.reduce_mean(self.data['IS_ratio'] * qr_loss)

            return loss

        _, priority = self._compute_priority(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])

        with tf.name_scope('loss'):
            quantile_values_target = tiled_n_step_target()
            quantile_values_target = tf.transpose(quantile_values_target, [1, 2, 0])    # [B, 1, N']
            quantile_values = tf.transpose(self.Qnets.quantile_values, [1, 0, 2])       # [B, N, 1]
            quantile_error = tf.abs(quantile_values - quantile_values_target)

            loss = quantile_regression_loss(quantile_error)

        return priority, loss

    def _c51_loss(self):
        with tf.name_scope('loss'):
            # Eq.7 in c51 paper
            v_min, v_max = self.Qnets.v_min, self.Qnets.v_max
            N = self.Qnets.n_atoms
            z_support = self.Qnets.z_support
            delta_z = self.Qnets.delta_z
            
            z_target = n_step_target(self.data['reward'], self.data['done'], 
                                    z_support, self.gamma, self.data['steps'])           # [B, N]
            z_target = tf.clip_by_value(z_target, v_min, v_max)[:, None, :]              # [B, 1, N]
            z_original = z_support[None, :, None]                                        # [1, N, 1]

            weight = tf.clip_by_value(1.-tf.abs(z_target - z_original) / delta_z, 0, 1)  # [B, N, N]
            dist_target = tf.reduce_sum(weight * self.Qnets.dist_next_target, axis=2)    # [B, N]
            dist_target = tf.stop_gradient(dist_target)

            kl_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=dist_target, logits=self.Qnets.logits)
            loss = tf.reduce_mean(kl_loss)

        with tf.name_scope('priority'):
            priority = self._rescale(kl_loss)

        return priority, loss
        
    def _q_loss(self):
        Q_error, priority = self._compute_priority(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])

        with tf.name_scope('loss'):
            loss_func = huber_loss if self.critic_loss_type == 'huber' else tf.square
            if self.buffer_type == 'proportional':
                loss = tf.reduce_mean(self.data['IS_ratio'][:, None] * loss_func(Q_error), name='loss')
            else:
                loss = tf.reduce_mean(loss_func(Q_error), name='loss')

        return priority, loss

    def _compute_priority(self, reward, done, Q_next_target, gamma, steps):
        with tf.name_scope('priority'):
            Q_target = n_step_target(reward, done, Q_next_target, gamma, steps)
            Q_error = self.Qnets.Q - Q_target
            priority = tf.abs(Q_error)
            priority = self._rescale(priority)
        
        return Q_error, priority

    def _rescale(self, priority):
        priority = tf.squeeze(priority)
        priority = (priority + self.prio_epsilon)**self.prio_alpha
        
        return priority

    def _target_net_ops(self):
        with tf.name_scope('target_net_op'):
            target_main_var_pairs = list(zip(self.Qnets.target_variables, self.Qnets.main_variables))
            init_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='init_target_op'), target_main_var_pairs))
            update_target_op = list(map(lambda v: tf.assign(v[0], v[1], name='update_target_op'), target_main_var_pairs))

        return init_target_op, update_target_op

    def _initialize_target_net(self):
        self.sess.run(self.init_target_op)

    def _update_target_net(self):
        if self.update_step % self.target_update_freq == 0:
            pwc('Target net synchronized.', color='green')
            self.sess.run(self.update_target_op)

    def _log_loss(self):
        if self.log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('loss_', self.loss)
            
            with tf.name_scope('networks'):
                stats_summary('Q', self.Qnets.Q)

    def _learn(self, lr=None):
        if lr:
            feed_dict = {self.learning_rate: lr}
            if self.log_tensorboard:
                if self.buffer_type == 'proportional':
                    priority, saved_mem_idxs, _, summary = self.sess.run([self.priority, 
                                                                        self.data['saved_mem_idxs'], 
                                                                        self.opt_op, 
                                                                        self.graph_summary], 
                                                                        feed_dict=feed_dict)
                else:
                    _, summary = self.sess.run([self.opt_op, self.graph_summary], feed_dict=feed_dict)

                if self.update_step % 100 == 0:
                    self.writer.add_summary(summary, self.update_step)
            else:
                if self.buffer_type == 'proportional':
                    priority, saved_mem_idxs, _ = self.sess.run([self.priority, 
                                                                self.data['saved_mem_idxs'], 
                                                                self.opt_op], feed_dict=feed_dict)
                else:
                    _ = self.sess.run([self.opt_op], feed_dict=feed_dict)
        else:
            if self.log_tensorboard:
                if self.buffer_type == 'proportional':
                    priority, saved_mem_idxs, _, summary = self.sess.run([self.priority, 
                                                                        self.data['saved_mem_idxs'], 
                                                                        self.opt_op, 
                                                                        self.graph_summary])
                else:
                    _, summary = self.sess.run([self.opt_op, self.graph_summary])

                if self.update_step % 100 == 0:
                    self.writer.add_summary(summary, self.update_step)
            else:
                if self.buffer_type == 'proportional':
                    priority, saved_mem_idxs, _ = self.sess.run([self.priority, 
                                                                self.data['saved_mem_idxs'], 
                                                                self.opt_op])
                else:
                    _ = self.sess.run([self.opt_op])
        # update the target networks
        self._update_target_net()
        self.update_step += 1
        if self.buffer_type == 'proportional':
            self.buffer.update_priorities(priority, saved_mem_idxs)
