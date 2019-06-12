from __future__ import absolute_import, division, print_function, unicode_literals  # provide backward compatibility

import time
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from ray.experimental.tf_utils import TensorFlowVariables

from utility.logger import Logger
from utility.debug_tools import assert_colorize
from basic_model.model import Model
from env.gym_env import GymEnv, GymEnvVec
from algo.apex.buffer import LocalBuffer
from replay.proportional_replay import ProportionalPrioritizedReplay
from replay.uniform_replay import UniformReplay
# from algo.rainbow_iqn.replay import ReplayBuffer as UniformReplay


class OffPolicyOperation(Model, ABC):
    """ Abstract base class for off-policy algorithms.
    Generally speaking, inherited class only need to define _build_graph
    and leave all interface as it-is.
    """
    """ Interface """
    def __init__(self, 
                 name, 
                 args, 
                 env_args, 
                 buffer_args, 
                 sess_config=None, 
                 save=True, 
                 log_tensorboard=True, 
                 log_params=False, 
                 log_stats=True, 
                 device=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99
        self.update_freq = args['update_freq']
        self.target_update_freq = args['target_update_freq']
        self.prefetches = args['prefetches'] if 'prefetches' in args else 0
        self.update_step = 0

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
        self.buffer_type = buffer_args['type']
        if self.buffer_type == 'proportional':
            self.buffer = ProportionalPrioritizedReplay(buffer_args, self.env.obs_space)
        elif self.buffer_type == 'uniform':
            self.buffer = UniformReplay(buffer_args, self.env.obs_space)
        elif self.buffer_type == 'local':
            self.buffer = LocalBuffer(buffer_args, self.env.obs_space)

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

        self._initialize_target_net()

        with self.graph.as_default():
            self.variables = TensorFlowVariables(self.loss, self.sess)
        
    @property
    def max_path_length(self):
        return self.env.max_episode_steps
    
    def atari_learn(self, t):
        if t % self.update_freq == 0:
            self._learn()
        else:
            pass # do nothing

    def act(self, obs, return_q=False):
        obs = self.buffer.encode_recent_obs(obs)
        obs = obs.reshape((-1, *self.obs_space))
        if return_q:
            action, q = self.sess.run([self.action, self.critic.Q_with_action], 
                                        feed_dict={self.data['obs']: obs})
            return np.squeeze(action), q
        else:
            action = self.sess.run(self.action, feed_dict={self.data['obs']: obs})
        
        return np.squeeze(action)

    def add_data(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def background_learning(self):
        from utility.debug_tools import timeit
        while not self.buffer.good_to_learn:
            time.sleep(1)
        print('Start Learning...')
        
        i = 0
        lt = []
        while True:
            i += 1
            duration, _ = timeit(self._learn)
            lt.append(duration)
            if i % 1000 == 0:
                print(f'{self.model_name}:\tTakes {np.sum(lt):3.2f}s to learn 1000 times')
                i = 0
                lt = []

    def _learn(self):
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
                # self.save()
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
    
    """ Implementation """
    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError

    def _prepare_data(self, buffer):
        if self.buffer_type == 'proportional':
            return self._prepare_data_per(buffer)
        elif self.buffer_type == 'uniform':
            return self._prepare_data_uniform(buffer)

    def _prepare_data_uniform(self, buffer):
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
            ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
            if self.prefetches > 0:
                ds = ds.prefetch(self.prefetches)
            iterator = ds.make_one_shot_iterator()
            samples = iterator.get_next(name='samples')

        obs, action, reward, next_obs, done, steps = samples
        
        data = {}
        data['obs'] = obs
        data['action'] = action
        data['reward'] = reward
        data['next_obs'] = next_obs
        data['done'] = done
        data['steps'] = steps

        return data

    def _prepare_data_per(self, buffer):
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
            ds = tf.data.Dataset.from_generator(buffer, sample_types, sample_shapes)
            if self.prefetches > 0:
                ds = ds.prefetch(self.prefetches)
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

    def _compute_priority(self, priority):
        with tf.name_scope('priority'):
            priority += self.prio_epsilon
            priority **= self.prio_alpha
        
        return priority

    def _initialize_target_net(self):
        raise NotImplementedError
    
    def _update_target_net(self):
        raise NotImplementedError

