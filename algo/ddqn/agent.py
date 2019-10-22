import numpy as np
import tensorflow as tf
# from ray.experimental.tf_utils import TensorFlowVariables

from basic_model.model import Model
from algo.ddqn.network import DuelDQN
from utility.losses import huber_loss
from utility.debug_tools import timeit
from utility.utils import pwc
from utility.schedule import PiecewiseSchedule
from utility.tf_utils import n_step_target, stats_summary
from env.gym_env import create_env
# from algo.apex.buffer import LocalBuffer
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
                 log=False,
                 log_tensorboard=False, 
                 log_params=False, 
                 log_stats=False, 
                 device=None,
                 reuse=None):
        # hyperparameters
        self.gamma = args['gamma'] if 'gamma' in args else .99
        self.update_freq = args['update_freq']
        self.target_update_freq = args['target_update_freq']
        self.update_step = 0

        self.train_steps = args['train_steps']
        self.eval_steps = args['eval_steps']

        # environment info
        env_args['log_video'] = True
        env_args['episode_life'] = False
        self.eval_env = create_env(env_args)
        video_path = env_args['video_path']
        env_args['seed'] += 10
        env_args['log_video'] = False
        env_args['episode_life'] = True
        self.env = create_env(env_args)
        
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
                         log=log,
                         log_tensorboard=log_tensorboard, 
                         log_params=log_params, 
                         log_stats=log_stats, 
                         device=device)

        # learing rate schedule
        decay_duration = float(self.args['max_steps'])
        lr = float(self.args['Qnets']['learning_rate'])
        end_lr = float(self.args['Qnets']['end_lr'])
        self.lr_schedule = PiecewiseSchedule([(0, lr), (decay_duration / 8, lr), (decay_duration / 4,  end_lr)],
                                             outside_value=end_lr)
        if not self.Qnets.use_noisy:
            # epsilon greedy schedulear if Q network does not use noisy layers
            self.exploration_schedule = PiecewiseSchedule([(0, 1.0), (1e6, 0.1), (decay_duration / 2, 0.01)], 
                                                            outside_value=0.01)

        self._initialize_target_net()

    @property
    def max_path_length(self):
        return self.env.max_episode_steps

    @property
    def good_to_learn(self):
        return self.buffer.good_to_learn

    def learn(self, step, lr):
        if not self.Qnets.args['schedule_lr']:
            lr = None
        # update only update_freq steps
        if step % self.update_freq == 0:
            self._learn(lr)
        else:
            return

    def run_episode(self, env, fn, step):
        done = False
        obs = env.reset()
        while not done:
            obs_stack = env.get_obs_stack()
            action = self.act(obs_stack)
            next_obs, reward, done, _ = env.step(action)
            fn(obs, action, reward, done, step)
            obs = next_obs
            step += 1
        
        return env.get_episode_score(), env.get_episode_length(), step
                

    def eval(self):
        def fn(obs, action, reward, done, step):
            pass
        pwc('Start Evaluating')
        step = 0
        scores = []
        epslens = []
        while step < self.eval_steps:
            score, epslen, step = self.run_episode(self.eval_env, fn, step)
            scores.append(score)
            epslens.append(epslen)

        return scores, epslens

    def train(self, step, itr):
        pwc('Start Training')
        def fn(obs, action, reward, done, step):
            self.add_data(obs, action, reward, done)
            if self.buffer.good_to_learn:
                self.learn(step, hasattr(self, 'lr_schedule') and self.lr_schedule.value(step))
            
        t = 0
        scores = []
        epslens = []
        while t < self.train_steps:
            score, epslen, new_step = self.run_episode(self.env, fn, step)
            scores.append(score)
            epslens.append(epslen)
            itr += 1
            t += new_step - step
            step = new_step

        return scores, epslens, step, itr

    def act(self, obs, random_act=False, return_q=False):
        assert obs.shape == self.obs_space
        obs = obs[None]
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
                                                                            opt_step=True, 
                                                                            schedule_lr=self.Qnets.args['schedule_lr'])

        # target net operations
        self.init_target_op, self.update_target_op = self._target_net_ops()

        self._log_info()

    def _prepare_data(self):
        with tf.device('/CPU:0'):
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
        self.args['Qnets']['batch_size'] = self.args['batch_size']
        Qnets = DuelDQN('Nets', 
                        self.args['Qnets'], 
                        self.graph, 
                        self.data, 
                        self.n_actions,    # n_actions
                        scope_prefix=self.name,
                        log_tensorboard=self.log_tensorboard,
                        log_params=self.log_params)
        
        return Qnets

    def _loss(self):
        Q_error, priority = self._compute_priority(self.data['reward'], self.data['done'],
                                    self.Qnets.Q_next_target, self.gamma, self.data['steps'])

        with tf.name_scope('loss'):
            loss_func = huber_loss if self.args['loss_type'] == 'huber' else tf.square
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
            self.sess.run(self.update_target_op)

    def _log_info(self):
        with tf.device('/CPU:0'):
            if self.log_tensorboard:
                with tf.name_scope('info'):
                    if self.buffer_type == 'proportional':
                        stats_summary('priority', self.priority, max=True, std=True)
                    stats_summary('Q', self.Qnets.Q, max=True)
                    tf.summary.scalar('loss_', self.loss)

    def _learn(self, lr=None):
        if lr:
            feed_dict = {self.learning_rate: lr}
        else:
            feed_dict = None
        if self.log_tensorboard:
            if self.buffer_type == 'proportional':
                priority, saved_mem_idxs, _, self.update_step, summary = self.sess.run(
                                                                            [self.priority, 
                                                                            self.data['saved_mem_idxs'], 
                                                                            self.opt_op, 
                                                                            self.opt_step,
                                                                            self.graph_summary], 
                                                                            feed_dict=feed_dict)
            else:
                _, self.update_step, summary = self.sess.run([self.opt_op, self.opt_step, self.graph_summary], feed_dict=feed_dict)

            if self.update_step % 1000 == 0:
                self.writer.add_summary(summary, self.update_step)
        else:
            if self.buffer_type == 'proportional':
                priority, saved_mem_idxs, _, self.update_step = self.sess.run([self.priority, 
                                                                    self.data['saved_mem_idxs'], 
                                                                    self.opt_op, 
                                                                    self.opt_step], feed_dict=feed_dict)
            else:
                _, self.update_step = self.sess.run([self.opt_op, self.opt_step], feed_dict=feed_dict)
        
        # update the target networks
        self._update_target_net()
        if self.buffer_type == 'proportional':
            self.buffer.update_priorities(priority, saved_mem_idxs)
