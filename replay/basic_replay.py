import threading
import numpy as np

from utility.debug_tools import assert_colorize
from replay.utils import init_buffer, add_buffer, copy_buffer
from env.utils import encode_obs


class Replay:
    """ Interface """
    def __init__(self, args, obs_space):
        self.memory = {}

        # params for general replay buffer
        self.capacity = int(float(args['capacity']))
        self.min_size = int(float(args['min_size']))
        self.batch_size = args['batch_size']

        self.n_steps = args['n_steps']
        self.gamma = args['gamma']
        
        # argument for atari games
        # concatenate frame_history_len observations as input to the network
        self.frame_history_len = args['frame_history_len']

        self.is_full = False
        self.mem_idx = 0

        init_buffer(self.memory, self.capacity, obs_space, False)

        # Code for single agent
        self.tb_capacity = args['tb_capacity']
        self.tb_idx = 0
        self.tb_full = False
        self.tb = {}
        init_buffer(self.tb, self.tb_capacity, obs_space, True)
        
        # locker used to avoid conflict introduced by tf.data.Dataset and multi-agent
        self.locker = threading.Lock()

    @property
    def good_to_learn(self):
        return len(self) >= self.min_size

    def __len__(self):
        return self.capacity if self.is_full else self.mem_idx

    def __call__(self):
        while True:
            yield self.sample()

    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:
            samples = self._sample()

        return samples

    def merge(self, local_buffer, length, start=0):
        """ Merge a local buffer to the replay buffer, useful for distributed algorithms """
        assert_colorize(length < self.capacity, 'Local buffer is too large')
        with self.locker:
            self._merge(local_buffer, length, start)

    def add(self, obs, action, reward, done):
        """ Add a single transition to the replay buffer """
        # locker should be handled in implementation
        raise NotImplementedError

    """ Implementation """
    def _add(self, obs, action, reward, done):
        """ add is only used for single agent, no multiple adds are expected to run at the same time
            but it may fight for resource with self.sample if background learning is enabled """
        add_buffer(self.tb, self.tb_idx, obs, action, reward, 
                    done, self.n_steps, self.gamma)
        
        if not self.tb_full and self.tb_idx == self.tb_capacity - 1:
            self.tb_full = True
        self.tb_idx = (self.tb_idx + 1) % self.tb_capacity

        if done:
            # flush all elements in temporary buffer to memory if an episode is done
            self.merge(self.tb, self.tb_capacity if self.tb_full else self.tb_idx)
            self.tb_full = False
            self.tb_idx = 0
        elif self.tb_full:
            # add the ready experiences in temporary buffer to memory
            n_not_ready = max(self.frame_history_len, self.n_steps) - 1
            n_ready = self.tb_capacity - n_not_ready
            self.merge(self.tb, n_ready, self.tb_idx)
            assert self.tb_idx == 0
            copy_buffer(self.tb, 0, n_not_ready, self.tb, self.tb_capacity - n_not_ready, self.tb_capacity)
            self.tb_idx = n_not_ready
            self.tb_full = False

    def _sample(self):
        raise NotImplementedError

    def _merge(self, local_buffer, length, start=0):
        end_idx = self.mem_idx + length

        if end_idx > self.capacity:
            first_part = self.capacity - self.mem_idx
            second_part = length - first_part
            
            copy_buffer(self.memory, self.mem_idx, self.capacity, local_buffer, start, start + first_part)
            copy_buffer(self.memory, 0, second_part, local_buffer, start + first_part, start + length)
        else:
            copy_buffer(self.memory, self.mem_idx, end_idx, local_buffer, start, start + length)

        # memory is full, recycle buffer via FIFO
        if not self.is_full and end_idx >= self.capacity:
            print('Memory is fulll')
            self.is_full = True
        
        self.mem_idx = end_idx % self.capacity

    def _get_samples(self, indexes):
        indexes = list(indexes) # convert tuple to list

        obs = np.stack([encode_obs(idx, self.memory['obs'], self.memory['done'],
                        self.frame_history_len, self.is_full, self.capacity) for idx in indexes])
        # squeeze steps since it is of shape [None, 1]
        next_indexes = (indexes + np.squeeze(self.memory['steps'][indexes])) % self.capacity
        next_obs = np.stack([encode_obs(idx, self.memory['obs'], self.memory['done'],
                              self.frame_history_len, self.is_full, self.capacity) for idx in next_indexes])
        # use zero obs as terminal obs
        next_obs = np.where(self.memory['done'][indexes][..., None, None], 
                            np.zeros_like(obs), next_obs)

        return (
            obs,
            self.memory['action'][indexes],
            self.memory['reward'][indexes],
            next_obs,
            self.memory['done'][indexes],
            self.memory['steps'][indexes],
        )
