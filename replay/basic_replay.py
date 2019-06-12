import threading
import numpy as np

from utility.debug_tools import assert_colorize
from replay.utils import init_buffer, add_buffer, copy_buffer


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

    def encode_recent_obs(self, obs):
        # to avoid complicating code, we expect tb_capacity >= frame_history_len
        assert_colorize(self.tb_capacity >= self.frame_history_len, 
                        'Ops: encode_recent_obs will not work correctly')
        assert_colorize(obs.shape == (84, 84, 1), f'Error shape: {obs.shape}')
        self.tb['obs'][self.tb_idx] = obs
        obs = self._encode_obs(self.tb_idx, self.tb['obs'], 
                                self.tb['done'], self.frame_history_len, 
                                self.tb_full, self.tb_capacity)

        assert_colorize(obs.shape == (84, 84, 4), f'Error shape: {obs.shape}')

        return obs

    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:
            samples = self._sample()

        return samples

    def merge(self, local_buffer, length, start=0):
        assert_colorize(length < self.capacity, 'Local buffer is too large')
        with self.locker:
            self._merge(local_buffer, length, start)

    def add(self, obs, action, reward, next_obs, done):
        # locker should be handled in implementation
        raise NotImplementedError

    """ Implementation """
    def _add(self, obs, action, reward, next_obs, done):
        """ add is only used for single agent, no multiple adds are expected to run at the same time
            but it may fight for resource with self.sample if background learning is enabled """
        add_buffer(self.tb, self.tb_idx, obs, action, reward, 
                    next_obs, done, self.n_steps, self.gamma)
        
        if not self.tb_full and self.tb_idx == self.tb_capacity - 1:
            self.tb_full = True
        self.tb_idx = (self.tb_idx + 1) % self.tb_capacity

        if done:
            # flush all elements in temporary buffer to memory if an episode is done
            self.merge(self.tb, self.tb_capacity if self.tb_full else self.tb_idx)
            self.tb_full = False
            self.tb_idx = 0
        elif self.tb_full:
            # add the oldest ready experience in temporary buffer to memory
            self.merge(self.tb, 1, self.tb_idx)

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

        obs = np.stack([self._encode_obs(idx, self.memory['obs'], self.memory['done'],
                        self.frame_history_len, self.is_full, self.capacity) for idx in indexes])
        next_obs = np.stack([self._encode_obs(idx, self.memory['next_obs'], self.memory['done'],
                              self.frame_history_len, self.is_full, self.capacity) for idx in indexes])

        return (
            obs,
            self.memory['action'][indexes],
            self.memory['reward'][indexes],
            next_obs,
            self.memory['done'][indexes],
            self.memory['steps'][indexes],
        )

    def _encode_obs(self, idx, obs, done, frame_history_len, is_full, capacity):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - frame_history_len
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and not is_full:
            start_idx = 0
        # we do not consider episodes whose length is less than n_step
        if not done[idx]:
            for i in range(start_idx, end_idx - 1):
                if done[i % capacity]:
                    start_idx = i + 1
        missing_context = frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = ([np.zeros_like(obs[0]) for _ in range(missing_context)]
                        + [obs[i] for i in range(start_idx, end_idx)])
            # for idx in range(start_idx, end_idx):
            #     frames.append(obs[idx])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            h, w = obs.shape[1], obs.shape[2]
            return obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(h, w, -1)
