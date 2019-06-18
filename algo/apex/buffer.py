import numpy as np

from replay.utils import init_buffer, add_buffer, copy_buffer
from utility.debug_tools import assert_colorize
from env.utils import encode_obs


class LocalBuffer(dict):
    def __init__(self, args, obs_space):
        self.capacity = args['local_capacity']
        self.n_steps = args['n_steps']
        self.gamma = args['gamma']

        # The following two fake data members are only used to complete the data pipeline
        # self.fake_ratio = np.zeros(self.valid_size)
        # self.fake_ids = np.zeros(self.valid_size, dtype=np.int32)

        init_buffer(self, self.capacity, obs_space, True)
        self['q'] = np.zeros((self.capacity+self.n_steps, 1))

        self.idx = 0

        self.idxes = np.arange(0, self.capacity)
        self.next_idxes = np.arange(self.n_steps, self.capacity + self.n_steps)
    
    def encode_recent_obs(self, obs):
        # to avoid complicating code, we expect tb_capacity >= frame_history_len
        assert_colorize(obs.shape == (84, 84, 1), f'Error shape: {obs.shape}')
        self['obs'][self.idx] = obs
        obs = encode_obs(self.idx, self['obs'], 
                        self['done'], 4, 
                        False, self.capacity)

        assert_colorize(obs.shape == (84, 84, 4), f'Error shape: {obs.shape}')

        return obs

    def __call__(self):
        """ This function is actually not used, but is required since we use dataset instead of placeholders """
        while True:
            obs = encode_obs(0, self['obs'], self['done'], 4, False, self.capacity)[None]
            # we take obs as next_obs for simplicity, since this function will not be invoked in practice anyway
            yield (obs, self['action'][0, None], self['reward'][0, None],
                                obs, self['done'][0, None], self['steps'][0, None])

    def reset(self):
        self.idx = 0
        
    def add(self, obs, action, reward, done, q):
        """ Add experience to local buffer, return True if local buffer is full, otherwise false """
        add_buffer(self, self.idx, obs, action, reward, 
                    done, self.n_steps, self.gamma)
        self['q'][self.idx] = q
        
        self.idx = self.idx + 1

    def compute_priority(self, gamma, epsilon, alpha):
        q = self['q'][self.idxes]
        target_q = self['reward'][self.idxes] + ((1 - self['done'][self.idxes]) 
                                                * gamma**self['steps'][self.idxes] 
                                                * self['q'][self.next_idxes])
        priority = (abs(q-target_q) + epsilon)**alpha

        return priority
