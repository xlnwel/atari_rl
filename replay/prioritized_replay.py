import numpy as np

from utility.decorators import override
from utility.debug_tools import assert_colorize
from replay.basic_replay import Replay
from replay.utils import add_buffer, copy_buffer


class PrioritizedReplay(Replay):
    """ Interface """
    def __init__(self, args, obs_space, action_dim):
        super().__init__(args, obs_space, action_dim)
        # self.memory                           # mem_idx    -->     exp
        self.data_structure = None              # prio_id   -->     priority, mem_idx

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4
        self.beta_grad = (1 - self.beta) / float(args['beta_steps']) * 100

        self.top_priority = 2.

        self.sample_i = 0   # count how many times self.sample is called

    @override(Replay)
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:        
            samples = self._sample()
            self.sample_i += 1
            if self.sample_i % 100 == 0:
                self._update_beta()

        return samples

    @override(Replay)
    def add(self, obs, action, reward, next_obs, done):
        self.tb['priority'][self.tb_idx] = self.top_priority
        super()._add(obs, action, reward, next_obs, done)

    def update_priorities(self, priorities, saved_mem_idxs):
        with self.locker:
            for priority, mem_idx in zip(priorities, saved_mem_idxs):
                self.data_structure.update(priority, mem_idx)

    """ Implementation """
    def _update_beta(self):
        self.beta = min(self.beta + self.beta_grad, 1)

    @override(Replay)
    def _merge(self, local_buffer, length, start=0):
        end_idx = self.mem_idx + length
        for prio_id, mem_idx in enumerate(range(self.mem_idx, end_idx)):
            self.data_structure.add(local_buffer['priority'][prio_id], mem_idx % self.capacity, self.is_full)
            
        super()._merge(local_buffer, length, start)
        
    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = np.power(probabilities * N, -self.beta)
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upward

        return IS_ratios