import numpy as np

from utility.decorators import override
from utility.debug_tools import assert_colorize
from utility.schedule import PiecewiseSchedule
from replay.basic_replay import Replay
from replay.ds.sum_tree import SumTree
from replay.utils import add_buffer, copy_buffer


class ProportionalPrioritizedReplay(Replay):
    """ Interface """
    def __init__(self, args, obs_space):
        super().__init__(args, obs_space)
        # self.memory                                   # mem_idx    -->     exp
        self.data_structure = SumTree(self.capacity)    # mem_idx    -->     priority

        # params for prioritized replay
        self.alpha = float(args['alpha']) if 'alpha' in args else .5
        self.beta = float(args['beta0']) if 'beta0' in args else .4
        self.beta_schedule = PiecewiseSchedule([(0, args['beta0']), (float(args['beta_steps']), 1.)], 
                                                outside_value=1.)
        self.epsilon = float(args['epsilon']) if 'epsilon' in args else 1e-4

        self.top_priority = 2.
        self.to_update_priority = args['to_update_priority'] if 'to_update_priority' in args else True

        self.sample_i = 0   # count how many times self.sample is called

    @override(Replay)
    def sample(self):
        assert_colorize(self.good_to_learn, 'There are not sufficient transitions to start learning --- '
                                            f'transitions in buffer: {len(self)}\t'
                                            f'minimum required size: {self.min_size}')
        with self.locker:        
            samples = self._sample()
            self.sample_i += 1
            self._update_beta()

        return samples

    @override(Replay)
    def add(self, state, action, reward, done):
        if self.n_steps > 1:
            self.tb['priority'][self.tb_idx] = self.top_priority
        else:
            self.memory['priority'][self.mem_idx] = self.top_priority
            self.data_structure.update(self.top_priority, self.mem_idx)
        super()._add(state, action, reward, done)

    def update_priorities(self, priorities, saved_mem_idxs):
        with self.locker:
            if self.to_update_priority:
                self.top_priority = max(self.top_priority, np.max(priorities))
            for priority, mem_idx in zip(priorities, saved_mem_idxs):
                self.data_structure.update(priority, mem_idx)

    """ Implementation """
    def _update_beta(self):
        self.beta = self.beta_schedule.value(self.sample_i)

    @override(Replay)
    def _merge(self, local_buffer, length):
        end_idx = self.mem_idx + length
        assert np.all(local_buffer['priority'][: length])
        for idx, mem_idx in enumerate(range(self.mem_idx, end_idx)):
            self.data_structure.update(local_buffer['priority'][idx], mem_idx % self.capacity)
            
        super()._merge(local_buffer, length)
    
    @override(Replay)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, indexes = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                                        for i in range(self.batch_size)]))

        priorities = np.array(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        N = len(self)
        IS_ratios = self._compute_IS_ratios(N, probabilities)
        samples = self._get_samples(indexes)
        
        return IS_ratios, indexes, samples

    def _compute_IS_ratios(self, N, probabilities):
        IS_ratios = (probabilities * N) ** -self.beta
        IS_ratios /= np.max(IS_ratios)  # normalize ratios to avoid scaling the update upward

        return IS_ratios