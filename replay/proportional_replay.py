import numpy as np
import ray

from utility.decorators import override
from replay.ds.sum_tree import SumTree
from replay.prioritized_replay import PrioritizedReplay


class ProportionalPrioritizedReplay(PrioritizedReplay):
    """ Interface """
    def __init__(self, args, obs_space):
        super().__init__(args, obs_space)
        self.data_structure = SumTree(self.capacity)                   # prio_id   -->     priority, exp_id

    """ Implementation """
    @override(PrioritizedReplay)
    def _sample(self):
        total_priorities = self.data_structure.total_priorities
        
        segment = total_priorities / self.batch_size

        priorities, indexes = list(zip(*[self.data_structure.find(np.random.uniform(i * segment, (i+1) * segment))
                                        for i in range(self.batch_size)]))

        priorities = np.squeeze(priorities)
        probabilities = priorities / total_priorities

        # compute importance sampling ratios
        N = len(self)
        IS_ratios = self._compute_IS_ratios(N, probabilities)
        samples = self._get_samples(indexes)
        
        return IS_ratios, indexes, samples
