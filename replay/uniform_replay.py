import threading
import numpy as np

from utility.decorators import override
from replay.basic_replay import Replay
from replay.utils import add_buffer, copy_buffer


class UniformReplay(Replay):
    """ Interface """
    def __init__(self, args, obs_space, action_space):
        super(args, obs_space, action_space)

    @override(Replay)
    def add(self, obs, action, reward,  next_obs, done):
        super()._add(obs, action, reward, next_obs, done)

    """ Implementation """
    @override(Replay)
    def _sample(self):
        size = self.capacity if self.is_full else self.mem_idx
        indices = np.random.randint(0, size, self.batch_size)
        return [v[indices] for v in self.memory.values()]
