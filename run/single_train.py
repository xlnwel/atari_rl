"""
Code for training single agent in atari
"""
import time
import threading
from collections import deque
import numpy as np
import tensorflow as tf

from utility.utils import pwc, set_global_seed, timeformat
from utility.debug_tools import timeit
from utility.schedule import PiecewiseSchedule
from algo.rainbow_iqn.agent import Agent


def main(env_args, agent_args, buffer_args, render=False):
    set_global_seed()

    agent_args['env_stats']['times'] = 1
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                 inter_op_parallelism_threads=1,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    agent = Agent('Agent', agent_args, env_args, buffer_args, save=True,
                log_tensorboard=True, log_stats=True, log_params=False, device='/GPU:0')

    model = agent_args['model_name']
    
    agent.train(render, log_steps=int(1e4))
