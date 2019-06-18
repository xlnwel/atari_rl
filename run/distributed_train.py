import os

import time
import argparse
from multiprocessing import cpu_count
import numpy as np
import tensorflow as tf
import ray

from utility.yaml_op import load_args
from replay.proportional_replay import ProportionalPrioritizedReplay
from algo.apex.worker import get_worker
from algo.apex.learner import get_learner


def main(env_args, agent_args, buffer_args, render=False):
    from algo.rainbow_iqn.agent import Agent
    
    if 'n_workers' not in agent_args:
        # 1 cpu for each worker and 2 cpus for the learner so that network update happens at a background thread
        n_workers = cpu_count() - 2
        agent_args['n_workers'] = n_workers
        agent_args['env_stats']['times'] = n_workers
    else:
        n_workers = agent_args['n_workers']
        agent_args['env_stats']['times'] = n_workers

    ray.init(num_cpus=n_workers + 2, num_gpus=1)

    agent_name = 'Agent'
    learner = get_learner(Agent, agent_name, agent_args, env_args, buffer_args, device='/GPU:0')

    workers = []
    buffer_args['type'] = 'local'
    for worker_no in range(n_workers):
        max_episodes = 1    # np.random.randint(1, 10)
        
        agent_args['Qnets']['noisy_sigma'] = np.random.randint(3, 7) * .1
        
        env_args['seed'] = worker_no * 10
        worker = get_worker(Agent, agent_name, worker_no, agent_args, env_args, buffer_args, 
                            max_episodes, device='/CPU:{}'.format(worker_no + 1))
        workers.append(worker)

    pids = [worker.sample_data.remote(learner) for worker in workers]

    ray.get(pids)
