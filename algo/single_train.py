"""
Code for training single agent in atari
"""
import os
import time
import threading
from collections import deque
import numpy as np
import tensorflow as tf

from utility.utils import pwc, set_global_seed, timeformat
from utility.debug_tools import timeit
from utility.schedule import PiecewiseSchedule
from algo.rainbow_iqn.agent import Agent


def train(agent, render):
    max_steps = float(agent.args['max_steps'])
    score_best = -float('inf')

    itrtimes = deque(maxlen=1000)

    if os.path.isdir(agent.model_file):
        t = agent.sess.run(agent.stats[0]['counter'])
    else:
        t = 0

    while t <= max_steps:
        duration, t = timeit(agent.run, (t, True, False))

        # bookkeeping
        itrtimes.append(duration)
        episode_scores = agent.env.get_episode_rewards()
        episode_lengths = agent.env.get_episode_lengths()

        if not episode_scores:
            continue

        score = episode_scores[-1]
        score_mean = np.mean(episode_scores[-100:])
        score_std = np.std(episode_scores[-100:])
        score_best = max(score_best, np.max(episode_scores))
        epslen_mean = np.mean(episode_lengths[-100:])
        epslen_std = np.mean(episode_lengths[-100:])

        if hasattr(agent, 'stats'):
            agent.record_stats(global_step=t, score=score, score_mean=score_mean, 
                                score_std=score_std, score_best=score_best,
                                epslen_mean=epslen_mean, epslen_std=epslen_std)

        if hasattr(agent, 'logger'):
            log_info = {
                'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                'Timestep': f'{(t//1000):,}k',
                'Iteration': len(episode_scores),
                'IterationTime': timeformat(np.mean(itrtimes)) + 's',
                'Score': score,
                'ScoreMean': score_mean,
                'ScoreStd': score_mean,
                'ScoreBest': score_best,
                'EpsLenMean': epslen_mean,
                'EpsLenStd': epslen_std,
                'LearningRate': agent.lr_schedule.value(t),
            }
            if hasattr(agent, 'exploration_schedule'):
                log_info['Exploration'] = agent.exploration_schedule.value(t)

            [agent.log_tabular(k, v) for k, v in log_info.items()]
            agent.dump_tabular()

        if hasattr(agent, 'saver'):
            agent.save()

def main(env_args, agent_args, buffer_args, render=False, restore=False):
    set_global_seed()

    agent_args['env_stats']['times'] = 1
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                 inter_op_parallelism_threads=1,
                                 allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    agent = Agent('Agent', agent_args, env_args, buffer_args, 
                  save=True, log=True, log_tensorboard=False, 
                  log_stats=True, log_params=False, 
                  device='/GPU:0')

    if restore:
        agent.restore()
    
    train(agent, render)
