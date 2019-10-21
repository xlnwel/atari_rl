"""
Code for training single agent in atari
"""
import os
import time
import numpy as np
import tensorflow as tf

from utility.utils import pwc, set_global_seed
from utility.tf_utils import get_sess_config


def train(agent):
    if os.path.isdir(agent.model_file):
        pwc('Global step is loaded')
        step = agent.sess.run(agent.stats[0]['counter'])
    else:
        step = 0

    env = agent.env
    pwc(f'Initialize replay buffer')
    while not agent.good_to_learn:
        done = False
        obs = env.reset()
        while not done:
            action = env.random_action()
            next_obs, reward, done, _ = env.step(action)
            agent.add_data(obs, action, reward, done)
            obs = next_obs

    max_steps = float(agent.args['max_steps'])
    score_best = -float('inf')
    score_best_mean = -float('inf')

    itr = 0
    obs = agent.env.reset()
    start_time = time.time()
    while step <= max_steps:
        scores, epslens, step, itr = agent.train(step, itr)

        eval_scores, eval_epslens = agent.eval()

        # bookkeeping
        score = np.max(eval_scores)
        score_mean = np.mean(eval_scores)
        score_std = np.std(eval_scores)
        score_best = max(score_best, score)
        epslen_mean = np.mean(eval_epslens)
        epslen_std = np.mean(eval_epslens)

        agent.record_stats(global_step=step, score=score, score_mean=score_mean, 
                            score_std=score_std, score_best=score_best,
                            epslen_mean=epslen_mean, epslen_std=epslen_std)

        log_info = {
            'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
            'Timestep': f'{(step//1000):,}k',
            'Episode': itr,
            'TimeElapsed': f'{time.time() - start_time:.2f}s',
            'Score': score,
            'ScoreMean': score_mean,
            'ScoreStd': score_std,
            'ScoreBest': score_best,
            'ScoreBestMean': score_best_mean,
            'EpsLenMean': epslen_mean,
            'EpsLenStd': epslen_std,
            'LearningRate': agent.lr_schedule.value(step),
        }
        if hasattr(agent, 'exploration_schedule'):
            log_info['Exploration'] = agent.exploration_schedule.value(step)

        [agent.log_tabular(k, v) for k, v in log_info.items()]
        agent.dump_tabular()

        if score_best_mean > score_mean:
            score_best_mean = score_mean
            agent.save()

def get_agent(agent_args):
    algo = agent_args['algorithm']
    if algo == 'rainbow-iqn':
        from algo.rainbow_iqn.agent import Agent
    elif algo == 'rainbow':
        from algo.rainbow.agent import Agent
    elif algo == 'duel':
        from algo.ddqn.agent import Agent
    else:
        raise NotImplementedError

    return Agent
    
def main(env_args, agent_args, buffer_args, render=False, restore=False):
    Agent = get_agent(agent_args)
    set_global_seed()

    agent_args['env_stats']['times'] = 1
    sess_config = get_sess_config(2)
    agent = Agent('Agent', agent_args, env_args, buffer_args, 
                  save=True, log=True, log_tensorboard=True, 
                  log_stats=True, log_params=False)
    
    if restore:
        agent.restore()
    
    train(agent)
