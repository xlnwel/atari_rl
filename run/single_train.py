"""
Code for training single agent in atari
"""
import time
import threading
from collections import deque
import numpy as np

from utility import utils
from utility.debug_tools import timeit
from utility.schedule import PiecewiseSchedule


def train(agent, render, log_steps, print_terminal_info=True, background_learning=True):
    n_iterations = 2e8 / 4.
    exploration_schedule = PiecewiseSchedule([(0, 1.0), (1e6, 0.1), (n_iterations / 2, 0.01)], 
                                            outside_value=0.01)
    lr_schedule = PiecewiseSchedule([(0, 1e-4), (n_iterations / 10, 1e-4), (n_iterations / 2,  5e-5)],
                                    outside_value=5e-5)

    episode_lengths = deque(maxlen = 100)
    best_score = -float('inf')
    el = 0
    t = 0
    buffer_type = agent.buffer_type

    obs = agent.env.reset()
    while agent.env.get_total_steps() < 2e8:
        el += 1

        if buffer_type == 'uniform':
            idx = agent.buffer.store_frame(obs)
        if render:
            agent.env.render()
        action = agent.act(obs) if agent.buffer.good_to_learn else agent.env.random_action()

        next_obs, reward, done, _ = agent.env.step(action)
        
        if buffer_type == 'uniform':
            agent.buffer.store_effect(idx, action, reward, done)
        else:
            agent.add_data(obs, action, reward, next_obs, done)
        
        if not background_learning and agent.buffer.good_to_learn:
            agent.learn(t, lr_schedule.value(t))

        obs = agent.env.reset() if done else next_obs
        if done:
            episode_lengths.append(el)
            el = 0

        t += 1

        if t % log_steps == 0:
            episode_scores = agent.env.get_episode_rewards()
            eps_len = agent.env.get_length()
            score = episode_scores[-1]
            avg_score = np.mean(episode_scores[-100:])
            best_score = max(best_score, avg_score)
            eps_len = episode_lengths[-1]
            avg_eps_len = np.mean(episode_lengths)
            if hasattr(agent, 'stats'):
                agent.record_stats(t=t, score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

            log_info = {
                'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                'Timestep': f'{(t//1000):3d}k',
                'Iteration': len(episode_scores),
                'Score': score,
                'AvgScore': avg_score,
                'BestScore': best_score,
                'EpsLen': eps_len,
                'AvgEpsLen': avg_eps_len,
                'LearningRate': lr_schedule.value(t),
                'Exploration': exploration_schedule.value(t)
            }
            [agent.log_tabular(k, v) for k, v in log_info.items()]
            agent.dump_tabular(print_terminal_info=print_terminal_info)

def main(env_args, agent_args, buffer_args, render=False):
    # print terminal information if main is running in the main thread
    utils.set_global_seed()

    algorithm = agent_args['algorithm']
    if algorithm == 'rainbow-iqn':
        from algo.rainbow_iqn.agent import Agent
    else:
        raise NotImplementedError

    agent_args['env_stats']['times'] = 1
    agent = Agent('Agent', agent_args, env_args, buffer_args, 
                log_tensorboard=True, log_stats=True, log_params=False, device='/GPU:0')
    if agent_args['background_learning']:
        utils.pwc('Background Learning...')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    else:
        utils.pwc('Foreground Learning...')
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, log_steps=int(1e4), background_learning=agent_args['background_learning'])
