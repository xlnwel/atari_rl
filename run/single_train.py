"""
Code for training single agent in atari
"""
import time
import threading
from collections import deque
import numpy as np

from utility import utils
from utility.debug_tools import timeit

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


def train(agent, render, log_steps, print_terminal_info=True, background_learning=True):
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (2e6 / 2, 0.01),
        ], outside_value=0.01
    )
    obs = agent.env.reset()
    t = 0
    acttimes = deque(maxlen=log_steps)
    envtimes = deque(maxlen=log_steps)
    addtimes = deque(maxlen=log_steps)
    learntimes = deque(maxlen=log_steps)
    episode_lengths = deque(maxlen = 100)
    el = 0
    while agent.env.get_total_steps() < 2e8:
        el += 1
        t += 1
        if render:
            agent.env.render()
        if not agent.buffer.good_to_learn or np.random.random_sample() < exploration_schedule.value(t):
            acttime, action = timeit(lambda: agent.env.random_action())
        else:
            acttime, action = timeit(lambda: agent.act(obs))
        acttimes.append(acttime)

        envtime, (next_obs, reward, done, _) = timeit(lambda: agent.env.step(action))
        envtimes.append(envtime)
        
        addtime, _ = timeit(lambda: agent.add_data(obs, action, reward, next_obs, done))
        addtimes.append(addtime)
        if not background_learning and agent.buffer.good_to_learn:
            learntime, _ = timeit(lambda: agent.atari_learn(t))
            learntimes.append(learntime)

        obs = agent.env.reset() if done else next_obs
        if done:
            episode_lengths.append(el)
            el = 0

        if t % log_steps == 0:
            episode_scores = agent.env.get_episode_rewards()
            # episode_lengths = agent.env.get_episode_lengths()
            eps_len = agent.env.get_length()
            score = episode_scores[-1]
            avg_score = np.mean(episode_scores[-100:])
            eps_len = episode_lengths[-1]
            avg_eps_len = np.mean(episode_lengths)
            if hasattr(agent, 'stats'):
                agent.record_stats(score=score, avg_score=avg_score, eps_len=eps_len, avg_eps_len=avg_eps_len)

            log_info = {
                'ModelName': f'{agent.args["algorithm"]}-{agent.model_name}',
                'Timestep': f'{(t//1000):3d}k',
                'ActTime': utils.timeformat(np.mean(acttimes)),
                'EnvTime': utils.timeformat(np.mean(envtimes)),
                'AddTime': utils.timeformat(np.mean(addtimes)),
                'LearnTime': utils.timeformat(np.mean(learntimes) if learntimes else 0),
                'Iteration': len(episode_scores),
                'Score': score,
                'AvgScore': avg_score,
                'EpsLen': eps_len,
                'AvgEpsLen': avg_eps_len
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
    agent = Agent('Agent', agent_args, env_args, buffer_args, log_tensorboard=False, log_stats=True, log_params=False, device='/device:GPU:0')
    if agent_args['background_learning']:
        utils.pwc('Background Learning...')
        lt = threading.Thread(target=agent.background_learning, daemon=True)
        lt.start()
    else:
        utils.pwc('Foreground Learning...')
    model = agent_args['model_name']
    utils.pwc(f'Model {model} starts training')
    
    train(agent, render, log_steps=int(1e4), background_learning=agent_args['background_learning'])