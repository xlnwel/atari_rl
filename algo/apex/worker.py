import os
from time import time
from collections import deque
import numpy as np
import ray

from utility.utils import pwc


def get_worker(BaseClass, *args, **kwargs):

    @ray.remote(num_cpus=1)
    class Worker(BaseClass):
        """ Interface """
        def __init__(self, 
                    name, 
                    worker_no,
                    args, 
                    env_args,
                    buffer_args,
                    max_episodes,
                    sess_config=None, 
                    save=False, 
                    log_tensorboard=False, 
                    log_params=False,
                    log_stats=False,
                    device=None):
            self.no = worker_no
            buffer_args['type'] = 'local'
            args['batch_size'] = 1
            self.max_episodes = max_episodes
            self.random_eps = float(buffer_args['min_size']) / args['n_workers'] // 1000

            super().__init__(name, 
                            args, 
                            env_args,
                            buffer_args,
                            sess_config=sess_config,
                            save=save,
                            log_tensorboard=log_tensorboard,
                            log_params=log_params,
                            log_stats=log_stats,
                            device=device)

            pwc('Worker {} has been constructed.'.format(self.no), 'cyan')

        def sample_data(self, learner):
            # I intend not to synchronize the worker's weights at the beginning for initial exploration 
            score_deque = deque(maxlen=100)
            eps_len_deque = deque(maxlen=100)
            episode_i = 0
            t = 0
            
            while True:
                obs = self.env.reset()

                while True:
                    t += 1
                    random_act = episode_i < self.random_eps
                    action, q = self.act(obs, random_act=random_act, return_q=True)
                    next_obs, reward, done, _ = self.env.step(action)
                    
                    self.buffer.add(obs, action, reward, done, q)

                    obs = next_obs

                    if done:
                        break

                self.buffer['priority'] = self.buffer.compute_priority(self.gamma, self.prio_epsilon, self.prio_alpha)
                
                learner.merge_buffer.remote(dict(self.buffer), self.buffer.idx)
                self.buffer.reset()

                score = self.env.get_score()
                eps_len = self.env.get_length()
                episode_i += 1
                score_deque.append(score)
                eps_len_deque.append(eps_len)
                stats = dict(t=t, score=score, avg_score=np.mean(score_deque), 
                            eps_len=eps_len, avg_eps_len=np.mean(eps_len_deque), 
                            worker_no=self.no)
                            
                learner.record_stats.remote(stats)
                
                # pull weights from learner
                if episode_i % self.max_episodes == 0:
                    weights = ray.get(learner.get_weights.remote())
                    self.variables.set_flat(weights)

    return Worker.remote(*args, **kwargs)
