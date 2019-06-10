import numpy as np
import gym
import ray

from utility import tf_distributions
from env.wrappers import TimeLimit
from env.atari_wrappers import wrap_deepmind, get_wrapper_by_name
from utility.utils import assert_colorize


def action_dist_type(env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        return tf_distributions.Categorical
    elif isinstance(env.action_space, gym.spaces.Box):
        return tf_distributions.DiagGaussian
    else:
        raise NotImplementedError

class envstats:
    """ Provide Environment Stats Records """
    def __init__(self, env):
        self.EnvType = env
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        
        self.env_reset = env.reset
        self.EnvType.reset = self.reset
        self.env_step = env.step
        self.EnvType.step = self.step
        self.EnvType.early_done = self.early_done

        self.EnvType.get_score = lambda _: self.score
        self.EnvType.get_length = lambda _: self.eps_len
    def __call__(self, *args, **kwargs):
        self.env = self.EnvType(*args, **kwargs)

        return self.env

    def reset(self):
        self.score = 0
        self.eps_len = 0
        self.early_done = 0
        
        return self.env_reset(self.env)

    def step(self, action):
        next_obs, reward, done, info = self.env_step(self.env, action)
        self.score += np.where(self.early_done, 0, reward)
        self.eps_len += np.where(self.early_done, 0, 1)
        self.early_done = np.array(done)

        return next_obs, reward, done, info

@envstats
class GymEnv:
    def __init__(self, args):
        if 'atari' in args and args['atari']:
            # self.env = env = self._make_atari(args)
            self.env = env = gym.make(args['name'])
            self.env = env = gym.wrappers.Monitor(self.env, args['video_path'], force=True)
            self.env = env = wrap_deepmind(env)
        else:
            self.env = env = gym.make(args['name'])
            # Monitor cannot be used when an episode is terminated due to reaching max_episode_steps
            if 'video_path' in args:
                self.env = env = gym.wrappers.Monitor(self.env, args['video_path'], force=True)

        env.seed(args['seed'])

        self.obs_space = env.observation_space.shape

        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = 1
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def get_episode_scores(self):
        return get_wrapper_by_name(self.env, 'Monitor').get_episode_rewards()
    
    def get_episode_lengths(self):
        return get_wrapper_by_name(self.env, 'Monitor').get_episode_lengths()

    def get_total_steps(self):
        return get_wrapper_by_name(self.env, 'Monitor').get_total_steps()

    def reset(self):
        return self.env.reset()

    def random_action(self):
        return self.env.action_space.sample()
        
    def step(self, action):
        action = np.squeeze(action)
        return self.env.step(action)
        
    def render(self):
        return self.env.render()

    def _make_atari(self, args):
        env = make_atari(args['name'], max_episode_steps=1000)
        if 'video_path' in args:
            env = gym.wrappers.Monitor(env, args['video_path'], force=True)
        if 'atari' in args and args['atari']:
            env = wrap_deepmind(env)
        return env


@envstats
class GymEnvVec:
    def __init__(self, args):
        assert_colorize('n_envs' in args, f'Please specify n_envs in args.yaml beforehand')
        n_envs = args['n_envs']
        self.envs = [gym.make(args['name']) for i in range(n_envs)]
        [env.seed(args['seed'] + 10 * i) for i, env in enumerate(self.envs)]

        env = self.envs[0]
        self.obs_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = n_envs
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
