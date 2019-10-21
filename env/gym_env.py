from collections import deque
import numpy as np
import gym
import ray

from utility import tf_distributions
from utility.utils import pwc
from env.wrappers import TimeLimit, get_wrapper_by_name
from env.atari_wrappers import make_deepmind_atari
from utility.debug_tools import assert_colorize



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
        self.epslen = 0
        self.early_done = 0
        
        self.env_reset = env.reset
        self.EnvType.reset = self.reset
        self.env_step = env.step
        self.EnvType.step = self.step
        self.EnvType.early_done = self.early_done

        self.EnvType.get_episode_score = lambda _: self.score
        self.EnvType.get_episode_length = lambda _: self.epslen
        self.EnvType.get_obs_stack = lambda _: np.concatenate(self.obs_stack, axis=-1)
        self.obs_stack = deque(maxlen=4)

    def __call__(self, *args, **kwargs):
        self.env = self.EnvType(*args, **kwargs)

        return self.env

    def reset(self):
        self.score = 0
        self.epslen = 0
        self.early_done = 0
        obs = self.env_reset(self.env)
        for _ in range(3):
            self.obs_stack.append(np.zeros_like(obs))
        self.obs_stack.append(obs)
        return obs

    def step(self, action):
        next_obs, reward, done, info = self.env_step(self.env, action)
        self.score += np.where(self.early_done, 0, reward)
        self.epslen += np.where(self.early_done, 0, 1)
        self.early_done = np.array(done)
        self.obs_stack.append(next_obs)

        return next_obs, reward, done, info

@envstats
class GymEnv:
    def __init__(self, args):
        if 'atari' in args and args['atari']:
            self.env = env = make_deepmind_atari(args)
        else:
            self.env = env = gym.make(args['name'])
            # Monitor cannot be used when an episode is terminated due to reaching max_episode_steps
            if 'log_video' in args and args['log_video']:
                pwc(f'video will be logged at {args["video_path"]}')
                self.env = env = gym.wrappers.Monitor(self.env, args['video_path'], force=True)

        env.seed(args['seed'])

        self.obs_space = env.observation_space.shape

        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.n_envs = 1
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def get_episode_rewards(self):
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


@envstats
class GymEnvVec:
    """ Not tested yet """
    def __init__(self, args):
        assert_colorize('n_envs' in args, f'Please specify n_envs in args.yaml beforehand')
        self.n_envs = n_envs = args['n_envs']
        self.envs = [make_deepmind_atari(args) for i in range(n_envs)]
        [env.seed(args['seed'] + 10 * i) for i, env in enumerate(self.envs)]

        env = self.envs[0]
        self.obs_space = env.observation_space.shape
        self.is_action_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = env.action_space
        self.action_dim = env.action_space.n if self.is_action_discrete else env.action_space.shape[0]
        self.action_dist_type = action_dist_type(env)
        
        self.max_episode_steps = int(float(args['max_episode_steps'])) if 'max_episode_steps' in args \
                                    else env.spec.max_episode_steps

    def get_episode_rewards(self):
        return np.array([get_wrapper_by_name(env, 'Monitor').get_episode_rewards() for env in self.envs])
    
    def get_episode_lengths(self):
        return np.array([get_wrapper_by_name(env, 'Monitor').get_episode_lengths() for env in self.envs])

    def get_total_steps(self):
        return np.array([get_wrapper_by_name(env, 'Monitor').get_total_steps() for env in self.envs])

    def random_action(self):
        return [env.action_space.sample() for env in self.envs]
        
    def reset(self):
        return [env.reset() for env in self.envs]
    
    def step(self, actions):
        actions = np.squeeze(actions)
        return list(zip(*[env.step(a) for env, a in zip(self.envs, actions)]))


def create_env(args):
    if 'n_envs' not in args or args['n_envs'] == 1:
        return GymEnv(args)
    else:
        return GymEnvVec(args)
