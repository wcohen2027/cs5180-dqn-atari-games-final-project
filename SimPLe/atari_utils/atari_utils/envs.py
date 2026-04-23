import cv2

# See https://stackoverflow.com/questions/54013846/pytorch-dataloader-stucked-if-using-opencv-resize-method
# See https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import torch
import numpy as np
from gymnasium.wrappers import TimeLimit

from atari_utils.utils import one_hot_encode, DummyVecEnv, VecEnv


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    """Take action FIRE on reset for environments that require it."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class VecEnvWrapper(VecEnv):

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()

    def close(self):
        return self.venv.close()

    def get_images(self):
        return self.venv.get_images()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    @property
    def unwrapped(self):
        if isinstance(self.venv, VecEnvWrapper):
            return self.venv.unwrapped
        return self.venv


def _make_shmem_vec_env(env_fns):
    return DummyVecEnv(env_fns)


class WarpFrame(gym.ObservationWrapper):

    def __init__(self, env, width=84, height=84, grayscale=True, inter_area=False):
        super().__init__(env)

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.inter_area = inter_area

        channels = 1 if grayscale else self.env.observation_space.shape[-1]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = cv2.resize(
            obs,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA if self.inter_area else cv2.INTER_NEAREST
        )

        obs = torch.tensor(obs, dtype=torch.uint8)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(-1)
        obs = obs.permute((2, 0, 1))

        return obs


class RenderingEnv(gym.ObservationWrapper):

    def observation(self, observation):
        self.render()
        return observation


class ClipRewardEnv(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.cum_reward = 0

    def reset(self, **kwargs):
        self.cum_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.cum_reward += reward
        if done:
            info['r'] = self.cum_reward
            self.cum_reward = 0
        return observation, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        return (reward > 0) - (reward < 0)


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPytorchWrapper(VecEnvWrapper):

    def __init__(self, venv, device, nstack=4):
        self.venv = venv
        self.device = device
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        obs = torch.tensor(obs).to(self.device)
        rews = torch.tensor(rews).unsqueeze(1)
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        obs = torch.tensor(obs).to(self.device)
        self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs


class VecRecorderWrapper(VecEnvWrapper):

    def __init__(self, venv, gamma, stacking, device):
        super().__init__(venv)
        self.venv = venv
        self.gamma = gamma
        self.stacking = stacking
        self.device = device

        assert self.venv.num_envs == 1

        self.buffer = []
        self.obs = None
        self.initial_frames = None

    def new_epoch(self):
        self.initial_frames = None

    def get_first_small_rollout(self):
        return self.initial_frames

    def add_interaction(self, action, reward, new_obs, done):
        obs = self.obs.squeeze().byte().to(self.device)
        action = one_hot_encode(action.squeeze(), self.action_space.n).to(self.device)
        reward = (reward.squeeze() + 1).byte().to(self.device)
        new_obs = new_obs.squeeze().byte()
        new_obs = new_obs[-len(new_obs) // self.stacking:].to(self.device)
        done = torch.tensor(done[0], dtype=torch.uint8).to(self.device)
        self.buffer.append([obs, action, reward, new_obs, done, None])

    def sample_buffer(self, batch_size):
        if self.buffer[0][5] is None:
            return None

        samples = self.buffer[0]
        data = [torch.empty((batch_size, *sample.shape), dtype=sample.dtype) for sample in samples]

        for i in range(batch_size):
            value = None
            while value is None:
                index = int(torch.randint(len(self.buffer), size=(1,)))
                samples = self.buffer[index]
                value = samples[5]
            for j in range(len(data)):
                data[j][i] = samples[j]

        return data

    def reset(self):
        self.obs = self.venv.reset()
        for _ in range(self.stacking - 1):
            self.obs = self.venv.step(torch.tensor(0))[0].clone()
        if self.initial_frames is None:
            self.initial_frames = self.obs[0]
        return self.obs

    def step(self, action):
        new_obs, reward, done, infos = self.venv.step(action)

        self.add_interaction(action, reward, new_obs, done)

        if done:
            value = torch.tensor(0.).to(self.device)
            self.buffer[-1][5] = value
            index = len(self.buffer) - 2
            while reversed(range(len(self.buffer) - 1)):
                value = (self.buffer[index][2] - 1).to(self.device) + self.gamma * value
                self.buffer[index][5] = value
                index -= 1
                if self.buffer[index][4] == 1:
                    break

        self.obs = new_obs.clone()

        return new_obs, reward, done, infos

    def step_wait(self):
        return self.venv.step_wait()


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        obs = None
        total_reward = 0.0
        terminated = False
        truncated = False
        info = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def _make_env(
        env_name,
        render=False,
        max_episode_steps=18000,
        frame_shape=(1, 84, 84),
        inter_area=False,
        noop_max=30
):
    render_mode = 'human' if render else None
    env = gym.make(f'ALE/{env_name}-v5', render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = SkipEnv(env, skip=4)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    grayscale = frame_shape[0] == 1
    height, width = frame_shape[1:]
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale, inter_area=inter_area)
    env = ClipRewardEnv(env)
    return env


def make_envs(env_name, num, device, stacking=4, record=False, gamma=0.99, buffer_device='cpu', **kwargs):
    env_fns = [lambda: _make_env(env_name, **kwargs)]
    kwargs_no_render = kwargs.copy()
    kwargs_no_render['render'] = False
    env_fns += [lambda: _make_env(env_name, **kwargs_no_render)] * (num - 1)
    if num == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = _make_shmem_vec_env(env_fns)
    env = VecPytorchWrapper(env, device, nstack=stacking)
    if record:
        env = VecRecorderWrapper(env, gamma, stacking, buffer_device)
    return env


def make_env(env_name, device, **kwargs):
    return make_envs(env_name, 1, device, **kwargs)
