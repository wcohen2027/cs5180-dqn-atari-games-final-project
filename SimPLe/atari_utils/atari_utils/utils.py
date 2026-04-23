import torch
import logging

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# Minimal VecEnv base class (replaces baselines.common.vec_env.VecEnv)
class VecEnv:
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise NotImplementedError

    def step_async(self, actions):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


def obs_space_info(obs_space):
    """Return keys, shapes, dtypes for an observation space."""
    if isinstance(obs_space, spaces.Dict):
        keys = sorted(obs_space.spaces.keys())
        shapes = {k: obs_space.spaces[k].shape for k in keys}
        dtypes = {k: obs_space.spaces[k].dtype for k in keys}
    elif isinstance(obs_space, spaces.Tuple):
        keys = list(range(len(obs_space.spaces)))
        shapes = {k: obs_space.spaces[k].shape for k in keys}
        dtypes = {k: obs_space.spaces[k].dtype for k in keys}
    else:
        keys = [None]
        shapes = {None: obs_space.shape}
        dtypes = {None: obs_space.dtype}
    return keys, shapes, dtypes


def copy_obs_dict(obs):
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


class DummyVecEnv(VecEnv):
    """
    Fixes the close function
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def close(self):
        for env in self.envs:
            env.close()

    def close_extras(self):
        for env in self.envs:
            env.close()

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # ALE expects a plain integer scalar, not a numpy array
            if isinstance(action, np.ndarray):
                action = int(action.flat[0])
            obs, self.buf_rews[e], terminated, truncated, self.buf_infos[e] = self.envs[e].step(action)
            self.buf_dones[e] = terminated or truncated
            if self.buf_dones[e]:
                obs, _ = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs, _ = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render() for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render()
        else:
            raise NotImplementedError('render not supported for multiple envs')


def sample_with_temperature(logits, temperature):
    assert temperature > 0
    reshaped_logits = logits.view((-1, logits.shape[-1])) / temperature
    reshaped_logits = torch.exp(reshaped_logits)
    choices = torch.multinomial(reshaped_logits, 1)
    choices = choices.view((logits.shape[:len(logits.shape) - 1]))
    return choices


def print_config(config):
    args = vars(config)
    max_len = 0
    for arg in args:
        max_len = max(max_len, len(arg))
    for arg in args:
        value = str(getattr(config, arg))
        display = '{:<%i}: {}' % (max_len + 1)
        print(display.format(arg, value))


def disable_baselines_logging():
    # baselines no longer used; nothing to configure
    pass


def one_hot_encode(action, n, dtype=torch.uint8):
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    res = action.long().view((-1, 1))
    res = torch.zeros((len(res), n)).to(res.device).scatter(1, res, 1).type(dtype).to(res.device)
    res = res.view((*action.shape, n))

    return res
