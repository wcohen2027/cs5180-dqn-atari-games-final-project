# From https://github.com/hill-a/stable-baselines
# Stable Baselines' license is located at https://github.com/hill-a/stable-baselines/blob/master/LICENSE

from abc import ABC, abstractmethod
import logging

import torch
from cloudpickle import cloudpickle

from atari_utils.envs import VecEnvWrapper, VecPytorchWrapper
from atari_utils.utils import one_hot_encode
from simple.simulated_env import _make_simulated_env

import multiprocessing
from collections import OrderedDict
from typing import Sequence, Optional, List, Union

import gymnasium as gym
import numpy as np


def _tile_images(imgs):
    """Tile a list of images (H, W, C) or (H, W) into a single image grid."""
    imgs = [np.array(img) for img in imgs]
    n = len(imgs)
    if n == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        c = imgs[0].shape[2]
        grid = np.zeros((rows * h, cols * w, c), dtype=imgs[0].dtype)
    else:
        grid = np.zeros((rows * h, cols * w), dtype=imgs[0].dtype)
    for idx, img in enumerate(imgs):
        r, c_ = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c_ * w:(c_ + 1) * w] = img
    return grid


class CloudpickleWrapper(object):
    def __init__(self, var):
        """
        Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
        :param var: (Any) the variable you wish to wrap for pickling with cloudpickle
        """
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


class _VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    :param num_envs: (int) the number of environments
    :param observation_space: (Gym Space) the observation space
    :param action_space: (Gym Space) the action space
    """
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_attr(self, attr_name, indices=None):
        pass

    @abstractmethod
    def set_attr(self, attr_name, value, indices=None):
        pass

    @abstractmethod
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    @abstractmethod
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def get_images(self) -> Sequence[np.ndarray]:
        raise NotImplementedError

    def render(self, mode: str = 'human'):
        try:
            imgs = self.get_images()
        except NotImplementedError:
            logging.warning('Render not defined for {}'.format(self))
            return

        bigimg = _tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def getattr_depth_check(self, name, already_found):
        if hasattr(self, name) and already_found:
            return "{0}.{1}".format(type(self).__module__, type(self).__name__)
        else:
            return None

    def _get_indices(self, indices):
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    info['terminal_observation'] = observation
                    observation, _ = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'seed':
                remote.send(env.reset(seed=data))
            elif cmd == 'reset':
                observation, _ = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render())
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break


class _SubprocVecEnv(_VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.
    """

    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        _VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed=None):
        for idx, remote in enumerate(self.remotes):
            remote.send(('seed', seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes:
            pipe.send(('render', 'rgb_array'))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def _flatten_obs(obs, space):
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


class SubprocVecEnv(_SubprocVecEnv):

    def __init__(self, env_fns, model, n_action, config):
        super().__init__(env_fns)
        self.model = model
        self.n_action = n_action
        self.config = config
        self.step_count = 0
        self.frames = None
        self.actions = None

    def step_async(self, actions):
        if self.step_count == 0:
            res = self.env_method('get_initial_frames')

            self.frames = torch.stack(res)
            self.frames = self.frames.float() / 255
            self.frames = self.frames.to(self.config.device)

            if self.config.stack_internal_states:
                self.model.init_internal_states(self.config.agents)

        self.step_count += 1

        actions = one_hot_encode(actions, self.n_action, dtype=torch.float32)
        actions = actions.to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            new_states, rewards, values = self.model(self.frames, actions)

        new_states = torch.argmax(new_states, dim=1)
        self.frames = torch.cat((self.frames[:, 3:], new_states.float() / 255), dim=1)
        new_states = (self.frames * 255).byte().detach().cpu()
        rewards = (torch.argmax(rewards, dim=1).detach().cpu() - 1).numpy().astype('float')

        if self.step_count == self.config.rollout_length:
            self.step_count = 0
            rewards += values.detach().cpu().numpy().astype('float')

        if self.config.done_on_last_rollout_step:
            done = self.step_count == self.config.rollout_length
        else:
            done = False

        for remote, arg in zip(self.remotes, zip(new_states, list(rewards))):
            arg = (*arg, done)
            remote.send(('step', arg))
        self.waiting = True


def make_simulated_env(config, model, action_space):
    def constructor(i):
        def _constructor():
            return _make_simulated_env(config, action_space, i == 0)

        return _constructor

    env = SubprocVecEnv([constructor(i) for i in range(config.agents)], model, action_space.n, config)
    env = VecPytorchWrapper(env, config.device, nstack=1)
    return env
