import numpy as np
import gym
from gym import spaces
from collections import deque

class FramesStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super(FramesStack, self).__init__(env)
        self.num_stack = num_stack
        self.obs_shape = self.env.observation_space.shape
        self._buffer = deque([], maxlen=self.num_stack)
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8,
            shape=(self.num_stack * self.obs_shape[0],) + self.obs_shape[1:])

    def _get_observation(self):
        assert len(self._buffer) == self.num_stack
        return LazyFrames(list(self._buffer))

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        null_observation = np.zeros(self.obs_shape, dtype=np.uint8)
        for _ in range(self.num_stack - 1):
            self._buffer.append(null_observation)
        self._buffer.append(observation)
        return self._get_observation()

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self._buffer.append(observation)
        return (self._get_observation(), reward, done, info)

class RollAxisObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(RollAxisObservationWrapper, self).__init__(env)
        obs_shape = self.env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8,
            shape=(obs_shape[-1],) + obs_shape[:-1])

    def observation(self, observation):
        return np.rollaxis(observation, axis=2)

    def reset(self):
        observation, reward, done, info = self.env.reset()
        return self.observation(observation), reward, done, info

class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    @property
    def out(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self.out
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._frames is None:
            return len(self.out)
        else:
            return len(self._frames)
