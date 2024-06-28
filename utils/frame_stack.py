"""Wrapper that stacks frames."""
from collections import deque

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box

class FrameStack(gym.ObservationWrapper):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
          - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,

    Example:
        >>> import gym
        >>> env = gym.make('CarRacing-v1')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 96, 96, 3)
        >>> obs = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack
        
        
        self.frames = dict(zip(self.observation_space.spaces.keys(), [deque([], maxlen=num_stack) for _ in range(len(self.observation_space.spaces))]))

        self.observation_space = gym.spaces.Dict()
        for key in self.env.observation_space.spaces.keys():
            low = np.repeat(self.env.observation_space[key].low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(
                self.env.observation_space[key].high[np.newaxis, ...], num_stack, axis=0
            )
            self.observation_space[key] = Box(
                low=low, high=high, dtype=self.env.observation_space[key].dtype
            )

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        for key in self.frames.keys():
            assert len(self.frames[key]) == self.num_stack, (len(self.frames[key]), self.num_stack)
        new_frames = dict(zip(self.observation_space.spaces.keys(), [np.stack(self.frames[key], axis=0) for key in self.frames.keys()]))
        return new_frames

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        for key in self.frames.keys():
            self.frames[key].append(observation[key])
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, _ = self.env.reset(**kwargs)
        
        for key in self.frames.keys():
            [self.frames[key].append(obs[key]) for _ in range(self.num_stack)]

        return self.observation(None), {}
    
    def render(self, highres=False):
        if highres:
            img = self.env.env.env.env.observation(None)['image']
            return img.astype(np.float32)/255
        else:
            return self.env.render()