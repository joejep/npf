import collections

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class AddImageObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.image_shape=(160,120,1)
        self.observation_space=gym.spaces.Box(
            low=0,
            high=255,
            shape=self.image_shape,
            dtype=np.uint8,
        )
        self.camera_id=2
    
    def reset(self,**kwargs):
        obs,info= self.env.reset(**kwargs)
        obs = self.env.render(camera_id=self.camera_id)
        return obs,info
    def render(self):
        return self.env.render(camera_id=self.camera_id)
        # return 

    def step(self, action):
        obs, reward, done,terminate, info = self.env.step(action)
        obs = self.env.render(camera_id=self.camera_id)
        return obs,reward,done,terminate,info
    