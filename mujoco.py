import numpy as np
import gym
from gym import spaces
from codes.utils.util import show_tensor_as_image

class MujocoRGBObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=False, get_obs=False):

        super(MujocoRGBObservationWrapper, self).__init__(env)
        self.height = height
        self.width = width
        self.should_render = should_render
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(height, width, 3), dtype=np.uint8)
        if not get_obs:
            self.env.unwrapped._get_obs = lambda: None

    def viewer_setup(self):
        if 'track' not in self.env.unwrapped.model.camera_names:
            raise ValueError()
        camera_id = self.env.unwrapped.model.camera_name2id('track')
        self.env.unwrapped.viewer.cam.type = 2
        self.env.unwrapped.viewer.cam.fixedcamid = camera_id
        # Hide the overlay
        self.env.unwrapped.viewer._hide_overlay = True

    def observation(self, observation):
        # Observation of the form HXWXC
        if 'rgb_array' not in self.metadata['render.modes']:
            raise AttributeError()
        if self.env.unwrapped.viewer is None:
            self.env.unwrapped._get_viewer()
            self.viewer_setup()
        viewer = self.env.unwrapped._get_viewer()
        if(self.should_render):
            viewer.render()

        observation = viewer.read_pixels(self.width, self.height, depth=False)
        # show_tensor_as_image(observation)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Add the full state to infos
        info.update(state=self.env.unwrapped._get_obs())
        return self.observation(observation), reward, done, info

    def reset(self):

        state = self.env.reset()
        rewards = None
        done = None
        info = {
            "reward_run": None,
            "reward_ctrl": None,
            "state": state
        }
        return self.observation(state), rewards, done, info

class MujocoGrayObservationWrapper(MujocoRGBObservationWrapper):
    def __init__(self, env, height=84, width=84, should_render=True):
        super(MujocoGrayObservationWrapper, self).__init__(env,
            height=height, width=width, should_render=should_render)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(height, width, 1), dtype=np.uint8)


    def observation(self, observation):
        observation = super(MujocoGrayObservationWrapper,
            self).observation(observation)
        # QKFIX: Use raw Numpy instead of OpenCV for conversion RGB2Gray to
        # avoid any conflict with mujoco_py. The weightings are given by
        # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
        observation = np.dot(observation, [0.299, 0.587, 0.114])
        observation = np.expand_dims(observation, axis=2)

        return observation
