# 这里是rltvm底层
import math
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.spaces import Discrete, Dict, Box
import random
from gensim.models.doc2vec import Doc2Vec
from compiler_gym.mdp_search.action import action_functions
# from compiler_gym.mdp_search.observation import get_observation, obs_init
from compiler_gym.mdp_search.reward import reward_function

class TvmEnv(gym.Env):
    def __init__(self, observation, action, reward):
        self.observation = observation
        (
            self.interleave_action_meaning,
            self.action_space,
            self.observation_space,
        ) = get_action("map")
        self.state = None
        

    def get_obs(self, action, state_code):
        
        observation = get_observation("Doc2vec", state_code)
        return observation  # TODO：这里return obs，需要交互，输入action，调用远程计算后得到obs。

    def step(self, action, state_code, reward):
        try:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        except:
            action=1
            print("There is something wrong, put action to 1")
        obs = self.get_obs(action, state_code)
        self.state = obs
        done = True  #
        # rew = self.get_reward(action)
        rew = reward
        info = {}
        return self.state,rew,done,info
        
    def reset(self):
        self.state = obs_init("Doc2vec")
        # self.state = np.random.random(128)
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None

# if __name__ == '__main__':
#     env = TVMEnv()
#     env.reset()
#     env.step(env.action_space.sample())
#     print(env.state)
#     env.step(env.action_space.sample())
#     print(env.state)
