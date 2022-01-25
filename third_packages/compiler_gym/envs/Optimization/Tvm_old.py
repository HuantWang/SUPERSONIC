# 这里是rltvm底层
import math
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.spaces import Discrete, Dict, Box
import random
from gensim.models.doc2vec import Doc2Vec
from compiler_gym.mdp_search.action import get_action
from compiler_gym.mdp_search.observation import get_observation, obs_init
from compiler_gym.mdp_search.reward import get_reward

class TvmEnv(gym.Env):
    # def __init__(self,embedding_approch='/home/huanting/CG/tasks/tvm/AutoTVM/opt_test/doc2vec_tvm.model'):
    def __init__(self, observation, action, reward):
        self.observation_method = observation
        self.action_method = action
        self.reward_method = reward
        # self.embedding=embedding_approch
        # self.interleave_action_meaning = [_ for _ in range(10000)] # TODO: 根据需要更改action的空间
        # self.action_space = spaces.Discrete(len(self.interleave_action_meaning))  # action_len
        # self.observation_space = spaces.Box(low=-100,high=100,shape=(128,),dtype = np.float64)
        (
            self.interleave_action_meaning,
            self.action_space,
            self.observation_space,
        ) = get_action(self.action_method)
        # self.doc2vecmodel = Doc2Vec.load(self.embedding) # TODO: change
        self.state = None
        
    def get_reward(self, method, action, reward):
        # reward_new = get_reward("hamming", action)
        reward_new = state_reward
        return reward_new
    
    def get_obs(self, action, state_code):
        self.state_code = state_code
        # self.state_code = "abc"
        observation = get_observation(self.observation_method, self.state_code)
        # observation = get_observation("Doc2vec", state_code)
        # observation = np.array(
        #     self.doc2vecmodel.infer_vector(state_code.split(), 
        #     steps=6, 
        #     alpha=0.025).tolist())[0:128] #维度是128
        return observation  # TODO：这里return obs，需要交互，输入action，调用远程计算后得到obs。

    def step(self, action, state_code, reward):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        obs = self.get_obs(action, state_code) 
        self.state = obs
        done = True  #
        # rew = self.get_reward(action)
        rew = get_reward(self.reward_method, action, reward)
        info = {}
        return self.state,rew,done,info
        
    def reset(self):
        self.state = obs_init(self.observation_method)
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