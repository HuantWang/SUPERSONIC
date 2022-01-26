import random
from copy import deepcopy
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
import compiler_gym
from gensim.models.doc2vec import Doc2Vec

#add for grpc
import gym
from gym import spaces
from gym.utils import seeding
import grpc
from SuperSonic.bin.service import schedule_pb2
from SuperSonic.bin.service import schedule_pb2_grpc
import os
import random


class halide_rl:
    """
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    """
    def init_from_env_config(self,env_config):
        self.inference_mode = env_config.get('inference_mode', False)
        if self.inference_mode:
            self.improvements=[]

    def __init__(self, env_config):
        self.init_from_env_config(env_config)
        #for grpc
        #channel = grpc.insecure_channel(env_config.get("target"))
        #self.stub = schedule_pb2_grpc.ScheduleServiceStub(channel)

        log_path = env_config.get("log_path")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # transform stub
        self.env = gym.make("Halide-v0",
                            algorithm_id=env_config.get("algorithm_id"),
                            input_image=env_config.get("input_image"),
                            max_stage_directive=env_config.get("max_stage_directive"),
                            log_path=env_config.get("log_path"),
                            state_function=env_config.get("state_function"),
                            action_function=env_config.get("action_function"),
                            reward_function=env_config.get("reward_function"),
                            )

        self.action_space = Discrete(self.env.action_space.n)
        self.observation_space = Dict({
            "obs": self.env.observation_space,
            "action_mask": Box(low=0, high=1, shape=(self.env.action_space.n, ))
        })
        self.running_reward = 0
        self.num=0

        # print("zczczczczc")
        #self.doc2vecmodel = Doc2Vec.load(env_config.get("embedding"))

    def reset(self):
        self.running_reward = 0
        return {"obs": self.env.reset(), "action_mask": np.array([1]*self.env.action_space.n)}

    def step(self, action):
        # self.num = self.num+1
        # print ("self.num :",self.num)
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        # obs=np.random.rand(128)
        # print(obs)
        # done = True
        score = self.running_reward if done else 0
        return {"obs": obs, "action_mask": np.array([1] * self.env.action_space.n)}, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        print("self.env.unwrapped.state",self.env.unwrapped.state)
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1]*self.env.action_space.n)}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
