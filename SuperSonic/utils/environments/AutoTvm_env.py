import grpc
from SuperSonic.service import schedule_pb2
from SuperSonic.service import schedule_pb2_grpc
# 因为 RPC 应该长时间运行，考虑到性能，还需要用到并发的库。
import time  # 设置系统延时,
from concurrent import futures

_ONE_DAY_IN_SECONDS = 60 * 60 * 24  # 设置服务器运行的默认时间长度
# add by zc
from multiprocessing import Lock
import numpy as np

# 同步锁
import threading

lock = threading.Lock()
lock_s = threading.Lock()

maxLen_start = 4000
init_space = np.array([1 for _ in range(4000)])

from copy import deepcopy
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import gym
import compiler_gym
from gym.spaces import Discrete, Dict, Box
from gym import spaces
import logging
import random

DEBUG = False
# end by zc grpc
logger = logging.getLogger(__name__)

# NeuroVectorizer RL Environment
# client
class autotvm_rl(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(
            "Tvm-v0",
            state_function=env_config.get("state_function"),
            action_function=env_config.get("action_function"),
            reward_function=env_config.get("reward_function"),
        )

        self.maxLen = 200

        self.interleave_action_meaning = [
            _ for _ in range(maxLen_start)
        ]  # TODO: 根据需要更改action的空间
        self.action_space = Discrete(
            maxLen_start
        )  # action_len

        print(self.action_space, "/n=================action space=======/n")

        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,  
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return {"obs": self.env.reset(), "action_mask": init_space}
        # self.action_space.n/2

    def step(self, action):
        with grpc.insecure_channel('localhost:50061') as channel:  
            stub = schedule_pb2_grpc.ScheduleServiceStub(channel) 
            response = stub.GetTvm(
                schedule_pb2.TvmRequest(action=action)
            )  

            obs, rew, done, info = self.env.step(
                action, response.state, response.reward
            )
            self.running_reward += rew
            score = self.running_reward if done else 0

            if self.maxLen != response.maxLen:
                self.maxLen = response.maxLen
                for _ in range(self.maxLen):
                    init_space[_] = 1  
                for _ in range(self.maxLen, maxLen_start):
                    init_space[_] = 0  
            init_space[action] = 0

        return (
            {"obs": obs, "action_mask": init_space},
            score,
            done,
            info,
        ) 

    def set_state(self, state):  
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": init_space}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
