# https://www.cnblogs.com/pinard/p/10609228.html

# 这里改了, 这里强化学习端作为客户端, 向tvm进行请求

import grpc
import tvm_pb2
import tvm_pb2_grpc

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
# for _ in range(1000):
#     init_space[_] = 1 # 这里根据传回来的最大action范围进行限定
# for _ in range(maxLen_start):
#     init_space[_] = 0 # 这里根据传回来的最大action范围进行限定

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
class mcts(gym.Env):
    def init_from_env_config(self, env_config):
        self.inference_mode = env_config.get("inference_mode", False)
        if self.inference_mode:
            self.improvements = []

    def __init__(self, env_config):
        self.init_from_env_config(env_config)
        self.maxLen = 200
        self.env = gym.make(
            "Tvm-v0",
            observation=env_config.get("observation"),
            action=env_config.get("action"),
            reward=env_config.get("reward"),
        )

        self.interleave_action_meaning = [
            _ for _ in range(maxLen_start)
        ]  # TODO: 根据需要更改action的空间
        self.action_space = spaces.Discrete(
            len(self.interleave_action_meaning)
        )  # action_len
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,  # 调用的是cg的环境
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.running_reward = 0
        # self.server.stop() # 关闭服务

    def reset(self):
        self.running_reward = 0
        return {"obs": self.env.reset(), "action_mask": init_space}
        # self.action_space.n/2
        # TODO：return这里，返回初始的obs和action_mask就可以

    def step(self, action):
        # print("给客户端上锁, 等到grpc请求完成在进行")
        # lock.acquire() # 这里判断是否已经有锁, 没有的话就上锁, 然后向下执行, 否则等待解锁
        # global action_remote
        # action_remote = action

        with grpc.insecure_channel("localhost:50060") as channel:  # 定义grpc端口号
            # print("开始请求")
            stub = tvm_pb2_grpc.TvmServiceStub(channel)  # 定义通道
            response = stub.GetTvm(
                tvm_pb2.TvmRequest(action=action)
            )  # 这是请求的数据给服务端, 然后在读取服务端返回来的数据
            # response = stub.GetTvm(tvm_pb2.TvmRequest(state='world', reward=3.1415926)) # 这是请求的数据给服务端, 然后在读取服务端返回来的数据
            # print(f"Client received: {response.reward}, {response.state}, {response.maxLen}")
            # print(response.reward, " action=",action)

            obs, rew, done, info = self.env.step(
                action, response.state, response.reward
            )
            # obs, rew, done, info = self.env.step(action_remote, state_remote, reward_remote)# TODO: state_remote, reward_remote
            self.running_reward += rew
            score = self.running_reward if done else 0

            if self.maxLen != response.maxLen:
                # print("这里修改了malen")
                self.maxLen = response.maxLen
                for _ in range(self.maxLen):
                    init_space[_] = 1  # 这里根据传回来的最大action范围进行限定
                for _ in range(self.maxLen, maxLen_start):
                    init_space[_] = 0  # 这里根据传回来的最大action范围进行限定
            init_space[action] = 0

        return (
            {"obs": obs, "action_mask": init_space},
            score,
            done,
            info,
        )  # 下一步的obs，reward等信息

    def set_state(self, state):  # 设置模拟的内部状态。state：要在环境中设置的目标状态。
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": init_space}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
