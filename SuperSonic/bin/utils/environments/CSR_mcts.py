import random
from copy import deepcopy
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
import compiler_gym
from compiler_gym.envs.llvm.llvm_env import make_benchmark

from compiler_gym import CompilerEnvState, CompilerEnvStateWriter
from compiler_gym.compiler_env_state import CompilerEnvStateReader
import time
import sqlite3


class csr_rl:
    """
    Wrapper for gym MCTS environment where the reward
    is accumulated to the end
    """

    # def init_from_env_config(self,env_config):
    #     self.inference_mode = env_config.get("inference_mode", False)
    #     if self.inference_mode:
    #         self.improvements = []

    def __init__(self, env_config):
        # self.init_from_env_config(env_config)
        self.benchmarks = env_config.get('benchmark')
        self.log_path = env_config.get('log_path')
        self.pass_path = env_config.get('pass_path')
        self.deadline = env_config.get('deadline')
        # print(self.benchmarks)
        self.env = gym.make(
            "llvm-autophase-ic-v0",  # selects the compiler to use
            benchmark=make_benchmark([self.benchmarks]),  # selects the program to compile
        )

        self.action_space = Discrete(self.env.action_space.n)

        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )

        self.running_reward = 0
        self.episode_reward = 0

        # record best
        self.view = 0
        self.best_actions = []
        self.time = time.time()

        self.num = 0
        self.seeds = env_config.get('seed')
        self.flag = False
        self.best_command = ""
        self.env.seed(env_config.get('seed'))
        self.score = 0
        # self.env.init()
        print(f"seed:{self.seeds}")

    def reset(self):
        self.running_reward = 0
        self.episode_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.ones((self.env.action_space.n,), dtype=int),
        }

    def step(self, action):

        used_time = time.time() - self.time
        benchmarks = self.benchmarks.split("/")[-1]
        with open(f"{self.pass_path}/{benchmarks}_pass.txt", "a+") as f:
            f.write("=================Step======================\r\n")
            f.write(f"step:{self.step}\r\nactions:{self.env.actions}\r\n")
            f.write(f"command line:{self.env.commandline()}\r\n")
            f.write(
                f"time={used_time:.3},action_len={len(self.env.actions)},episode_reward={self.env.episode_reward:.3%},reward_all={self.view:.3%}\r\n")
        try:

            conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
            c = conn.cursor()
            sql = "INSERT INTO CSR (TIME, BENCHMARK, RESULT, REWARD) \
                            VALUES (?, ?, ?, ?)"
            # print("time",round(time.time() - self.time))
            c.execute(sql, (time.time() - self.time, benchmarks, self.env.commandline(), self.env.episode_reward))

            conn.commit()
            conn.close()
        except Exception as e:
            # print("11111111111111111111111111111")
            print(e)

        obs, rew, done, info = self.env.step(action)

        self.running_reward += rew
        self.episode_reward = self.episode_reward + rew

        if self.env.episode_reward > self.view:
            self.view = self.env.episode_reward
            self.best_actions = self.env.actions
            self.best_command = self.env.commandline()

        if len(self.env.actions) > self.env.action_space.n:
            used_time = time.time() - self.time

            if used_time > self.deadline and not self.flag:
                self.flag = True
                with open(f"{self.log_path}/{benchmarks}_result_ppo.txt", "a+") as f:
                    # pass_time = time.time() - self.time
                    f.write(f"actions{str(self.best_actions)}\r\ncommand_line:{self.best_command}\r\n")

                    f.write(f"seeds:{self.seeds}\r\nreward: " + str(self.view) + '\r\n ')
                    f.close()

            done = True
            self.score = self.running_reward if done else 0
            # print(f"time={used_time},quality_finally={self.view:.3%}")
            # print("action_masks",np.ones((self.env.action_space.n,)))

        return (
            {"obs": obs, "action_mask": np.ones((self.env.action_space.n,), dtype=int)},
            self.score,
            done,
            info,
        )

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = state[0].fork()
        obs = self.env.observation["Autophase"]
        return {"obs": obs, "action_mask": np.ones((self.env.action_space.n,), dtype=int)}

    def get_state(self):
        env_1 = self.env.fork()
        return env_1, self.running_reward
