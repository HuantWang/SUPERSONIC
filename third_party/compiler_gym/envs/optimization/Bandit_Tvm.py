# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines policy search environments for TVM"""
import math
import gym
from gym.utils import seeding
import numpy as np
from gym import spaces
from SuperSonic.policy_definition.action import action_functions

# from compiler_gym.mdp_search.observation import get_observation, obs_init
from SuperSonic.policy_definition.reward import reward_function
import sys
import SuperSonic.utils.engine.tasks_engine as start_engine
import shutil
import os
import json
import sqlite3

# python SuperSonic/PolSear_MTL/tvm_main.py
class BanditTvmEnv(gym.Env):
    """A :class:
        we formulate RL component search as a multi-armed bandit (MAB) problem.
    Observation:
        Type: Box(20)
        By default, the meta-optimizer runs each client RL for past few exploration
    steps during a trial to allow it to converge before taking the
    observation. We use the 20 most recently chosen policy architectures as the state.
    Actions:
        Type: Discrete(n)
        Given a budget of 𝑛 candidate RL component configurations (or slot machines)
    Reward:
        We then use the most frequently chosen policy architecture as the outcome of policy search.


    """

    def __init__(self, policy, data):
        """Initialise with an environment.
        :param policy: The candidate policy list.
        :param data: The program that programer intend to optimize.
        """
        self.info = {}
        self.all_policy, self.n_bandits = policy[0], policy[1]
        self.dataset = data
        self.action_space = spaces.Discrete(self.n_bandits)
        self.obsv_size = 100
        self.observation_space = spaces.box.Box(
            -1.0, 1.0, shape=(self.obsv_size,), dtype=np.float64
        )  #
        self.num = 0
        self.observation = np.zeros(self.obsv_size)
        self.actions = []
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self, action):
        """ feedback the observation."""

        if len(self.actions) < self.obsv_size:
            self.observation = np.hstack(
                (np.array(self.actions), np.zeros(self.obsv_size - len(self.actions)))
            )[: self.obsv_size]
        else:
            self.observation = self.actions[-self.obsv_size :]
        return self.observation

    def generate_reward(self, action):
        """ feedback the reward."""
        # the path to save execution performance
        DocSave = "SuperSonic/logs/model_save"

        # Delete dir
        if os.path.exists(DocSave):
            shutil.rmtree(DocSave)

        exist_policy = self.all_policy[action]

        print(f"exist_policy:{exist_policy}")

        start_engine.Tvm(exist_policy,self.dataset).main()

        # load loss from json

        for relpath, dirs, files in os.walk(DocSave):
            for _ in files:
                if _.split(".")[-1] == "json":
                    full_path = os.path.join(relpath, _)
        with open(full_path, "r") as f:
            data = json.load(f)
        try:
            reward = data["checkpoints"][0]["last_result"]["episode_reward_max"]
        except:
            reward = 0
        conn = sqlite3.connect("SuperSonic/SQL/supersonic.db")
        c = conn.cursor()
        sql = "INSERT INTO SUPERSONIC (ID,TASK,ACTION,REWARD) \
                                                              VALUES (?, ?, ?, ?)"
        c.execute(sql, (self.num, "TVM", action, reward))
        conn.commit()
        conn.close()

        return reward

    def step(self, action):
        """Take a step.

                :param action: An action, or a sequence of actions. When multiple
                    actions are provided the observation and reward are returned after
                    running all of the actions.

                :return: A tuple of observation, reward, done, and info. Observation and
                    reward are None if default observation/reward is not set. If done is
                    True, observation and reward may also be None (e.g. because the
                    service failed).
        """
        assert self.action_space.contains(action)
        self.num = self.num + 1
        self.actions.append(action)
        # reward = self.get_reward_1(action)
        reward = self.generate_reward(action)
        obs = self.get_observation(action)
        done = True

        return obs, reward, done, self.info

    def reset(self):
        """ reset the RL environment.
                """
        return np.random.rand(20)

    def render(self, mode="human", close=False):
        pass

    def close(self):
        return None
