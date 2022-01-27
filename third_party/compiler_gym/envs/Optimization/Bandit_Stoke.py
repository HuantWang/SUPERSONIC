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


class BanditStokeEnv(gym.Env):
    """
               Bandit environment base to allow agents to interact with the class n-armed bandit
               in different variations
               p_dist:
                   A list of probabilities of the likelihood that a particular bandit will pay out
               r_dist:
                   A list of either rewards (if number) or means and standard deviations (if list)
                   of the payout that bandit has
               info:
                   Info about the environment that the agents is not supposed to know. For instance,
                   info can releal the index of the optimal arm, or the value of prior parameter.
                   Can be useful to evaluate the agent's perfomance
               """

    def __init__(self, policy, data):
        self.info = {}
        self.all_policy, self.n_bandits = policy[0], policy[1]
        self.dataset = data
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.box.Box(-1.0, 1.0, shape=(20,), dtype=np.float64)  #
        # self.observation_space = spaces.Discrete(1)
        self.num = 0
        self.obsv_size = 20
        self.observation = np.zeros(self.obsv_size)
        self.actions = []
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self, action):
        # print("actionspace",self.action_space)
        # print("action",action)
        if len(self.actions) < self.obsv_size:
            self.observation = np.hstack((np.array(self.actions), np.zeros(self.obsv_size - len(self.actions))))[
                               :self.obsv_size]
            # print("observation111111",self.observation)
        else:
            self.observation = self.actions[-self.obsv_size:]
            # print("observation222222", self.observation)
        return self.observation

    def generate_reward(self, action):

        # the path to save execution performance
        DocSave = "../../SuperSonic/logs/model_save"

        # Delete dir
        if os.path.exists(DocSave):
            shutil.rmtree(DocSave)

        exist_policy = self.all_policy[action]

        print(f"exist_policy:{exist_policy}")

        start_engine.Stoke(exist_policy).main()

        # load loss from json

        for relpath, dirs, files in os.walk(DocSave):
            for _ in files:
                if _.split(".")[-1] == "json":
                    full_path = os.path.join(relpath, _)
        with open(full_path, "r") as f:
            data = json.load(f)
        reward = data['checkpoints'][0]['last_result']['episode_reward_max']
        # data = [json.loads(line) for line in open(full_path, "r")]
        # reward_mean = data[-1]["episode_reward_mean"]
        # reward_min = data[-1]["episode_reward_max"]
        # reward_max = data[-1]["episode_reward_min"]
        # total_loss = data[0]['info']['learner']['default_policy']['total_loss']
        # print("reward is finished!!!!!!!!!!!!!!!!!!!!")
        # result['policy'].append(action)
        # result['reward_mean'].append(reward_mean)
        # result['reward_min'].append(reward_min)
        # result['reward_max'].append(reward_max)
        # result['total_loss'].append(total_loss)

        conn = sqlite3.connect('../../SuperSonic/SQL/supersonic.db')
        c = conn.cursor()

        sql = "INSERT INTO SUPERSONIC (ID,TASK,ACTION,REWARD) \
                                                              VALUES (?, ?, ?, ?)"
        c.execute(sql, (self.num, "STOKE", action, reward))
        # c.execute("INSERT INTO HALIDE (RESULT,ACTIONS,REWARD,LOG) \
        #                               VALUES (code, self.actions, reward, self.min_exec_time_sec )")

        conn.commit()

        conn.close()

        return reward

    def get_reward_1(self, action):

        # the path to save execution performance
        DocSave = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        name = "result.json"

        # Delete dir
        if os.path.exists(DocSave):
            shutil.rmtree(DocSave)

        exist_policy = self.all_policy[action]

        self.num = self.num + 1
        print("self.num : ", self.num)
        import random
        if action == 0:
            reward = 0
        else:
            reward = random.randint(1, 10)

        # load loss from json

        print("reward is finished!!!!!!!!!!!!!!!!!!!!")
        # result['policy'].append(action)
        # result['reward_mean'].append(reward_mean)
        # result['reward_min'].append(reward_min)
        # result['reward_max'].append(reward_max)
        # result['total_loss'].append(total_loss)

        conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
        c = conn.cursor()

        sql = "INSERT INTO SUPERSONIC (ID,TASK,ACTION,REWARD) VALUES (?, ?, ?, ?)"
        c.execute(sql, (self.num, "HALIDE", action, reward))
        # c.execute("INSERT INTO HALIDE (RESULT,ACTIONS,REWARD,LOG) \
        #                               VALUES (code, self.actions, reward, self.min_exec_time_sec )")

        conn.commit()
        conn.close()

        # cursor = c.execute("SELECT ID,TASK,ACTION,REWARD,LOG  from SUPERSONIC")
        # for row in cursor:
        #     print("TASK = ", row[0])
        #     print("ACTION = ", row[1])
        #     print("REWARD = ", row[2])
        #     print("LOG = ", row[3], "\n")
        # conn.close()
        # print("reward",reward)
        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        self.num = self.num + 1
        self.actions.append(action)
        # reward = self.get_reward_1(action)
        reward = self.generate_reward(action)
        obs = self.get_observation(action)
        # print(obs)
        # print("action", action)

        done = True

        # if np.random.uniform() < self.p_dist[action]:
        #     if not isinstance(self.r_dist[action], list):
        #         reward = self.r_dist[action]
        #     else:
        #         reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        return [0], reward, done, self.info  #

    def reset(self):
        return np.random.rand(20)  #

    def render(self, mode='human', close=False):
        pass

    def close(self):
        return None

