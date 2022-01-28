import math
import gym
from gym.utils import seeding
import numpy as np
from gym import spaces
from compiler_gym.mdp_search.action import get_action
from compiler_gym.mdp_search.observation import get_observation, obs_init
from compiler_gym.mdp_search.reward import get_reward
import sys

# sys.path.append('/home/huanting/CG/tasks/stoke_env/opt_test')
import opt_test.MCTS.examples.train_stoke

# import opttest.MCTS.examples.train_stoke as train_stoke
import shutil
import os
import json


class BanditEnv(gym.Env):
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

        self.all_policy, self.n_bandits = policy, len(policy)
        self.dataset = data
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.box.Box(
            -1.0, 1.0, shape=(128,), dtype=np.float64
        )  #
        # self.observation_space = spaces.Discrete(1)
        self.num = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def generate_reward(self, action):

        # the path to save execution performance
        DocSave = "/home/huanting/CG/MTL_test/opt_test/MCTS/AutoMDP/model_save"
        name = "result.json"

        # Delete dir
        if os.path.exists(DocSave):
            shutil.rmtree(DocSave)

        exist_policy = self.all_policy[action]
        # child = subprocess.Popen(
        #     f"cd {hacker_data} && python train_stoke.py {hacker_data} {observe_file}",
        #     shell=True,
        # )
        # print("Child Finished")

        opt_test.MCTS.examples.train_stoke.find_mdp(
            exist_policy,
            observe_file="/home/huanting/CG/MTL_test/opt_test/MCTS/AutoMDP/test/finish.txt",
            hacker_data="/home/huanting/CG/MTL_test/opt_test/stoke_env/stoke/stoke/example/hacker/p03",
        )

        # load loss from json
        for relpath, dirs, files in os.walk(DocSave):
            if name in files:
                full_path = os.path.join(relpath, name)
        with open(full_path, "r") as f:
            data = [json.loads(line) for line in open(full_path, "r")]
            reward_mean = data[-1]["episode_reward_mean"]
            reward_min = data[-1]["episode_reward_max"]
            reward_max = data[-1]["episode_reward_min"]
            total_loss = data[0]["info"]["learner"]["default_policy"]["total_loss"]

            # result['policy'].append(action)
            # result['reward_mean'].append(reward_mean)
            # result['reward_min'].append(reward_min)
            # result['reward_max'].append(reward_max)
            # result['total_loss'].append(total_loss)
        self.num = self.num + 1
        print("self.num : ", self.num)

        return reward_mean

    def get_reward_1(self, action):
        self.num = self.num + 1
        print("self.num : ", self.num)
        reward = 0.0
        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        reward = self.generate_reward(action)
        # reward = self.get_reward_1(action)
        obs = np.random.rand(20)
        done = True

        # if np.random.uniform() < self.p_dist[action]:
        #     if not isinstance(self.r_dist[action], list):
        #         reward = self.r_dist[action]
        #     else:
        #         reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        return [0], reward, done, self.info  #

    def reset(self):
        return np.random.rand(20)  #

    def render(self, mode="human", close=False):
        pass

    def close(self):
        return None
