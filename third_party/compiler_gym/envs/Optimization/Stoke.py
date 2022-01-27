"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym.utils import seeding
from SuperSonic.policy_definition.action import action_functions
from SuperSonic.policy_definition.observation import observation_function
from SuperSonic.policy_definition.reward import reward_function
import numpy as np
import sqlite3
import time


class StokeEnv(gym.Env):
    """
    Description:
        STOKE is a stochastic optimizer and program synthesizer for the x86-64 instruction set.

    """

    def __init__(
        self, state_function, action_function, reward_function
    ):  # state_function,action_function,reward_function
        """ Defines the reinforcement leaning environment.
        Modify to match different task shape.
        """

        self.state_function = state_function
        self.action_function = action_function
        self.reward_function = reward_function

        self.interleave_action_length, self.obsv_size = 9, 100
        self.obsv_low = 0
        self.obsv_high = 1
        self.action_space, self.observation_space = action_functions().init_actions(
            self.interleave_action_length,
            self.obsv_low,
            self.obsv_high,
            self.obsv_size,
            self.action_function,
        )

        self.state = None
        self.tstart = time.time()

    def get_reward(self, action, state_reward):
        # reward_new = get_reward("hamming", action)
        # self.reward_function = "relative_measure"
        # # reward,self.min_exec_time_sec = reward_function().get_reward\
        # #     (self.response.exec_time_sec, self.min_exec_time_sec, weight=self.reward_scale, method=self.reward_function)
        reward_new = state_reward
        return reward_new

    def get_obs(self, action, state_code):
        self.state_code = state_code
        # self.state_code = "abc"
        # observation = get_observation(self.observation_method, self.state_code)

        self.input = [self.state_code]
        observation = observation_function().get_observation(
            self.input, self.obsv_size, self.state_function
        )
        return observation

    def step(self, action, state_code, reward):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        obs = self.get_obs(action, state_code)  # update observation
        self.state = obs

        done = True
        rew = reward  # update reward
        # save to db
        # print(state_code.replace("nop\n",""))
        # print(str(self.env.actions))
        """ 
        conn = sqlite3.connect('/home/SuperSonic/tasks/src/opt_test/MCTS/examples/supersonic.db')
        c = conn.cursor()
        sql = "INSERT INTO STOKE (TIME, RESULT, REWARD) \
                            VALUES (?, ?, ?)"
        c.execute(sql,(round(time.time() - self.tstart),state_code.replace("nop\n",""), reward ))

        conn.commit()
        conn.close()
        """
        info = {}
        return self.state, rew, done, info

    def reset(self):
        # self.state = obs_init(self.observation_method)
        self.state = np.random.random(100)
        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        return None


# if __name__ == '__main__':
#     env = StokeEnv("Doc2vec","transform","hamming")
#     env.reset()
#     env.step(env.action_space.sample())
#     print(env.state)
#     env.step(env.action_space.sample())
#     print(env.state)
