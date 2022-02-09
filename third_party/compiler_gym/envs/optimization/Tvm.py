# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines environments for STOKE"""
import math
import random
import gym
from gym.utils import seeding
from SuperSonic.policy_definition.action import action_functions
from SuperSonic.policy_definition.observation import observation_function
from SuperSonic.policy_definition.reward import reward_function
import numpy as np
import sqlite3
import time


class TvmEnv(gym.Env):
    """A :class:
            AUTOTVM is a learning-based framework to optimize tensor programs for deep learning workloads. Our evaluation uses six AUTOTVM applications that have been heavily tested on the TVM compiler.

        Source:
            This environment corresponds to the version of the Halide. (https://github.com/apache/tvm/tree/main/python/tvm/autotvm)
            paper link: https://arxiv.org/pdf/1805.08166.pdf

        Observation:
            Type: Box(100)
            Pipeline’s schedule will be convert to vectors by different embedding approaches,
            e.g. Word2vec, Doc2vec, CodeBert ...

        Actions:
            Type: Discrete(4)
            Num      Action      Description
            0        adding      Adds an optimization to the stage.
            1        removing    Removes an optimization to the stage.
            2        decreasing  Decreases the value (by one) of an enabled parameterized option.
            3        increasing  Increases the value (by one) of an enabled parameterized option.


        Reward:
            In all cases, lower cost is better. We measure the execution time of each benchmark when processing three image
            datasets provided by the Halide benchmark suite.

        Starting State:
            All observations are assigned a uniform random value in [-1..1]

    """


    def __init__(
        self, state_function, action_function, reward_function
    ):
        """ Defines the reinforcement leaning environment. Modify to match different task shape.

        :param state_function:  a state function that can summarize the program after each action as a
                                finite feature vector.
        :param action_function: an action function that can discrete set of actions or transformations that can be applied
                                to a program, such as passes in a compiler
        :param reward_function: a reward function that reports the quality of the actions taken so far.
        """

        self.state_function = state_function
        self.action_function = action_function
        self.reward_function = reward_function

        # TODO
        self.interleave_action_length, self.obsv_size = 4000, 100
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

    def get_reward(self,reward,reward_func):
        """ Calculate reward with method "reward_function".

            :param action: What will agent do in next step.
            :param reward: Describe how the agent "ought" to behave.
            :return: return a reward score after calculating.
        """
        reward = reward_function().get_rew(input=reward,reward_function=reward_function
        )
        return reward
        # return 3.1415
        # return random.random()


    def get_obs(self, action_history, state_code):
        """ feedback the observation with method "observation_function".

            :param action_history: This is the action history in past few steps.
            :param state_code: The X86 code after executing specific action.
            :return: return an embedding vector.
        """
        self.state_code = state_code
        self.input = [self.state_code]
        observation = observation_function().get_observation(
            self.input, self.obsv_size, self.state_function
        )
        return observation

    def step(self, action, state_code, reward):
        """Take a step.

            :param action: An action, or a sequence of actions. When multiple
                    actions are provided the observation and reward are returned after
                    running all of the actions.
            :param reward: A reward, to describe how the agent "ought" to behave.
            :param state_code: The X86 code after executing specific action.
            :return: A tuple of observation, reward, done, and info. Observation and
                    reward are None if default observation/reward is not set. If done is
                    True, observation and reward may also be None (e.g. because the
                    service failed).
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        obs = self.get_obs(action, state_code)  # update observation
        self.state = obs
        done = True
        rew = self.get_reward(reward, self.reward_function)  # update reward
        info = {}
        return self.state, rew, done, info

    def reset(self):
        """ reset the RL environment.
        """
        self.state = np.random.random(100)
        return self.state

    def render(self, mode="human"):
        return None

    def close(self):
        return None
