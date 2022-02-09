# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines environments for STOKE"""
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
    """A :class:
        STOKE is a stochastic optimizer and program synthesizer for the x86-64 instruction set.
        This classical compiler optimization task finds a valid code sequence to maximize the performance
        of a loop-free sequence of instructions. Superoptimizaiton is an expensive
        optimization technique as the number of possible configurations grows exponentially as
        the instruction count to be optimized increases.

    Source:
        This environment corresponds to the version of the STOKE
        described by stanfordPL. (https://github.com/StanfordPL/stoke)
        paper link: https://raw.githubusercontent.com/StanfordPL/stoke/develop/docs/papers/asplos13.pdf

    Observation:
        Type: Box(100)
        Optimized code will be convert to vectors by different embedding approaches,
        e.g. Word2vec, Doc2vec, CodeBert ...

    Actions:
        Type: Discrete(9)
        NUm      Action      Description
        0        add_nops	 Adds one extra nop instruction into the rewrite.
        1        delete	     Deletes one instruction at random.
        2        instruction Replaces an instruction with another one chosen at random.
        3        opcode	     Replaces an instruction's opcode with a new one that takes operands of the same type.
        4        operand	 Replaces an operand of one instruction with another.
        5        rotate	     Formerly "resize". Moves an instruction from one basic block to another, and shifts all the instructions in between.
        6        local_swap	 Takes two instructions in the same basic block and swaps them.
        7        global_swap Takes two instructions in the entire program and swaps them.
        8        weighted	 Selects from among several other transforms at random.

    Reward:
        In all cases, lower cost is better. We combine the value of correctness with other values we want to optimize for.
        Name	    Description
        binsize	    The size (in bytes) of the assembled rewrite using the x64asm library.
        correctness	How "correct" the rewrite's output appears. Very configurable.
        size	    The number of instructions in the assembled rewrite.
        latency	    A poor-man's estimate of the rewrite latency, in clock cycles, based on the per-opcode latency table in src/cost/tables.
        measured	An estimate of running time by counting the number of instructions actually executed on the testcases. Good for loops and algorithmic improvements.
        sseavx	    Returns '1' if both avx and sse instructions are used (this is usually bad!), and '0' otherwise. Often used with a multiplier like correctness + 1000*sseavx
        nongoal	    Returns '1' if the code (after minimization) is found to be equivalent to one in --non_goal. Can also be used with a multiplier.

    Starting State:
        All observations are assigned a uniform random value in [-1..1]

    """


    def __init__(
        self, state_function, action_function, reward_function
    ):
        """ Defines the reinforcement leaning environment. Initialise with an environment.

        :param state_function:  a state function that can summarize the program after each action as a
                                finite feature vector.
        :param action_function: an action function that can discrete set of actions or transformations that can be applied
                                to a program, such as passes in a compiler
        :param reward_function: a reward function that reports the quality of the actions taken so far.
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

    def get_reward(self,reward,reward_func):
        """ Calculate reward with method "reward_function".

            :param action: What will agent do in next step.
            :param reward: Describe how the agent "ought" to behave.
            :return: return a reward score after calculating.
        """
        reward = reward_function().get_rew(reward,reward_function
        )
        return reward

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
        rew = self.get_reward(reward,self.reward_function)  # update reward
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


# if __name__ == '__main__':
#     env = StokeEnv("Doc2vec","transform","hamming")
#     env.reset()
#     env.step(env.action_space.sample())
#     print(env.state)
#     env.step(env.action_space.sample())
#     print(env.state)
