import gym
import numpy as np
import grpc
import threading
import sqlite3
import time
from gym.spaces import Discrete, Dict, Box
from SuperSonic.service import schedule_pb2
from SuperSonic.service import schedule_pb2_grpc
from concurrent import futures
from copy import deepcopy
_ONE_DAY_IN_SECONDS = 60 * 60 * 24  # set timeout
# add mutex
lock = threading.Lock()
lock_s = threading.Lock()

# global variable for update action,reward,observation
state_code = ""
Action = 2
state_reward = 1000.0

class ScheduleServicer(schedule_pb2_grpc.ScheduleServiceServicer):

    def GetStokeMsg(self, request, context):
        global state_code
        global state_reward
        lock_s.acquire()
        state_code = request.code
        state_reward = request.cost

        if lock.locked():
            lock.release()
        return schedule_pb2.MsgStokeResponse(action=Action)

class stoke_rl:
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

    def __init__(self, env_config):
        """ Defines the reinforcement leaning environment. Initialise with an environment.

            :param env_config: including  "state_function", "action_function", "reward_function", "observation_space"
        """

        self.env = gym.make(
            "Stoke-v0",
            state_function=env_config.get("state_function"),
            action_function=env_config.get("action_function"),
            reward_function=env_config.get("reward_function"),
        )
        self.sql_path = env_config.get("sql_path")
        self.action_space = Discrete(9)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.running_reward = 0
        self.tstart = time.time()
        # grpc connect
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        schedule_pb2_grpc.add_ScheduleServiceServicer_to_server(
            ScheduleServicer(), self.server
        )
        self.server.add_insecure_port(env_config.get("target"))
        self.server.start()

    def reset(self):
        """ reset the RL environment.
                """
        self.running_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        }

    def step(self, action):
        """Take a step.

                    :param action: An action, or a sequence of actions. When multiple
                            actions are provided the observation and reward are returned after
                            running all of the actions.

                    :return: A tuple of observation, observation_mask, score, done, and info.
                """

        lock.acquire()
        global Action
        Action = action
        # print("action",action)
        obs, rew, done, info = self.env.step(action, state_code, state_reward)

        self.running_reward += rew
        score = self.running_reward if done else 0
        if lock_s.locked():
            lock_s.release()
        try:
            conn = sqlite3.connect(
                self.sql_path
            )
            c = conn.cursor()
            sql = "INSERT INTO STOKE (TIME, RESULT, REWARD) \
                            VALUES (?, ?, ?)"
            c.execute(
                sql, (time.time(), state_code.replace("nop\n", ""), rew)
            )
            conn.commit()
            conn.close()

        except Exception as e:
            print(e)

        return (
            {"obs": obs, "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])},
            score,
            done,
            info,
        )

    def set_state(self, state):
        """ Set policy to specific state and action mask.

        :param state: Current reward and environments
        :return: state and action mask
        """
        self.env = deepcopy(state[0])
        self.running_reward = state[1]
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}

    def get_state(self):
        """Returns actor state.

        :return: current environment and reward
        """
        return deepcopy(self.env), self.running_reward