import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
from compiler_gym.envs.llvm.llvm_env import make_benchmark
import time
import sqlite3


class csr_rl:
    """A :class:
            This task is concerned with determining the LLVM passes and their order to minimize the code size.
            Following the setup of CompilerGym, we compute the code size reduction by measuring the ratio of
            LLVM IR instruction count reduction with respect to the LLVM -Oz code size optimization option.
            This metric is platform-independent and deterministic.

        Source:
            This environment is following the setup of CompilerGym(https://github.com/facebookresearch/CompilerGym).


        Observation:
            Type: Box(0, 9223372036854775807, (56,), int64)
            The Autophase observation space is a 56-dimension integer feature vector summarizing the static LLVM-IR representation.
            It is described in 'AutoPhase: Juggling HLS phase orderings in random forests with deep reinforcement learning'.
            paper link: https://proceedings.mlsys.org/paper/2020/file/4e732ced3463d06de0ca9a15b6153677-Paper.pdf


        Actions:
            Type: Discrete(123)
            Num      Action                         Description
            0        -add-discriminators            Add DWARF path discriminators.
            1        -adce                          Aggressive Dead Code Elimination.
            2        -aggressive-instcombine        Combine pattern based expressions.
            3        -alignment-from-assumptions    Replaces an instruction's opcode with a new one that takes operands of the same type.
            4        -always-inline                 Inliner for always_inline functions.
            ... ...

            120      -strip                         Strip all symbols from a module.
            121      -tailcallelim                  Tail Call Elimination.
            122      -mergereturn                   Unify function exit nodes.


        Reward:
            In all cases, lower code size is better. We compute the code size reduction by measuring the ratio of
            LLVM IR instruction count reduction with respect to the LLVM -Oz code size optimization option.

        Starting State:
            All observations are assigned a uniform random value in [-1..1]

        """


    def __init__(self, env_config):
        """ Defines the reinforcement leaning environment. Initialise with an environment.

                    :param env_config: including  "state_function", "action_function", "reward_function", "observation_space"
                """
        # self.init_from_env_config(env_config)
        self.benchmarks = env_config.get("benchmark")
        self.log_path = env_config.get("log_path")
        self.pass_path = env_config.get("pass_path")
        self.deadline = env_config.get("deadline")
        # print(self.benchmarks)
        self.env = gym.make(
            "llvm-autophase-ic-v0",  # selects the compiler to use
            benchmark=make_benchmark(
                [self.benchmarks]
            ),  # selects the program to compile
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
        self.seeds = env_config.get("seed")
        self.flag = False
        self.best_command = ""
        self.env.seed(env_config.get("seed"))
        self.score = 0
        # self.env.init()
        print(f"seed:{self.seeds}")

    def reset(self):
        """ reset the RL environment.
                        """
        self.running_reward = 0
        self.episode_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.ones((self.env.action_space.n,), dtype=int),
        }

    def step(self, action):
        """Take a step.

        :param action: An action, or a sequence of actions. When multiple
                actions are provided the observation and reward are returned after
                running all of the actions.

        :return: A tuple of observation, observation_mask, score, done, and info.
        """
        used_time = time.time() - self.time
        benchmarks = self.benchmarks.split("/")[-1]
        with open(f"{self.pass_path}/{benchmarks}_pass.txt", "a+") as f:
            f.write("=================Step======================\r\n")
            f.write(f"step:{self.step}\r\nactions:{self.env.actions}\r\n")
            f.write(f"command line:{self.env.commandline()}\r\n")
            f.write(
                f"time={used_time:.3},action_len={len(self.env.actions)},episode_reward={self.env.episode_reward:.3%},reward_all={self.view:.3%}\r\n"
            )
        try:

            conn = sqlite3.connect(
                "/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db"
            )
            c = conn.cursor()
            sql = "INSERT INTO CSR (TIME, BENCHMARK, RESULT, REWARD) \
                            VALUES (?, ?, ?, ?)"
            # print("time",round(time.time() - self.time))
            c.execute(
                sql,
                (
                    time.time() - self.time,
                    benchmarks,
                    self.env.commandline(),
                    self.env.episode_reward,
                ),
            )

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
                    f.write(
                        f"actions{str(self.best_actions)}\r\ncommand_line:{self.best_command}\r\n"
                    )

                    f.write(
                        f"seeds:{self.seeds}\r\nreward: " + str(self.view) + "\r\n "
                    )
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
        """ Set policy to specific state and action mask.

        :param state: Current reward and environments
        :return: state and action mask
        """
        self.running_reward = state[1]
        self.env = state[0].fork()
        obs = self.env.observation["Autophase"]
        return {
            "obs": obs,
            "action_mask": np.ones((self.env.action_space.n,), dtype=int),
        }

    def get_state(self):
        """Returns actor state.

        :return: current environment and reward
        """
        env_1 = self.env.fork()
        return env_1, self.running_reward
