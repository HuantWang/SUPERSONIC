from copy import deepcopy
import numpy as np
from gym.spaces import Discrete, Dict, Box

# add for grpc
import gym
import os


class halide_rl:
    """A :class:
            Halide is a domain-specific language and compiler for image processing pipelines (or graphs) with multiple computation stages.
            This task builds upon Halide version 10. Our evaluation uses ten Halide applications that have been heavily tested on the Halide
            compiler. We measure the execution time of each benchmark when processing three image datasets provided by the Halide benchmark suite.
            The benchmarks are optimized to run on a multi-core CPU.

        Source:
            This environment corresponds to the version of the Halide. (https://github.com/halide/Halide)
            paper link: https://people.csail.mit.edu/jrk/halide-pldi13.pdf

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

    def init_from_env_config(self, env_config):
        """ Defines the reinforcement leaning environment. Initialise with an environment.

                    :param env_config: including  "state_function", "action_function", "reward_function", "observation_space"
                """
        self.inference_mode = env_config.get("inference_mode", False)
        if self.inference_mode:
            self.improvements = []

    def __init__(self, env_config):
        """ Defines the reinforcement leaning environment. Initialise with an environment.

                    :param env_config: including  "state_function", "action_function", "reward_function", "observation_space"
                """
        self.init_from_env_config(env_config)
        # for grpc
        # channel = grpc.insecure_channel(env_config.get("target"))
        # self.stub = schedule_pb2_grpc.ScheduleServiceStub(channel)

        log_path = env_config.get("log_path")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # transform stub
        self.env = gym.make(
            "Halide-v0",
            algorithm_id=env_config.get("algorithm_id"),
            input_image=env_config.get("input_image"),
            max_stage_directive=env_config.get("max_stage_directive"),
            log_path=env_config.get("log_path"),
            state_function=env_config.get("state_function"),
            action_function=env_config.get("action_function"),
            reward_function=env_config.get("reward_function"),
        )

        self.action_space = Discrete(self.env.action_space.n)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": Box(low=0, high=1, shape=(self.env.action_space.n,)),
            }
        )
        self.running_reward = 0
        self.num = 0

        # print("zczczczczc")
        # self.doc2vecmodel = Doc2Vec.load(env_config.get("embedding"))

    def reset(self):
        """ reset the RL environment.
                        """
        self.running_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.array([1] * self.env.action_space.n),
        }

    def step(self, action):
        """Take a step.

            :param action: An action, or a sequence of actions. When multiple
                    actions are provided the observation and reward are returned after
                    running all of the actions.

            :return: A tuple of observation, observation_mask, score, done, and info.
        """
        # self.num = self.num+1
        # print ("self.num :",self.num)
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        # obs=np.random.rand(128)
        # print(obs)
        # done = True
        score = self.running_reward if done else 0
        return (
            {"obs": obs, "action_mask": np.array([1] * self.env.action_space.n)},
            score,
            done,
            info,
        )

    def set_state(self, state):
        """ Set policy to specific state and action mask.

        :param state: Current reward and environments
        :return
        """
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        print("self.env.unwrapped.state", self.env.unwrapped.state)
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1] * self.env.action_space.n)}

    def get_state(self):
        """Returns actor state.

        :return: current environment and reward
        """
        return deepcopy(self.env), self.running_reward
