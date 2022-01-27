import numpy as np
from gym import logger, spaces
from gym.spaces import Box, Dict, Discrete


class action_functions:
    def init_actions(
        self, interleave_action_length, obsv_low, obsv_high, obsv_size, method
    ):
        self.action_space = spaces.Discrete(interleave_action_length)
        self.observation_space = spaces.Box(
            low=obsv_low, high=obsv_high, shape=(obsv_size,), dtype=np.int32
        )
        # print(self.action_space)
        # print(self.observation_space)
        return self.action_space, self.observation_space

        # if method == "transform":
        #     interleave_action_meaning = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        #     action_space = spaces.Discrete(len(interleave_action_meaning))
        #     observation_space = spaces.Box(low=-1, high=1, shape=(128,), dtype=np.float64)
        #     return interleave_action_meaning, action_space, observation_space
        #
        # if method == "map":
        #     interleave_action_meaning = [_ for _ in range(2000)] #
        #     action_space = spaces.Discrete(len(interleave_action_meaning))  # action_len
        #     observation_space = spaces.Box(low=-1,high=1,shape=(128,))
        #     return interleave_action_meaning, action_space, observation_space
        #
        # # TODO: all possible schedule
        # if method == "action_space":
        #     interleave_action_meaning = [0, 1, 2, 3, 4]
        #     action_space = spaces.Discrete(len(interleave_action_meaning))
        #     observation_space = spaces.Box(low=-1, high=1, shape=(128,), dtype=np.float64)
        #     return interleave_action_meaning, action_space, observation_space
