import numpy as np
from gym import spaces


class action_functions:
    # TODO: This class should be extended in the future.
    """:class:
    action_functions, defines the action space, ovservation space
    by inheriting a default Action class.
    """

    def init_actions(
        self, interleave_action_length, obsv_low, obsv_high, obsv_size, method
    ):
        """Construct and initialize action and observation space of different tasks.

        :param interleave_action_length: Action space. This must be defined for single-agent envs.
        :param obsv_low: lower boundary of observation space.
        :param obsv_high: higher boundary of observation space.
        :param obsv_size: Observation space. This must be defined for single-agent envs.
        :param method: Action methods, different parameters mapping to different definition approaches.

        """
        self.action_space = spaces.Discrete(interleave_action_length)
        self.observation_space = spaces.Box(
            low=obsv_low, high=obsv_high, shape=(obsv_size,), dtype=np.int32
        )
        return self.action_space, self.observation_space
