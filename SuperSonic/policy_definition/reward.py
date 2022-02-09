import numpy as np

class reward_function:
    """:class:
    A reward function that reports the quality of the actions taken so far.
    It provides candidate reward functions like RelativeMeasure and tanh to compute
    the reward based on the metric given by the measurement interface.
    """
    def __init__(self):
        """Construct and initialize reward-transition method of different tasks."""
        self.rew_fun = "tanh"

    def get_rew(self, input, baseline=1, weight=1, reward_function="usr_define"):
        """Get reward with specific reward functions

                :param input: Input, usually as input of an transition function, e.g. runtime, speedup and hamming distance.
                :param baseline: Using baseline to calculate speedup etc.
                :param weight: Using weight parameter to set how important of specific action.
                :param reward_function: reward functions, reward-transition method.
                """
        global reward
        self.baseline = (
            baseline
        )
        self.current = input
        if reward_function == "usr_define":
            reward = self.current

        if reward_function == "relative_measure":
            reward = self.current / self.baseline

        if reward_function == "tan":
            reward = np.tan(self.current)

        if reward_function == "func":
            if self.current < self.baseline:
                reward = 0
            else:
                reward = 1


        if reward_function == "weight":
            if self.current < self.baseline:
                exec_diff = self.baseline - self.current
                self.current = self.current
                reward = exec_diff * weight
            else:
                reward = 0.0

    
        return reward

