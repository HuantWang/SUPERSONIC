import numpy as np

class reward_function:
    def __init__(self):
        self.rew_fun = "tanh"

    def get_reward(self, input, baseline=1, weight=1, reward_function="usr_define"):
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
            return reward, self.baseline

        if reward_function == "weight":
            if self.current < self.baseline:
                exec_diff = self.baseline - self.current
                self.current = self.current
                reward = exec_diff * weight
            else:
                reward = 0.0
            return reward, self.baseline

        else:
            return reward, self.baseline
