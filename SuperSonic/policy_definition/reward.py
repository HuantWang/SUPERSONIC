import random
import numpy as np

# TODO: Supplement GRPC module
# def get_reward(method, action, hamming, hamming_last):
class reward_function:
    def __init__(self):
        self.rew_fun = "tanh"

    def get_reward(self, input, baseline=1, weight=1, reward_function="usr_define"):
        self.baseline = (
            baseline  # eg: using grpc to obtain hamming distance as baseline
        )
        self.current = input
        if reward_function == "usr_define":
            reward = self.current

        if reward_function == "relative_measure":
            # hamming = random.randint(-5, 5)  # eg: using grpc to obtain hamming distance
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

        return reward

        # if method == "cost":
        #     reward = hamming

        # if method == "tanh_weight":
        #     hamming = random.randint(-5, 5)  # eg: using grpc to obtain hamming distance
        #     highweigh_list = [
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #     ]  # eg: expertise knowledge, which is important
        #     if action in highweigh_list:
        #         reward = np.tan(hamming * 2)
        #     else:
        #         reward = np.tan(hamming * 0.5)
