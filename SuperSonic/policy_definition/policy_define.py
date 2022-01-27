from itertools import product
import os
import numpy as np
import sklearn.model_selection

# from compiler_gym.mdp_search.observation import observation_function
# from compiler_gym.mdp_search.reward import reward_function
# from compiler_gym.mdp_search.Algorithm import RLAlgorithms
# from compiler_gym.mdp_search.action import action_functions


class SuperOptimizer:
    def __init__(
        self,
        StateFunctions=["Word2vec", "Doc2vec", "Bert"],
        RewardFunctions=["relative_measure", "tan", "func", "weight"],
        RLAlgorithms=["MCTS", "PPO", "DQN", "QLearning"],
        ActionFunctions=["init"],
        datapath="",
    ):
        self.StateFunctions = StateFunctions
        self.RewardFunctions = RewardFunctions
        self.RLAlgorithms = RLAlgorithms
        self.ActionFunctions = ActionFunctions
        self.datapath = datapath

    def PolicyDefined(self):
        global policy
        self.policy = {
            "StatList": self.StateFunctions,
            "ActList": self.ActionFunctions,
            "RewList": self.RewardFunctions,
            "AlgList": self.RLAlgorithms,
        }

        self.policy_all = [
            dict(zip(self.policy, v)) for v in product(*self.policy.values())
        ]
        self.policy_amount = len(self.policy_all) - 1
        return self.policy_all, self.policy_amount

    def cross_valid(self,):
        data_list = []

        for root, dirs, files in os.walk(self.datapath):
            if files == []:
                for i in dirs:
                    i = root + "/" + i
                    data_list.append(i)
            else:
                for i in files:
                    i = root + "/" + i
                    data_list.append(i)
            break

        data_list = np.array(data_list)
        kfolder = sklearn.model_selection.KFold(
            n_splits=3, shuffle=False, random_state=None
        )
        for index, (train, test) in enumerate(kfolder.split(data_list)):
            TrainDataset = data_list[train]
            TestDataset = data_list[test]
            Dataset = [TrainDataset, TestDataset]
            return Dataset


# SuperOptimizer().cross_valid()
