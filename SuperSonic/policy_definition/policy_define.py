from itertools import product
import os
import numpy as np
import sklearn.model_selection


class SuperOptimizer:
    """:class:
    SuperOptimizer includes candidate functions (or models) for representing the environment state,
    objective functions for computing the reward, and the set of possible actions that can be taken from a given
    state.
    The compiler developer first defines the optimization problem by creating an RL policy interface.
    The definition includes a list of client RL components for the meta-optimizer to search over.
                """
    def __init__(
        self,
        StateFunctions=["Word2vec", "Doc2vec", "Bert"],
        RewardFunctions=["relative_measure", "tan", "func", "weight"],
        RLAlgorithms=["MCTS", "PPO", "DQN", "QLearning"],
        ActionFunctions=["init"],
        datapath="",
    ):
        """Construct and initialize the parameters of policy definition.

        :param StateFunctions: State functions, The SuperSonic RL components include pre-trained observation functions, such as
    Word2Vec and Doc2Vec.
        :param RewardFunctions: Reward functions, It provides candidate reward functions like RelativeMeasure and tanh to compute
    the reward based on the metric given by the measurement interface.
        :param RLAlgorithms: RL algorithms, SuperSonic currently supports 23 RL algorithms from RLLib, covering a wide range of established RL algorithms.
        :param ActionFunctions: Action functions, define a discrete set of actions or transformations that can be applied
to a program, such as passes in a compiler.
        :param datapath: The benchmarks' save path.
                        """
        self.StateFunctions = StateFunctions
        self.RewardFunctions = RewardFunctions
        self.RLAlgorithms = RLAlgorithms
        self.ActionFunctions = ActionFunctions
        self.datapath = datapath

    def PolicyDefined(self):
        """Each of the components can be chosen from
        a pool of SuperSonic built-in candidate methods, and the
        combination of these components can result in a large policy
        search space.

        :return policy_all: All policy strategies.
        :return policy_amount: A list includes index for each policy strategy.
        """
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
        """ split dataset to train/valid set, default using 3-fold cross validation """
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