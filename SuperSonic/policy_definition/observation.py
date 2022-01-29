import numpy as np
from gensim.models.doc2vec import Doc2Vec

class observation_function:
    """:class:
     The SuperSonic RL components include pre-trained observation functions, such as
    Word2Vec and Doc2Vec. The state-transition function can be selected from a
    further set of predefined transition functions, such as a transition probability matrix or LSTM
     """
    def __init__(self):
        """Construct and initialize state-transition method of different tasks."""
        self.obs_fun = "Doc2vec"

    def get_observation(self, input_obs, obsv_size, obs_fun):
        """Get observation with specific trasition functions

        :param input_obs: Input, usually as input of an embedding model, e.g. code slice or action history.
        :param obsv_size: Observation space.
        :param obs_fun: Observation functions, state-transition method.
        """
        self.input = input_obs[0]
        self.obs_fun = obs_fun
        self.obsv_size = obsv_size

        if self.obs_fun == "Doc2vec":
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0 : self.obsv_size]

        if self.obs_fun == "Word2vec":
            self.input = input_obs[0]
            # TODO: preprocess before embedding
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0 : self.obsv_size]

        if self.obs_fun == "Bert":
            self.input = input_obs[0]
            # TODO: preprocess before embedding
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0 : self.obsv_size]

        if self.obs_fun == "Actionhistory":
            # TODO: STOKE cant do that, without a action record array
            self.input = input_obs[0]
            if len(self.input) < self.obsv_size:
                observation = np.hstack(
                    (np.array(self.input), np.zeros(self.obsv_size - len(self.input)))
                )[: self.obsv_size]
            else:
                observation = self.input[-self.obsv_size :]

        return observation
