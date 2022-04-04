import numpy as np
from sklearn.decomposition import PCA

# from compiler_gym.views import ObservationSpaceSpec


class observation_function:
    """:class:
     The SuperSonic RL components include pre-trained observation functions, such as
    Word2Vec and Doc2Vec. The state-transition function can be selected from a
    further set of predefined transition functions, such as a transition probability matrix or LSTM
    """

    def __init__(self, env, state_fun="Autophase"):
        self.state_fun = state_fun
        self.env = env
        self.length = self.env.observation_space.shape[0]

    def get_observation(self):
        """Get observation with specific trasition functions

        :param input_obs: Input, usually as input of an embedding model, e.g. code slice or action history.
        :param obsv_size: Observation space.
        :param obs_fun: Observation functions, state-transition method.
        """
        global obs
        try:
            if self.state_fun == "Inst2vec":
                model_pca = PCA(n_components=1)
                X_pca = model_pca.fit(self.env.observation["Inst2vec"]).transform(
                    self.env.observation["Inst2vec"]
                )
                obs = X_pca[0 : self.length]
                return obs
            if self.state_fun == "InstCount":
                obs = self.env.observation["InstCount"][0 : self.length]
                return obs
            if self.state_fun == "InstCountNorm":
                obs = self.env.observation["InstCountNorm"][0 : self.length]
                return obs
            if self.state_fun == "Autophase":
                obs = self.env.observation["Autophase"][0 : self.length]
                return obs
            if self.state_fun == "Inst2vecEmbeddingIndices":
                obs = self.env.observation["Inst2vecEmbeddingIndices"][0 : self.length]
                return obs
        except:
            try:
                # The length is shorter than standard length
                obs_add = np.zeros(self.length)
                for i, j in zip(range(len(obs)), obs):
                    obs_add[i] = j
                return obs_add
            except:
                obs = self.env.observation["Autophase"][0 : self.length]
                return obs
