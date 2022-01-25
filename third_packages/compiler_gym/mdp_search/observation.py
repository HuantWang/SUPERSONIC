import numpy as np
from gensim.models.doc2vec import Doc2Vec

class observation_function():

    def __init__(self):
        # self.input = np.random.random(128)
        self.obs_fun="Doc2vec"

    def get_observation(self,input_obs,obsv_size,obs_fun):
        self.input = input_obs[0]
        self.obs_fun= obs_fun
        self.obsv_size= obsv_size

        if self.obs_fun == "Doc2vec":
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0:self.obsv_size]
            # print("aaaaaaaaaaaaaaaa",len(observation))

        if self.obs_fun == "Word2vec":
            self.input = input_obs[0]
            # TODO: preprocess before embedding
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0:self.obsv_size]

        if self.obs_fun == "Bert":
            self.input = input_obs[0]
            # TODO: preprocess before embedding
            path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            observation = np.array(
                Doc2Vec.load(path)
                .infer_vector(self.input.split(), steps=6, alpha=0.025)
                .tolist()
            )[0:self.obsv_size]

        if self.obs_fun == "Actionhistory":
            self.input = input_obs[0]
            # print("self.input",self.input)
            # print(np.zeros(5))
            # TODO: preprocess before embedding
            # path = "/home/huanting/CG/opt_test/mdp_search/AutoMDP/doc2vec.model.newdata_p23_1106_b.pkl"
            if len(self.input)<self.obsv_size:
                observation=np.hstack((np.array(self.input),np.zeros(self.obsv_size-len(self.input))))[:self.obsv_size]
                # print("observation",observation)
            else:
                observation = self.input[-self.obsv_size:]



        return observation



    # def obs_init(method):
    #     if method == "Doc2vec":
    #         obs_init = np.random.random(100)
    #
    #     if method == "Word2vec":
    #         obs_init = np.random.random(100)
    #
    #     if method == "Inst2vec":
    #         obs_init = np.random.random(100)
    #
    #     if method == "Graph2vec":
    #         obs_init = np.random.random(100)
    #
    #     if method == "History":
    #         obs_init = np.random.random(100)
    #     return obs_init
