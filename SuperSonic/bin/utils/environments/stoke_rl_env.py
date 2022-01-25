from copy import deepcopy
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
import compiler_gym
from gensim.models.doc2vec import Doc2Vec

# add for grpc
import grpc
from SuperSonic.bin.service import schedule_pb2
from SuperSonic.bin.service import schedule_pb2_grpc
import time
from concurrent import futures

_ONE_DAY_IN_SECONDS = 60 * 60 * 24  # set timeout
from multiprocessing import Lock
import threading

#add for new code
from compiler_gym.mdp_search.action import action_functions
import sqlite3
import time
# add mutex
lock = threading.Lock()
lock_s = threading.Lock()

# global variable for update action,reward,observation
state_code = ""
Action = 2
state_reward = 1000.0


class ScheduleServicer(schedule_pb2_grpc.ScheduleServiceServicer):
    def GetStokeMsg(self, request, context):
        lock_s.acquire()
        global state_code

        global state_reward
        state_code = request.code
        state_reward = request.cost

        if lock.locked():
            lock.release()
            #print("lock release")
        return schedule_pb2.MsgStokeResponse(action=Action)


class stoke_rl:
    def __init__(self, env_config):
        
        self.env = gym.make("Stoke-v0", 
            state_function=env_config.get("state_function"),
            action_function=env_config.get("action_function"),
            reward_function=env_config.get("reward_function"),
            )
        # self.interleave_action_length,self.obsv_size = 9,100
        # self.obsv_low = 0
        # self.obsv_high = 1
        # self.action_function = env_config.get("action_function")
        # self.action_space, self.observation_space = action_functions().init_actions(self.interleave_action_length,
        #                                                                             self.obsv_low,self.obsv_high,self.obsv_size,self.action_function)
        self.action_space = Discrete(9)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.running_reward = 0
        #self.doc2vecmodel = Doc2Vec.load(env_config.get("embedding"))
        self.tstart = time.time()
        # grpc connect
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        schedule_pb2_grpc.add_ScheduleServiceServicer_to_server(ScheduleServicer(), self.server)
        self.server.add_insecure_port(env_config.get("target"))
        self.server.start()

    def reset(self):
        self.running_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        }

    def step(self, action):
        lock.acquire()
        global Action
        Action = action
        obs, rew, done, info = self.env.step(action, state_code, state_reward)
        self.running_reward += rew
        score = self.running_reward if done else 0
        # print(action)
        if lock_s.locked():
            lock_s.release()
           # print("lock_s release")
        try:
            conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
            c = conn.cursor()
            sql = "INSERT INTO STOKE (TIME, RESULT, REWARD) \
                            VALUES (?, ?, ?)"
            c.execute(sql,(time.time() - self.tstart,state_code.replace("nop\n",""), rew))

            conn.commit()
            conn.close()
        except Exception as e:
            print(e)

        return (
            {"obs": obs, "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])},
            score,
            done,
            info,
        )

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
