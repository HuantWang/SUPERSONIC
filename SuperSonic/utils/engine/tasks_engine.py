"""Example of using training on Halide."""

import argparse
import sys

import compiler_gym
import gym
import ray
from ray import tune
from ray.tune import register_env
import third_packages.models.custom_torch_models
# import SuperSonic.bin.utils.interface.environments.halide_env
import SuperSonic.bin.utils.environments.halide_env
import SuperSonic.bin.utils.environments.stoke_rl_env
# import SuperSonic.bin.utils.environments.CSR_mcts_ppo
import SuperSonic.bin.utils.environments.CSR_mcts
from ray.rllib.models.catalog import ModelCatalog
from pathlib import Path
import subprocess
import sqlite3
import os
from compiler_gym.mdp_search.Algorithm import *
import re
import time

class Halide():
    def __init__(self,policy):
        conn = sqlite3.connect('supersonic.db')
        c = conn.cursor()
        # result,action history,reward,execution outputs
        try:
            c.execute('''CREATE TABLE HALIDE
                           (
                           TIME         FLOAT       NOT NULL,
                           RESULT        TEXT      NOT NULL,
                           ACTIONS       TEXT    NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           LOG FLOAT );''')
            
    
            print("Table created successfully")
        except:
            pass

        conn.commit()
        conn.close()
        # print("dbl success")
        print("halide halide halide")

        self.algorithm_id = 11
        self.input_image = 'alfaromeo_gray_64x.png'
        self.max_stage_directive = 8
        self.target = "localhost:50051"
        self.log_path = "/home/huanting/CG/MTL_test/tasks/src/opt_test/MCTS/result"
        self.halide_path = "/home/halide/grpc-halide/"
        self.environment_path = SuperSonic.bin.utils.interface.environments.halide_env.halide_rl
        self.state_function=policy["StatList"]
        self.action_function=policy["ActList"]
        self.reward_function=policy["RewList"]
        self.algorithm=policy["AlgList"]
        self.local_dir = "/home/SuperSonic/tasks/src/opt_test/MCTS/examples/model_save_Halide"
        stopper = {"training_iteration": 5}
        self.task_config = {'algorithm_id': self.algorithm_id,
                            'input_image': self.input_image,
                            'max_stage_directive': self.max_stage_directive,
                            'target': self.target,
                            'log_path': self.log_path,
                            'stop': stopper,
                            "state_function": self.state_function,
                            "action_function": self.action_function,
                            "reward_function": self.reward_function,
                            "algorithm": self.algorithm,
                            'local_dir':self.local_dir,
                            }
        # self.environment_path = "tasks.src.opt_test.MCTS.environments.halide_env.HalideEnv_PPO"

    def startserve(self):
        # print(f"{halide_path}")
        self.child = subprocess.Popen(f"cd {self.halide_path} && ./grpc-halide", shell=True, )
        # print("Child Finished")
        print(f"id = {self.algorithm_id},input_image = {self.input_image}")

    def sql(self):
        conn = sqlite3.connect('halide.db')
        print ("Opened database successfully")

    def run(self):
        RLAlgorithms().Algorithms(self.algorithm,self.task_config, self.environment_path)
    def main(self):
        Halide.sql(self)
        Halide.startserve(self)
        Halide.run(self)

#add by zc

from ray.tune import Stopper

class CustomStopper(Stopper):
    def __init__(self,obs_file):
        self.obs_file = obs_file
        print("enter stop")
        self.should_stop = False
        self._start = time.time()
        self._deadline = 80
        

    def __call__(self, trial_id, result):
        #if not self.should_stop and time.time() - self._start > self.deadline:
        if not self.should_stop and os.path.exists(self.obs_file):
            os.remove(self.obs_file)
            with os.popen(f'netstat -nutlp | grep  "50055"') as r:
                result = r.read()
            PID = []
            for line in result.split("\n"):
                if r"/" in line:
                    PID.extend(re.findall(r".*?(\d+)\/", line))
            PID = list(set(PID))
            for pid in PID:
                try:
                    os.system(f"kill -9 {pid}")
                except Exception as e:
                    print(e)

            self.should_stop = True

        return self.should_stop

    def stop_all(self):
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop

class Stoke():
    def __init__(self,policy):
        #database
        conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
        c = conn.cursor()
        # result,action history,reward,execution outputs
        try:
            c.execute('''CREATE TABLE STOKE
                           (
                           TIME         FLOAT       NOT NULL,
                           RESULT        TEXT    NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           PRIMARY KEY ('TIME'));''')

            print("Table created successfully")
        except:
            pass

        conn.commit()
        conn.close()
        # print("dbl success")

        #init parameter

        self.RLAlgo = None
        self.target = "localhost:50055"
        self.log_path = "/home/huanting/SuperSonic/tasks/stoke/result"
        self.obs_file = "/home/huanting/SuperSonic/tasks/stoke/example/record/finish.txt"
        self.stoke_path = "/home/huanting/SuperSonic/tasks/stoke/example/p04"
        self.environment_path = SuperSonic.bin.utils.environments.stoke_rl_env.stoke_rl
        self.state_function=policy["StatList"]
        self.action_function=policy["ActList"]
        self.reward_function=policy["RewList"]
        self.algorithm=policy["AlgList"]
        self.experiment = "stoke"
        self.local_dir = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        #   "/home/SuperSonic/tasks/src/opt_test/MCTS/examples/model_save_Stoke"
        #stopper = CustomStopper(self.obs_file)
        self.deadline = 50
        stopper = {"time_total_s": self.deadline}
        self.task_config = {'target': self.target,
                            'log_path': self.log_path,
                            'obs_file':self.obs_file,
                            'stop':stopper,
                            'stoke_path':self.stoke_path,
                            "state_function": self.state_function,
                            "action_function": self.action_function,
                            "reward_function": self.reward_function,
                            "algorithm": self.algorithm,
                            "experiment":self.experiment,
                            'local_dir':self.local_dir
                            }
        # self.environment_path = "tasks.src.opt_test.MCTS.environments.halide_env.HalideEnv_PPO"

    def startclient(self):
        os.system(f"rm {self.obs_file}") #clean the observation file
        self.RLAlgo = RLAlgorithms(self.task_config)
        # print(f"{halide_path}")
        print(f"{self.obs_file}")
        self.child = subprocess.Popen(
            f"cd {self.stoke_path} && python run_synch.py {self.stoke_path} {self.obs_file}",
            shell=True,
        )

        print("Child Finished")
        #print(f"id = {self.algorithm_id},input_image = {self.input_image}")

    def sql(self):
        conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
        print ("Opened database successfully")

    def run(self):
        if os.path.exists(self.obs_file):
            os.system(f"rm {self.obs_file}")
        #self.RLAlgo.Algorithms(self.algorithm,self.task_config, self.environment_path)
        try:
            RLAlgorithms().Algorithms(self.algorithm,self.task_config, self.environment_path)
        except Exception as e :
            print(e)
            print("finish stoke")

    def main(self):
        Stoke.sql(self)
        #Stoke.startserve(self)
        Stoke.run(self)

# add by zc for csr
class TimeStopper(Stopper):
     def __init__(self,deadline):
         self._start = time.time()
         self._deadline = deadline   #set time
     def __call__(self, trial_id, result):
         return False
     def stop_all(self):
         return time.time() - self._start > self._deadline

class CSR():
    def __init__(self,policy):
        #database
        print(policy)
        
        conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
        c = conn.cursor()
        # result,action history,reward,execution outputs
        try:
            c.execute('''CREATE TABLE CSR
                           (
                           TIME          FLOAT       NOT NULL,
                           BENCHMARK     TEXT  NOT NULL,
                           RESULT        TEXT  NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           PRIMARY KEY ('TIME'));''')
            print("Table created successfully")
        except:
            pass

        conn.commit()
        conn.close()
        # print("dbl success")
        
        #init parameter

        #

        #
        # '''
        # import os
        # rootpath = os.path.abspath('../../../')  # 获取上级路径
        # '''
        #
        self.benchmark = "/home/huanting/SuperSonic/tasks/CSR/DATA/mandel-text.bc"
        self.seed = "0xCC"
        self.log_path = "/home/huanting/SuperSonic/tasks/CSR/result"
        self.pass_path = "/home/huanting/SuperSonic/tasks/CSR/pass"
        self.deadline = 20
        self.environment_path = SuperSonic.bin.utils.environments.CSR_mcts.csr_rl
        self.state_function=policy["StatList"]
        self.action_function=policy["ActList"]
        self.reward_function=policy["RewList"]
        self.algorithm=policy["AlgList"]
        self.experiment = "csr"
        self.local_dir = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        #stopper = TimeStopper(self.deadline)
        stopper = {"time_total_s": self.deadline}
        self.task_config = {'benchmark': self.benchmark,
                            'seed': self.seed,
                            'log_path':self.log_path,
                            'pass_path':self.pass_path,
                            'deadline':self.deadline,
                            'stop':stopper,
                            "state_function": self.state_function,
                            "action_function": self.action_function,
                            "reward_function": self.reward_function,
                            "algorithm": self.algorithm,
                            "experiment":self.experiment,
                            'local_dir':self.local_dir
                            }
        # self.environment_path = "tasks.src.opt_test.MCTS.environments.halide_env.HalideEnv_PPO"

    def startclient(self):
        pass      
        

    def sql(self):
        conn = sqlite3.connect('/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db')
        print ("Opened database successfully")

    def run(self):
        
        RLAlgorithms().Algorithms(self.algorithm,self.task_config, self.environment_path)
        # try:
        #     RLAlgorithms().Algorithms(self.algorithm,self.task_config, self.environment_path)
        # except :
        #     print("finish csr")
    def main(self):
        #CSR.sql(self)
        #CSR.startserve(self)
        CSR.run(self)

# if __name__ == "__main__":
#     '''
#     #halide
#
#     policy = {
#         "StatList": "Actionhistory",
#         "ActList": "map",
#         "RewList": "weight",
#         "AlgList": "PPO",
#     }
#     Halide(policy).main()
#     #Halide = Halide(policy)
#     #Halide.sql()
#     #Halide.startserve()
#     #Halide.run()
#    '''
#     #stoke
#
#     print("start stoke")
#     policy = {
#         "StatList": "Doc2vec",
#          "ActList": "Doc2vec",
#          "RewList": "weight",
#          "AlgList": "PPO",
#     }
#     #         Stoke=Stoke(policy)
#     #         #Stoke.sql()
#     #         #Stoke.startclient()
#     Stoke(policy).main()
#
#     #csr
#
#     # print("start CSR")
#     # policy = {
#     #     "StatList": "Doc2vec",
#     #     "ActList": "Doc2vec",
#     #     "RewList": "weight",
#     #     "AlgList": "DQN",
#     # }
#     # CSR(policy).main()
#     # CSR(policy).main()
#     # CSR.sql()
#     # CSR.startclient()
#     # CSR.run()

    
