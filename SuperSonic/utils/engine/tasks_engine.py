import SuperSonic.utils.environments.Halide_env
import SuperSonic.utils.environments.stoke_env
import SuperSonic.utils.environments.CSR_env
import sqlite3
import os
import re
import time
from ray.tune import Stopper
from SuperSonic.policy_definition.Algorithm import *
import SuperSonic.utils.environments.rltvm_env
from third_party.rm_port import kill_pid


class TimeStopper(Stopper):
    """A :class: An interface for implementing a Tune experiment stopper.

        """
    def __init__(self, deadline):
        """Create the TimeStopper object.
                Stops the entire experiment when the time has past deadline
                """
        self._start = time.time()
        self._deadline = deadline  # set time

    def __call__(self, trial_id, result):
        """Returns true if the trial should be terminated given the result."""

        return False

    def stop_all(self):
        """Returns true if the experiment should be terminated."""

        return time.time() - self._start > self._deadline

class CustomStopper(Stopper):
    """A :class: An interface for user customization implementing a Tune experiment stopper.

        """
    def __init__(self, obs_file):
        """Create the TimeStopper object.
        Stops the entire experiment when the time has past deadline
        :param obs_file: the shared file location.
        """
        self.obs_file = obs_file
        print("enter stop")
        self.should_stop = False
        self._start = time.time()
        self._deadline = 80

    def __call__(self, trial_id, result):
        """Returns true if the trial should be terminated given the result."""

        # if not self.should_stop and time.time() - self._start > self.deadline:
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

class Halide:
    """A :class:
             A interface to run Halide.
             To apply a tuned RL, SuperSonic creates a session to
             apply a standard RL loop to optimize the input program
             by using the chosen RL exploration algorithms to select an action for a given state.
        """
    def __init__(self, policy):
        """ Defines the reinforcement leaning environment. Initialise with an environment and construct a database.

                :param policy: including "state_function", "action_function", "reward_function", "observation_space" transition
                methods.
                """
        conn = sqlite3.connect("supersonic.db")
        c = conn.cursor()
        try:
            c.execute(
                """CREATE TABLE HALIDE
                           (
                           TIME         FLOAT       NOT NULL,
                           RESULT        TEXT      NOT NULL,
                           ACTIONS       TEXT    NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           LOG FLOAT );"""
            )

            print("Table created successfully")
        except:
            pass

        conn.commit()
        conn.close()
        print("halide halide halide")

        self.algorithm_id = 11
        self.input_image = "alfaromeo_gray_64x.png"
        self.max_stage_directive = 8
        self.target = "localhost:50051"
        self.log_path = "/home/huanting/CG/MTL_test/tasks/src/opt_test/MCTS/result"
        self.halide_path = "/home/halide/grpc-halide/"
        self.environment_path = (
            SuperSonic.utils.interface.environments.halide_env.halide_rl
        )
        self.state_function = policy["StatList"]
        self.action_function = policy["ActList"]
        self.reward_function = policy["RewList"]
        self.algorithm = policy["AlgList"]
        self.local_dir = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        stopper = {"training_iteration": 5}
        self.task_config = {
            "algorithm_id": self.algorithm_id,
            "input_image": self.input_image,
            "max_stage_directive": self.max_stage_directive,
            "target": self.target,
            "log_path": self.log_path,
            "stop": stopper,
            "state_function": self.state_function,
            "action_function": self.action_function,
            "reward_function": self.reward_function,
            "algorithm": self.algorithm,
            "local_dir": self.local_dir,
        }

    def startserve(self):
        """ Start server, to start environment and measurement engine (For a given schedulingtemplate,
        the measurement engine invokes the user-supplied run function to compile and execute the program
        in the target environment.)
        """
        self.child = subprocess.Popen(
            f"cd {self.halide_path} && ./grpc-halide", shell=True,
        )
        print(f"id = {self.algorithm_id},input_image = {self.input_image}")

    def sql(self):
        """ Database connection"""
        conn = sqlite3.connect("halide.db")
        print("Opened database successfully")

    def run(self):
        """ To start RL agent with specific policy strategy and parameters"""
        RLAlgorithms().Algorithms(
            self.algorithm, self.task_config, self.environment_path
        )

    def main(self):
        Halide.sql(self)
        Halide.startserve(self)
        Halide.run(self)

class Stoke:
    """A :class:
         A interface to run Stoke.
         To apply a tuned RL, SuperSonic creates a session to
         apply a standard RL loop to optimize the input program
         by using the chosen RL exploration algorithms to select an action for a given state.
    """

    def __init__(self, policy):
        """ Defines the reinforcement leaning environment. Initialise with an environment and construct a database.

        :param policy: including "state_function", "action_function", "reward_function", "observation_space" transition
        methods.
        """
        # database
        conn = sqlite3.connect("/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db")
        c = conn.cursor()
        # result,action history,reward,execution outputs
        try:
            c.execute(
                """CREATE TABLE STOKE
                           (
                           TIME         FLOAT       NOT NULL,
                           RESULT        TEXT    NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           PRIMARY KEY ('TIME'));"""
            )
            print("Table created successfully")
        except:
            pass
        conn.commit()
        conn.close()

        # init parameter

        self.RLAlgo = None
        self.target = "localhost:50055"
        self.log_path = "/home/huanting/SuperSonic/tasks/stoke/result"
        self.obs_file = (
            "/home/huanting/SuperSonic/tasks/stoke/example/record/finish.txt"
        )
        self.stoke_path = "/home/huanting/SuperSonic/tasks/stoke/example/p04"
        self.environment_path = SuperSonic.utils.environments.stoke_env.stoke_rl
        self.state_function = policy["StatList"]
        self.action_function = policy["ActList"]
        self.reward_function = policy["RewList"]
        self.algorithm = policy["AlgList"]
        self.experiment = "stoke"
        self.local_dir = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        #   "/home/SuperSonic/tasks/src/opt_test/MCTS/examples/model_save_Stoke"
        # stopper = CustomStopper(self.obs_file)
        self.deadline = 50
        stopper = {"time_total_s": self.deadline}
        self.task_config = {
            "target": self.target,
            "log_path": self.log_path,
            "obs_file": self.obs_file,
            "stop": stopper,
            "stoke_path": self.stoke_path,
            "state_function": self.state_function,
            "action_function": self.action_function,
            "reward_function": self.reward_function,
            "algorithm": self.algorithm,
            "experiment": self.experiment,
            "local_dir": self.local_dir,
        }

    def startclient(self):
        """ Start client, to start environment and measurement engine (For a given optimization option,
        the measurement engine invokes the user-supplied run function to compile and execute the program
         in the target environment.)
        """
        os.system(f"rm {self.obs_file}")  # clean the observation file
        self.RLAlgo = RLAlgorithms(self.task_config)
        print(f"{self.obs_file}")
        self.child = subprocess.Popen(
            f"cd {self.stoke_path} && python run_synch.py {self.stoke_path} {self.obs_file}",
            shell=True,
        )
        print("Child Finished")

    def sql(self):
        """ Database connection"""
        conn = sqlite3.connect("/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db")
        print("Opened database successfully")

    def run(self):
        """ To start RL agent with specific policy strategy and parameters"""
        if os.path.exists(self.obs_file):
            os.system(f"rm {self.obs_file}")
        try:
            RLAlgorithms().Algorithms(
                self.algorithm, self.task_config, self.environment_path
            )
        except Exception as e:
            print(e)
            print("finish stoke")

    def main(self):
        Stoke.sql(self)
        Stoke.run(self)

class CSR:
    def __init__(self, policy):
        """A :class:
                 A interface to run CSR.
                 To apply a tuned RL, SuperSonic creates a session to apply a standard RL loop to minimize the code size
                 by using the chosen RL exploration algorithms to determines which pass to be added into or removed from
                 the current compiler pass sequence.
                """
        # database
        conn = sqlite3.connect("/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db")
        c = conn.cursor()
        try:
            c.execute(
                """CREATE TABLE CSR
                           (
                           TIME          FLOAT       NOT NULL,
                           BENCHMARK     TEXT  NOT NULL,
                           RESULT        TEXT  NOT NULL,
                           REWARD        FLOAT  NOT NULL,
                           PRIMARY KEY ('TIME'));"""
            )
            print("Table created successfully")
        except:
            pass

        conn.commit()
        conn.close()
        # print("dbl success")

        # init parameter

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
        self.environment_path = SuperSonic.utils.environments.CSR_env.csr_rl
        self.state_function = policy["StatList"]
        self.action_function = policy["ActList"]
        self.reward_function = policy["RewList"]
        self.algorithm = policy["AlgList"]
        self.experiment = "csr"
        self.local_dir = "/home/huanting/SuperSonic/SuperSonic/logs/model_save"
        # stopper = TimeStopper(self.deadline)
        stopper = {"time_total_s": self.deadline}
        self.task_config = {
            "benchmark": self.benchmark,
            "seed": self.seed,
            "log_path": self.log_path,
            "pass_path": self.pass_path,
            "deadline": self.deadline,
            "stop": stopper,
            "state_function": self.state_function,
            "action_function": self.action_function,
            "reward_function": self.reward_function,
            "algorithm": self.algorithm,
            "experiment": self.experiment,
            "local_dir": self.local_dir,
        }
        # self.environment_path = "tasks.src.opt_test.MCTS.environments.halide_env.HalideEnv_PPO"

    def startclient(self):
        pass

    def sql(self):
        """ Database connection"""
        conn = sqlite3.connect("/home/huanting/SuperSonic/SuperSonic/SQL/supersonic.db")
        print("Opened database successfully")

    def run(self):
        """ To start RL agent with specific policy strategy and parameters"""
        RLAlgorithms().Algorithms(
            self.algorithm, self.task_config, self.environment_path
        )
        # try:
        #     RLAlgorithms().Algorithms(self.algorithm,self.task_config, self.environment_path)
        # except :
        #     print("finish csr")

    def main(self):
        # CSR.sql(self)
        # CSR.startserve(self)
        CSR.run(self)

class TaskEngine:
    """A :class: An interface to run specific Task environment and agent.

            """
    def __init__(self, policy, tasks_name="Stoke"):
        """An interface to start environment and agent.

        :param policy: including "state_function", "action_function", "reward_function", "observation_space" transition
            methods.
        :param tasks_name: The task developer intend to optimize.
            """
        self.tasks=tasks_name
        self.policy = policy

    def run(self,policy):
        if self.tasks=="Stoke":
            Stoke(policy).main()
        if self.tasks=="Halide":
            Halide(policy).main()
        if self.tasks=="CSR":
            CSR(policy).main()

def createDB(db_path = "SuperSonic/SQL/supersonic.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # result,action history,reward,execution outputs
    try:
        c.execute(
            """CREATE TABLE STOKE
                        (
                        TIME         FLOAT       NOT NULL,
                        RESULT        TEXT    NOT NULL,
                        REWARD        FLOAT  NOT NULL,
                        PRIMARY KEY ('TIME'));"""
        )

        print("Table created successfully")
    except:
        pass

    conn.commit()
    conn.close()


class Tvm:
    """A :class:
         A interface to run TVM.
         To apply a tuned RL, SuperSonic creates a session to
         apply a standard RL loop to optimize the input program
         by using the chosen RL exploration algorithms to select an action for a given state.
         attention: tvm as server in there!
    """
    def __init__(self, policy):
        # database
        createDB("SuperSonic/SQL/supersonic.db")
        # init paramete:r
        self.RLAlgo = None
        self.target = "localhost:50061"
        self.log_path = "tasks/tvm/zjq/logs"
        self.obs_file = ("tasks/tvm/zjq/record/finish.txt")
        self.tvm_path = "tasks/tvm/zjq/grpc/"
        self.environment_path = SuperSonic.utils.environments.rltvm_env.RLClient
        self.state_function = policy["StatList"]
        self.action_function = policy["ActList"]
        self.reward_function = policy["RewList"]
        self.algorithm = policy["AlgList"]
        self.experiment = "tvm"
        self.local_dir = "SuperSonic/logs/model_save"
        self.deadline = 50
        stopper = {"time_total_s": self.deadline}
        self.task_config = {
            "target": self.target,
            "log_path": self.log_path,
            "obs_file": self.obs_file,
            "stop": stopper,
            "tvm_path": self.tvm_path,
            "state_function": self.state_function,
            "action_function": self.action_function,
            "reward_function": self.reward_function,
            "algorithm": self.algorithm,
            "experiment": self.experiment,
            "local_dir": self.local_dir,
        }

    # run child process for starting server of tvm
    def startTVMServer(self):
        os.system(f"rm {self.obs_file}")  # clean the observation file
        print(self.task_config)
        print(f"{self.obs_file}")
        kill_pid("50061")
        self.child = subprocess.Popen(
            f"cd {self.tvm_path} && python schedule.server.py {self.tvm_path} {self.obs_file}",
            shell=True,
        )

        print("Child Finished")
        # print(f"id = {self.algorithm_id},input_image = {self.input_image}")

    def sql(self):
        conn = sqlite3.connect("SuperSonic/SQL/supersonic.db")
        print("Opened database successfully")

    def run(self):
        if os.path.exists(self.obs_file):
            os.system(f"rm {self.obs_file}")

        RLAlgorithms().Algorithms(
            self.algorithm, self.task_config, self.environment_path
        )

    def main(self):
        self.sql()
        self.startTVMServer()
        self.run()

def tvmMain():
    print("start tvm")
    policy = {
        "StatList": "Actionhistory",
         "ActList": "Doc2vec",
         "RewList": "weight",
         "AlgList": "DQN",
    }

    Tvm(policy).main()

if __name__ == "__main__":
    tvmMain()

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
#     print("start stoke")
#     policy = {
#         "StatList": "Actionhistory",
#          "ActList": "Doc2vec",
#          "RewList": "weight",
#          "AlgList": "DQN",
#     }
#             # Stoke=Stoke(policy)
#             #Stoke.sql()
#             #Stoke.startclient()
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
