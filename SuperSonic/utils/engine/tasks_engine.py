import sqlite3
import os
import re
import time
from ray.tune import Stopper
from SuperSonic.policy_definition.Algorithm import *
from SuperSonic.utils.engine.config_search import ConfigSearch
import SuperSonic
import SuperSonic.utils.environments.CSR_env


class TimeStopper(Stopper):
    """A :class: An interface for implementing a Tune experiment stopper."""

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
    """A :class: An interface for user customization implementing a Tune experiment stopper."""

    def __init__(self, obs_file):
        """Create the TimeStopper object.
        Stops the entire experiment when the time has past deadline
        :param obs_file: the shared file location.
        """
        self.obs_file = obs_file
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


class CSR:
    def __init__(self, policy, data="", training_iterations=50):
        """A :class:
        A interface to run CSR.
        To apply a tuned RL, SuperSonic creates a session to apply a standard RL loop to minimize the code size
        by using the chosen RL exploration algorithms to determines which pass to be added into or removed from
        the current compiler pass sequence.
        """
        # database
        # rootpath = os.path.abspath('../SQL/supersonic.db')
        # print(rootpath)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!")

        self.sql_path = os.path.abspath("./SuperSonic/SQL/supersonic.db")
        # self.sql_path = os.path.abspath("/home/huanting/supersonic/SUPERSONIC/SuperSonic/SQL/supersonic.db")

        conn = sqlite3.connect(self.sql_path)
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
        self.training_iterations = training_iterations
        self.deadline = 5
        self.environment_path = SuperSonic.utils.environments.CSR_env.csr_rl
        self.state_function = policy["StatList"]
        self.action_function = policy["ActList"]
        self.reward_function = policy["RewList"]
        self.algorithm = policy["AlgList"]
        self.experiment = "csr"
        self.local_dir = os.path.abspath("./SuperSonic/logs/model_save")
        self.benchmark = data
        self.seed = "0xCC"
        self.log_path = os.path.abspath("./CSR/result")
        self.pass_path = os.path.abspath("./CSR/pass")

        # stopper = TimeStopper(self.deadline)
        stopper = {"time_total_s": self.deadline}
        self.task_config = {
            "sql_path": self.sql_path,
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
            "training_iterations": self.training_iterations,
        }
        # self.environment_path = "tasks.src.opt_test.MCTS.environments.halide_env.HalideEnv_PPO"

    def startclient(self):
        pass

    def sql(self):
        """Database connection"""
        conn = sqlite3.connect("./SuperSonic/SQL/supersonic.db")
        print("Opened database successfully")

    def run(self):
        """To start RL agent with specific policy strategy and parameters"""
        RLAlgorithms(self.task_config).Algorithms(
            self.algorithm, self.task_config, self.environment_path
        )

    def Config(self, iterations):
        best_config = ConfigSearch(self.task_config).Algorithms(
            self.algorithm, self.task_config, self.environment_path, iterations
        )
        return best_config

    # def main(self):
    #     # CSR.sql(self)
    #     # CSR.startserve(self)
    #     CSR.run(self)


class TaskEngine:
    """A :class: An interface to run specific Task environment and agent."""

    def __init__(self, policy):
        """An interface to start environment and agent.

        :param policy: including "state_function", "action_function", "reward_function", "observation_space" transition
            methods.
        :param tasks_name: The task developer intend to optimize.
        """

        self.policy = policy

    def run(self, policy, task="CSR", data="", training_iterations=20):
        if task == "CSR":
            CSR(policy, data, training_iterations).run()

    def Config(
        self, policy, task="CSR", iterations=2, benchmark="", training_iterations=20
    ):
        global best_config
        if task == "CSR":
            best_config = CSR(policy, benchmark, training_iterations).Config(iterations)

        return best_config
