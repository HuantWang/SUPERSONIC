from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import ray
import ray.tune as tune

class ConfigSearch:
    """:class:
            SuperSonic currently supports 23 RL algorithms from RLLib, covering a wide range of established RL algorithms.

            """
    # cleanpid("50055")
    # if ray.is_initialized():
    #    ray.shutdown()

    def __init__(self):
        self.num_workers = 0
        self.training_iteration = 1
        self.ray_num_cpus = 8
        self.num_samples = 2

    def PPO(self, task_config, environment_path):
        self.lamda = 0.95
        self.kl_coeff = 0.2
        self.vf_clip_param = 10.0
        self.entropy_coeff = 0.01
        self.model = {"fcnet_hiddens": [128, 128]}
        self.local_dir = task_config.get("local_dir")

        ray.init(num_cpus=self.ray_num_cpus, ignore_reinit_error=True)
        # start search config
        sched = ASHAScheduler('time_total_s',metric="mean_accuracy", mode="max",max_t=10)
        reporter = CLIReporter()
        analysis = tune.run("PPO",
                          scheduler=sched,
                          progress_reporter=reporter,
                          stop={"training_iteration": self.training_iteration},
                          reuse_actors=True,
                          checkpoint_at_end=True,
                          num_samples= self.num_samples,
                          config={  "env": environment_path,
                                    "env_config": task_config,
                                    "num_workers": self.num_workers,
                                    "lr": tune.uniform(1.0),
                                    }
                            )
        # save result
        # analysis.results_df.to_csv("result.csv")
        # get the best config
        best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
        print("Best config is:", best_config)

        return best_config

    def Algorithms(self, policy_algorithm, task_config, environment_path):
        """
        Algorithms, using to call different RL algorithms

        :param policy_algorithm:
        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        if policy_algorithm == "MCTS":
            ConfigSearch().MCTS(task_config, environment_path)
        if policy_algorithm == "PPO":
            best_config=ConfigSearch().PPO(task_config, environment_path)
        if policy_algorithm == "DQN":
            ConfigSearch().DQN(task_config, environment_path)
        if policy_algorithm == "QLearning":
            ConfigSearch().QLearning(task_config, environment_path)
        if policy_algorithm == "APPO":
            ConfigSearch().APPO(task_config, environment_path)
        if policy_algorithm == "A2C":
            ConfigSearch().A2C(task_config, environment_path)
        if policy_algorithm == "A3C":
            ConfigSearch().A3C(task_config, environment_path)
        if policy_algorithm == "ARS":
            ConfigSearch().ARS(task_config, environment_path)
        if policy_algorithm == "BC":
            ConfigSearch().BC(task_config, environment_path)
        if policy_algorithm == "ES":
            ConfigSearch().ES(task_config, environment_path)
        if policy_algorithm == "MARWIL":
            ConfigSearch().MARWIL(task_config, environment_path)
        if policy_algorithm == "PG":
            ConfigSearch().PG(task_config, environment_path)
        if policy_algorithm == "SimpleQ":
            ConfigSearch().SimpleQ(task_config, environment_path)

        return best_config