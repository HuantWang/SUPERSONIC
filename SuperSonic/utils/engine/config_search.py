from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule, PopulationBasedTraining
from ray.tune import CLIReporter
import ray
from ray.rllib.models.catalog import ModelCatalog
from ray import tune
import subprocess
import third_party.contrib.alpha_zero.models.custom_torch_models
from ray.rllib import _register_all
_register_all()

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
        self.ray_num_cpus = 10
        self.num_samples = 1
        self.sched = ASHAScheduler('time_total_s', metric="episode_reward_mean", mode="max", max_t=10)
        self.reporter = CLIReporter()
        import os
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''

    def PPO(self, task_config, environment_path):
        self.lamda = 0.95
        self.kl_coeff = 0.2
        self.vf_clip_param = 10.0
        self.entropy_coeff = 0.01
        self.model = {"fcnet_hiddens": [128, 128]}
        self.local_dir = task_config.get("local_dir")

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        # start search config

        analysis = tune.run("PPO",
                            scheduler=self.sched,
                            progress_reporter=self.reporter,
                            num_samples=self.num_samples,
                            stop={"training_iteration": self.training_iteration},
                            reuse_actors=True,
                            checkpoint_at_end=True,
                            config={"env": environment_path,
                                    "env_config": task_config,
                                    "num_workers": 0,
                                    "lr": tune.uniform(0.001, 1.0),
                                    "kl_coeff" : tune.uniform(0.2,0.5),
                                    }
                            )
        ray.shutdown(exiting_interpreter=False)
        # get the best config
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def MCTS(self, task_config, environment_path):
        """
        MCTS, An interface to start RL agent with MCTS algorithm.
        MCTS is an RL agent originally designed for two-player games.
        This version adapts it to handle single player games. The code can
        be sscaled to any number of workers. It also implements the ranked
        rewards (R2) strategy to enable self-play even in the one-player setting.
        The code is mainly purposed to be used for combinatorial optimization.

        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        self.mcts_config = {
            "puct_coefficient": 1.5,
            "num_simulations": 5,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        }

        self.ranked_rewards = {
            "enable": True,
        }

        self.model = {
            "custom_model": "dense_model",
        }
        ModelCatalog.register_custom_model(
            "dense_model",
            third_party.contrib.alpha_zero.models.custom_torch_models.DenseModel,
        )
        print(f"init {task_config}")
        self.local_dir = task_config.get("local_dir")

        if task_config.get("experiment") == "stoke":
            try:
                self.child = subprocess.Popen(
                    f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                    shell=True,
                )
            except:
                print("subprocess error")

        analysis=tune.run(
            "contrib/MCTS",
            stop=task_config.get("stop"),
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            config={
                "env": environment_path,
                "env_config": task_config,
                "num_workers": self.num_workers,
                "lr": tune.uniform(0.001, 1.0),
                "mcts_config": self.mcts_config,
                "ranked_rewards": self.ranked_rewards,
                "model": self.model,
            },
        )

        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def APPO(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "APPO",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def DQN(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "DQN",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def QLearning(self, task_config, environment_path):
        """
         Q-networks, An interface to start RL agent with Q-networks algorithm.
         Use two Q-networks (instead of one) for action-value estimation.
         Each Q-network will have its own target network.

        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        analysis=tune.run(
            "SAC",
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "num_workers": self.num_workers,
                "timesteps_per_iteration": 1,
                "learning_starts": tune.uniform(0.1, 1.0),
                "normalize_actions": False,
                # "model": self.model,
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")

        return best_config,best_metric_score

    def A2C(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "A2C",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def A3C(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "A3C",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def ARS(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "ARS",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def ES(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "ES",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")

        return best_config,best_metric_score

    def MARWIL(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "MARWIL",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def PG(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "PG",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def SimpleQ(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )

        analysis=tune.run(
            "SimpleQ",  # 内置算法PPO
            scheduler=self.sched,
            progress_reporter=self.reporter,
            num_samples=self.num_samples,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": tune.uniform(0.001, 1.0),
            },
        )
        ray.shutdown(exiting_interpreter=False)
        best_config,best_metric_score = analysis.get_best_config(metric="episode_reward_mean", mode="max")


        return best_config,best_metric_score

    def Algorithms(self, policy_algorithm, task_config, environment_path,iterations):
        """
        Algorithms, using to call different RL algorithms

        :param policy_algorithm:
        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        reward_list = []
        config_list = []
        for i in range(iterations):
            if policy_algorithm == "MCTS":
                best_config,reward=ConfigSearch().MCTS(task_config, environment_path)
            if policy_algorithm == "PPO":
                best_config,reward=ConfigSearch().PPO(task_config, environment_path)
            if policy_algorithm == "DQN":
                best_config,reward=ConfigSearch().DQN(task_config, environment_path)
            if policy_algorithm == "QLearning":
                best_config,reward=ConfigSearch().QLearning(task_config, environment_path)
            if policy_algorithm == "APPO":
                best_config,reward=ConfigSearch().APPO(task_config, environment_path)
            if policy_algorithm == "A2C":
                best_config,reward=ConfigSearch().A2C(task_config, environment_path)
            if policy_algorithm == "A3C":
                best_config,reward=ConfigSearch().A3C(task_config, environment_path)
            if policy_algorithm == "ARS":
                best_config,reward=ConfigSearch().ARS(task_config, environment_path)
            if policy_algorithm == "ES":
                best_config,reward=ConfigSearch().ES(task_config, environment_path)
            if policy_algorithm == "MARWIL":
                best_config,reward=ConfigSearch().MARWIL(task_config, environment_path)
            if policy_algorithm == "PG":
                best_config,reward=ConfigSearch().PG(task_config, environment_path)
            if policy_algorithm == "SimpleQ":
                best_config,reward=ConfigSearch().SimpleQ(task_config, environment_path)

            config_list.append(best_config)
            reward_list.append(reward)
        index=reward_list.index(max(reward_list))
        best_config = config_list[index]

        print("Best config is:", best_config)
        return best_config