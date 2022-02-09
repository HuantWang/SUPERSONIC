import ray
from ray.rllib.models.catalog import ModelCatalog
from ray import tune
from ray.tune.logger import TBXLogger, CSVLogger, JsonLogger
import subprocess
import third_party.contrib.alpha_zero.models.custom_torch_models
from ray.rllib import _register_all
_register_all()

class RLAlgorithms:
    """:class:
            SuperSonic currently supports 23 RL algorithms from RLLib, covering a wide range of established RL algorithms.

            """
    # cleanpid("50055")
    # if ray.is_initialized():
    #    ray.shutdown()

    def __init__(self):
        """Construct and initialize the RL algorithm parameters.
                """
        self.num_workers = 0
        #self.training_iteration = 50
        self.training_iteration = 20
        self.ray_num_cpus = 10
        self.rollout_fragment_length = 50
        self.train_batch_size = 50
        self.sgd_minibatch_size = 50
        self.lr = 1e-4
        self.num_sgd_iter = 50

        # ray.init(num_cpus=self.ray_num_cpus, ignore_reinit_error=True)

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

        tune.run(
            "contrib/MCTS",
            checkpoint_freq=1,
            stop=task_config.get("stop"),
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
                "train_batch_size": self.train_batch_size,
                "sgd_minibatch_size": self.sgd_minibatch_size,
                "lr": self.lr,
                "num_sgd_iter": self.num_sgd_iter,
                "mcts_config": self.mcts_config,
                "ranked_rewards": self.ranked_rewards,
                "model": self.model,
            },
            loggers=[TBXLogger],
        )

        ray.shutdown(exiting_interpreter=False)
        """
        Construct and initialize a CompilerGym service environment.
        
        """

    def PPO(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """
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

        tune.run(
            "PPO",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lambda": self.lamda,
                "kl_coeff": self.kl_coeff,
                "vf_clip_param": self.vf_clip_param,
                "entropy_coeff": self.entropy_coeff,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "sgd_minibatch_size": self.sgd_minibatch_size,
                "num_sgd_iter": self.num_sgd_iter,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
                "model": self.model,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def APPO(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "APPO",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "num_sgd_iter": self.num_sgd_iter,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def A2C(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "A2C",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def DQN(self, task_config, environment_path):
        """
         DQN, An interface to start RL agent with DQN algorithm.
         A deep learning model to successfully learn control policies directly from high-dimensional sensory
         input using reinforcement learning. The model is a convolutional neural network, trained with a
         variant of Q-learning, whose input is raw pixels and whose output is a value function estimating future rewards.

        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        self.model = {"fcnet_hiddens": [128, 128]}
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        tune.run(
            "DQN",
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "num_atoms": 1,
                "v_min": -10.0,
                "v_max": 10.0,
                "noisy": False,
                "sigma0": 0.5,
                "dueling": True,
                "hiddens": [256],
                "double_q": True,
                "n_step": 1,
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "final_prioritized_replay_beta": 0.4,
                "prioritized_replay_beta_annealing_timesteps": 200,
                "prioritized_replay_eps": 1e-6,
                "before_learn_on_batch": None,
                "training_intensity": None,
                "worker_side_prioritization": False,
                "lr": self.lr,
                "train_batch_size": 10,
                "num_workers": self.num_workers,
                "rollout_fragment_length": 10,
                "model": self.model,
                "timesteps_per_iteration": 10,
                "learning_starts": 10,
                "normalize_actions": False,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

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
        tune.run(
            "SAC",
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "train_batch_size": self.train_batch_size,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
                "timesteps_per_iteration": 1,
                "learning_starts": 1,
                "normalize_actions": False,
                # "model": self.model,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def A3C(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "A3C",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                # "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def ARS(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "ARS",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def ES(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "ES",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "num_workers": 1,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def MARWIL(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "MARWIL",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                # "lr": self.lr,
                # "train_batch_size": self.train_batch_size,
                # "num_workers": self.num_workers,
                # "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def PG(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "PG",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "train_batch_size": self.train_batch_size,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def SimpleQ(self, task_config, environment_path):
        """
                 PPO, An interface to start RL agent with PPO algorithm.
                 PPO’s clipped objective supports multiple SGD passes over the same batch of experiences.
                 Paper （https://arxiv.org/abs/1707.06347）

                :param task_config: The task_config, parameters passed to RL agent.
                :param environment_path: The environment_path, tasks' environment path that RL agent called.

                """

        if task_config.get("experiment") == "stoke":
            self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
            )
        self.local_dir = task_config.get("local_dir")
        tune.run(
            "SimpleQ",  # 内置算法PPO
            checkpoint_freq=1,
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            config={
                "env": environment_path,
                "env_config": task_config,
                "lr": self.lr,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
            },
            loggers=[TBXLogger],
        )
        ray.shutdown(exiting_interpreter=False)

    def Algorithms(self, policy_algorithm, task_config, environment_path):
        """
        Algorithms, using to call different RL algorithms

        :param policy_algorithm:
        :param task_config: The task_config, parameters passed to RL agent.
        :param environment_path: The environment_path, tasks' environment path that RL agent called.

        """
        if policy_algorithm == "MCTS":
            RLAlgorithms().MCTS(task_config, environment_path)
        if policy_algorithm == "PPO":
            RLAlgorithms().PPO(task_config, environment_path)
        if policy_algorithm == "DQN":
            RLAlgorithms().DQN(task_config, environment_path)
        if policy_algorithm == "QLearning":
            RLAlgorithms().QLearning(task_config, environment_path)
        if policy_algorithm == "APPO":
            RLAlgorithms().APPO(task_config, environment_path)
        if policy_algorithm == "A2C":
            RLAlgorithms().A2C(task_config, environment_path)
        if policy_algorithm == "A3C":
            RLAlgorithms().A3C(task_config, environment_path)
        if policy_algorithm == "ARS":
            RLAlgorithms().ARS(task_config, environment_path)
        if policy_algorithm == "ES":
            RLAlgorithms().ES(task_config, environment_path)
        if policy_algorithm == "MARWIL":
            RLAlgorithms().MARWIL(task_config, environment_path)
        if policy_algorithm == "PG":
            RLAlgorithms().PG(task_config, environment_path)
        if policy_algorithm == "SimpleQ":
            RLAlgorithms().SimpleQ(task_config, environment_path)
