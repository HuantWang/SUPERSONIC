import ray
from ray.rllib.contrib.alpha_zero.models import custom_torch_models
from ray.rllib.models.catalog import ModelCatalog
import gym
from ray import tune
from ray.tune import register_env
from ray.tune.logger import TBXLogger, CSVLogger, JsonLogger
import third_packages.models.custom_torch_models
import SuperSonic.utils.environments.halide_env
# import tasks.src.opt_test.MCTS.environments.halide_env
from compiler_gym.util.clean import cleanpid


def RunStoke(task_config):
    import subprocess
    #print(task_config.get('stoke_path'))
    self.child = subprocess.Popen(
        f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
        shell=True,
    )

class RLAlgorithms():
    # cleanpid("50055")
    #if ray.is_initialized():
    #    ray.shutdown()

    def __init__(self):
        self.local_dir = "/home/huanting/SuperSonic/tasks/stoke/logs"
        self.num_workers = 0
        self.training_iteration = 5
        self.ray_num_cpus = 4
        self.rollout_fragment_length= 20
        self.train_batch_size= 50
        self.sgd_minibatch_size= 50
        self.lr= 1e-4
        self.num_sgd_iter= 55

        # ray.init(num_cpus=self.ray_num_cpus, ignore_reinit_error=True)
        
       

    '''
    def RunStoke(task_config):
        import subprocess
            #print(task_config.get('stoke_path'))
        self.child = subprocess.Popen(
            f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
            shell=True,
        )
    '''

    def MCTS(self, task_config, environment_path):

        self.mcts_config = {
            "puct_coefficient": 1.5,
            "num_simulations": 5,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        }

        self.ranked_rewards= {
            "enable": True,
        }
        
        self.model= {
            "custom_model": "dense_model",}
        ModelCatalog.register_custom_model("dense_model", third_packages.models.custom_torch_models.DenseModel)
        print(f"init {task_config}")
        self.local_dir = task_config.get("local_dir")

        import time 
        
        if task_config.get("experiment") == "stoke":
            import subprocess
            print(task_config.get('stoke_path'))
            try:
                self.child = subprocess.Popen(
                f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
                shell=True,
                )

            except:
                print("subprocess error")

        # sudo python run_synch.py "/home/huanting/SuperSonic/tasks/stoke/example/p04" "/home/huanting/SuperSonic/tasks/stoke/example/record/finish.txt"
        tune.run(
            "contrib/MCTS",
            checkpoint_freq=1,
            stop = task_config.get('stop'),
            #stop={"time_total_s": 80},
            #stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,

            config={
                #"env": tasks.src.opt_test.MCTS.environments.stoke_rl_env.stoke_rl,
                "env":environment_path,
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
            loggers=[TBXLogger]
        )
        
        ray.shutdown(exiting_interpreter=False)
        
    def PPO(self, task_config, environment_path):
        self.lamda = 0.95
        self.kl_coeff = 0.2
        self.vf_clip_param = 10.0
        self.entropy_coeff = 0.01
        self.model= {'fcnet_hiddens': [128, 128]}
        
        print(environment_path)
          
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            import subprocess
            self.child = subprocess.Popen(
            f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
            shell=True,
            )


        tune.run(
            "PPO",  # 内置算法PPO
            checkpoint_freq=1,
            # name="neurovectorizer_train",  # 实验名称
            stop = task_config.get('stop'),
            #stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            # scheduler=MedianStoppingRule(grace_period=10.0),
            config={
                'env':environment_path,
                #"env": tasks.src.opt_test.MCTS.environments.halide_env.halide_rl,
                "env_config": task_config,
                'lambda': self.lamda,
                'kl_coeff': self.kl_coeff,
                'vf_clip_param': self.vf_clip_param,
                'entropy_coeff': self.entropy_coeff,

                # 'clip_rewards': False,
                # 'num_envs_per_worker': 1,
                # 'batch_mode': 'truncate_episodes',
                # 'observation_filter': 'NoFilter',
                # 'vf_share_layers': 'true',

                'lr': self.lr,
                'train_batch_size': self.train_batch_size,
                'sgd_minibatch_size': self.sgd_minibatch_size,
                'num_sgd_iter': self.num_sgd_iter,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
                "model": self.model,
                "normalize_actions": False
                
            },  # 用于生成调优变量的特定算法配置
            loggers=[TBXLogger]
        )
        ray.shutdown(exiting_interpreter=False)

    def DQN(self, task_config, environment_path):
        #TODO: I cant limite the timesteps in one iteration, now its 1000. I dont know how to reduce it
        # self.lamda = 0.95
        # self.kl_coeff = 0.2
        # self.vf_clip_param = 10.0
        # self.entropy_coeff = 0.01
        self.model = {'fcnet_hiddens': [128, 128]}
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            import subprocess
            self.child = subprocess.Popen(
            f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
            shell=True,
            )
        tune.run(
            "DQN",
            checkpoint_freq=1,
            # name="neurovectorizer_train",  # 实验名称
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            # scheduler=MedianStoppingRule(grace_period=10.0),
            config={
                'env':environment_path,
                #"env": tasks.src.opt_test.MCTS.environments.halide_env.halide_rl,
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
                # "num_envs_per_worker" : 10,
                'lr': self.lr,
                "train_batch_size": 10,
                "num_workers": self.num_workers,
                "rollout_fragment_length": 10,
                # "training_intensity" : 2,
                "model": self.model,
                "timesteps_per_iteration":10,
                "learning_starts": 10,
                "normalize_actions": False

            },  # 用于生成调优变量的特定算法配置
            loggers=[TBXLogger]
        )
        ray.shutdown(exiting_interpreter=False)

    def QLearning(self, task_config, environment_path):
        self.lamda = 0.95
        self.kl_coeff = 0.2
        self.vf_clip_param = 10.0
        self.entropy_coeff = 0.01
        self.model = {'fcnet_hiddens': [128, 128]}
        self.local_dir = task_config.get("local_dir")
        if task_config.get("experiment") == "stoke":
            import subprocess
            self.child = subprocess.Popen(
            f"cd {task_config.get('stoke_path')} && python run_synch.py {task_config.get('stoke_path')} {task_config.get('obs_file')}",
            shell=True,
            )
        tune.run(
            "SAC",  
            checkpoint_freq=1,
            # name="neurovectorizer_train",  # 实验名称
            stop={"training_iteration": self.training_iteration},
            max_failures=0,
            reuse_actors=True,
            checkpoint_at_end=True,
            local_dir=self.local_dir,
            # scheduler=MedianStoppingRule(grace_period=10.0),
            config={
                'env':environment_path,
                #"env": tasks.src.opt_test.MCTS.environments.halide_env.halide_rl,
                "env_config": task_config,
                # 'lambda': self.lamda,
                # 'kl_coeff': self.kl_coeff,
                # 'vf_clip_param': self.vf_clip_param,
                # 'entropy_coeff': self.entropy_coeff,

                # 'clip_rewards': False,
                # 'num_envs_per_worker': 1,
                # 'batch_mode': 'truncate_episodes',
                # 'observation_filter': 'NoFilter',
                # 'vf_share_layers': 'true',
                # You should override this to point to an offline dataset.
                
                # 'lr': self.lr,
                'train_batch_size': self.train_batch_size,
                # 'sgd_minibatch_size': self.sgd_minibatch_size,
                # 'num_sgd_iter': self.num_sgd_iter,
                "num_workers": self.num_workers,
                "rollout_fragment_length": self.rollout_fragment_length,
                "timesteps_per_iteration": 1,
                "learning_starts": 1,
                "normalize_actions": False
                # "model": self.model,

            },  # 用于生成调优变量的特定算法配置
            loggers=[TBXLogger]
        )
        ray.shutdown(exiting_interpreter=False)
        
    def Algorithms(self,policy_algorithm,task_config,environment_path):
        if policy_algorithm == "MCTS":
            RLAlgorithms().MCTS(task_config, environment_path)
        if policy_algorithm == "PPO":
            RLAlgorithms().PPO(task_config, environment_path)
        if policy_algorithm == "DQN":
            RLAlgorithms().DQN(task_config, environment_path)
        if policy_algorithm == "QLearning":
            RLAlgorithms().QLearning(task_config, environment_path)
