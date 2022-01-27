from gym.envs.registration import make, register, registry, spec

from compiler_gym.envs.Optimization.Stoke import StokeEnv
from compiler_gym.envs.Optimization.Halide import HalideEnv
from compiler_gym.envs.Optimization.Tvm import  TvmEnv
from compiler_gym.envs.Optimization.Bandit_Halide import BanditHalideEnv
from compiler_gym.envs.Optimization.Bandit_CSR import BanditCSREnv
from compiler_gym.envs.Optimization.Bandit_Tvm import BanditTvmEnv
from compiler_gym.envs.Optimization.Bandit_Stoke import BanditStokeEnv

register(
    id="Stoke-v0",
    entry_point="compiler_gym.envs.Optimization:StokeEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id='Tvm-v0',
    entry_point='compiler_gym.envs.Optimization:TvmEnv',
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id='Halide-v0',
    entry_point='compiler_gym.envs.Optimization:HalideEnv',
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditHalideEnv-v0",
    entry_point="compiler_gym.envs.Optimization:BanditHalideEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditCSREnv-v0",
    entry_point="compiler_gym.envs.Optimization:BanditCSREnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditTvmEnv-v0",
    entry_point="compiler_gym.envs.Optimization:BanditTvmEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditStokeEnv-v0",
    entry_point="compiler_gym.envs.Optimization:BanditStokeEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)
