from gym.envs.registration import make, register, registry, spec

from compiler_gym.envs.optimization.Stoke import StokeEnv
from compiler_gym.envs.optimization.Halide import HalideEnv
from compiler_gym.envs.optimization.Tvm import TvmEnv
from compiler_gym.envs.optimization.Bandit_Halide import BanditHalideEnv
from compiler_gym.envs.optimization.Bandit_CSR import BanditCSREnv
from compiler_gym.envs.optimization.Bandit_Tvm import BanditTvmEnv
from compiler_gym.envs.optimization.Bandit_Stoke import BanditStokeEnv

register(
    id="Stoke-v0",
    entry_point="compiler_gym.envs.optimization:StokeEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="Tvm-v0",
    entry_point="compiler_gym.envs.optimization:TvmEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="Halide-v0",
    entry_point="compiler_gym.envs.optimization:HalideEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditHalideEnv-v0",
    entry_point="compiler_gym.envs.optimization:BanditHalideEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditCSREnv-v0",
    entry_point="compiler_gym.envs.optimization:BanditCSREnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditTvmEnv-v0",
    entry_point="compiler_gym.envs.optimization:BanditTvmEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)

register(
    id="BanditStokeEnv-v0",
    entry_point="compiler_gym.envs.optimization:BanditStokeEnv",
    # max_episode_steps=200,
    # reward_threshold=25.0,
)
