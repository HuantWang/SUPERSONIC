from gym.envs.registration import make, register, registry, spec
from SuperSonic.utils.environments.Bandit_CSR import BanditCSREnv
import gym

# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if "BanditCSREnv-v0" in env:
#          print("Remove {} from registry".format(env))
#          del gym.envs.registration.registry.env_specs[env]
try:
    register(
        id="BanditCSREnv-v0",
        entry_point="SuperSonic.utils.environments.Bandit_CSR:BanditCSREnv",
    )
except:
    pass
