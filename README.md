## SUPERSONIC
<div align="center">
![SUPERSONIC](https://github.com/HuantWang/SUPERSONIC/blob/master/docs/source/_static/img/logo.png)
</div>

SUPERSONIC is a tool to automate RL policy searching and tuning. Within SUPERSONIC, a policy architecture consists of an exploration algorithm to drive the RL agent to choose actions, a reward function for computing the expected reward based on past observations of the environment (e.g., resulting execution time of a code transformation), a method for modeling the environment state (e.g., a DNN or a linear function).

SUPERSONIC operates in the following four phases:

​	1. Task definition. To use SUPERSONIC, the user first defines the optimization task by creating a policy interface. 

​	2. Policy search. SUPERSONIC provides an API to automatically find the optimal policy architecture using training benchmarks. 

​	3. Policy tuning and servicing. The chosen policy architecture is passed to the policy tuning and servicing environment, which first tunes the hyperparameters (e.g., the learning rate of the RL algorithm) of the relevant models of a policy architecture. The tuned policy can then be inserted back into the tuner to guide the optimization for unseen programs (i.e., servicing).

​	4. Measurement engine. The measurement engine evaluates candidate options using a user-supplied measurement interface and servicing phases to obtain feedback from the environment and to observe the system state. 

*Check [our paper]() for detailed information.*

## Installation

SUPERSONIC builds upon [CompilerGym](https://github.com/facebookresearch/CompilerGym) and [Ray](https://docs.ray.io/en/latest/rllib.html), which require Python >= 3.6. The binary works on Linux (on Ubuntu 18.04, Fedora 28, Debian 10 or newer equivalents).
See [INSTALL.md](INSTALL.md) for further details.

## Resource

See [artifact_evaluation](https://github.com/NWU-NISL-Optimization/SuperSonic/tree/AE/artifact_evaluation) for the supporting artifact of the paper.


## Contributing

We welcome contributions to SuperSonic. If you are interested in contributing please see
[this document](https://github.com/HuantWang/SUPERSONIC/blob/master/CONTRIBUTING.md).

