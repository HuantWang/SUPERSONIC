
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/HuantWang/SUPERSONIC/graphs/commit-activity)
[![License CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://github.com/HuantWang/SUPERSONIC/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/supersonic/badge/?version=latest)](https://supersonic.readthedocs.io/en/latest/?badge=latest)

<div align="center">
 <img src="docs/source/_static/img/logo.png" width="617px" height="148px">
</div>

<p align="center">
  <i>Automating reinforcement learning architecture design for code optimization.</i>
</p>
<p align="center">
  <i>
    Check
    <a href="https://supersonic.readthedocs.io/en/latest/index.html">the website</a>
    for more information.
  </i>
</p>

## Introduction
SUPERSONIC is a tool to automate reinforcement learning (RL) policy searching and tuning. It is designed to help compiler developers to find the right RL architecture and algorithm to use for an optimization task. It finds the right RL exploration algorithm, the reward function and a method for modeling the environment state. 

Given an RL search space defined by the Supersonic Python API, the Supersonic meta-optimizer automatically searches for a suitable RL component composition for an optimization task. It also automatically tunes a set of tunable hyperparameters of the chosen components. No RL expertise is needed to use Supersonic. 

*Check [our paper]() for detailed information.*

## Installation

SUPERSONIC builds upon [CompilerGym](https://github.com/facebookresearch/CompilerGym) and [Ray](https://docs.ray.io/en/latest/rllib.html), which require Python >= 3.6. The binary works on Linux (on Ubuntu 18.04, Fedora 28, Debian 10 or newer equivalents).
See [INSTALL.md](INSTALL.md) for further details.

## Resource

See [artifact_evaluation](https://github.com/NWU-NISL-Optimization/SuperSonic/tree/AE/artifact_evaluation) for the supporting artifact of the paper.


## Contributing

We welcome contributions to SuperSonic. If you are interested in contributing please see
[this document](https://github.com/HuantWang/SUPERSONIC/blob/master/CONTRIBUTING.md).

