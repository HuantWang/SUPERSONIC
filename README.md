# **Compiler phase ordering optimisation**

This script eanbles the use of RL to tune the LLVM compiler phase order for code resize reduction or runtime optimisation. 

To set the optimisation target, choose one of the target of the [reward_function](https://github.com/HuantWang/SUPERSONIC/blob/541f408b821ffe9e60954cb093e30141d1bd7337/SuperSonic/policy_search/supersonic_main.py#L182) to select the approach to evaluate your code.

```python
RewardFunctions=[
                # "codesize", "ic" # For code size reduction
                "runtime" # For run time
            ],
```

## Installation

The system was tested on the following operating systems:

- Ubuntu 18.04

Supersonic builds upon [CompilerGym v0.2.3](https://github.com/facebookresearch/CompilerGym).

First, install CompilerGym required toolchain using:

```sh
sudo apt install -y clang-9 clang++-9 clang-format golang libjpeg-dev \
  libtinfo5 m4 make patch zlib1g-dev tar bzip2 wget libgl1-mesa-glx
mkdir -pv ~/.local/bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O ~/.local/bin/bazel
wget https://github.com/hadolint/hadolint/releases/download/v1.19.0/hadolint-Linux-x86_64 -O ~/.local/bin/hadolint
chmod +x ~/.local/bin/bazel ~/.local/bin/hadolint
go get github.com/bazelbuild/buildtools/buildifier
GO111MODULE=on go get github.com/uber/prototool/cmd/prototool@dev
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```

Then, we recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
to manage the remaining build dependencies. First create a conda environment
with the required dependencies:

```shell
(base) $ conda create -y -n supersonic python=3.8
(base) $ conda activate supersonic
(supersonic) $ conda install -y -c conda-forge cmake doxygen pandoc patchelf
(supersonic) $ cd SUPERSONIC
(supersonic) $ pip install -r requirements.txt
```

## Client RL search and testing usage

This demo shows how to apply the saved client RL to optimize a test program for Code Size Reduction.

```shell
(supersonic) $ export PYTHONPATH=$PYTHONPATH:`pwd`
(supersonic) $ export PYTHONPATH=$PYTHONPATH:`pwd`/third_party
#Client RL search
(supersonic) $ python SuperSonic/policy_search/supersonic_main.py --mode policy --total_steps 10 2>/dev/null
#Client RL Parameter Tuning
(supersonic) $ python SuperSonic/policy_search/supersonic_main.py --mode config --iterations 10 2>/dev/null
#Client RL Deployment
(supersonic) $ python SuperSonic/policy_search/supersonic_main.py --datapath "benchmark://cbench-v1/lame" --mode deploy --training_iterations 50 2>/dev/null
```


Note: Setup a symlink if you encounter the following issue:
FileNotFoundError: [Errno 2] No such file or directory: 'clang++'.
```shell
ln -s /usr/bin/clang-9 /usr/bin/clang
ln -s /usr/bin/clang++-9 /usr/bin/clang++
```

## Contributing

We welcome contributions to SuperSonic. If you are interested in contributing please see
[this document](https://github.com/HuantWang/SUPERSONIC/blob/master/CONTRIBUTING.md).

## Citation

If you use Supersonic in any of your work, please cite [our paper](https://dl.acm.org/doi/10.1145/3497776.3517769):

```
@inproceedings{10.1145/3497776.3517769,
author = {Huanting Wang, Zhanyong Tang, Cheng Zhang, Jiaqi Zhao, Chris Cummins, Hugh Leather, and Zheng Wang},
title = {Automating Reinforcement Learning Architecture Design for Code Optimization},
year = {2022},
isbn = {9781450391832},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3497776.3517769},
doi = {10.1145/3497776.3517769},
booktitle = {Proceedings of the 31st ACM SIGPLAN International Conference on Compiler Construction},
pages = {129–143},
numpages = {15},
keywords = {Compiler optimization, reinforcement learning, code optimization},
location = {Seoul, South Korea},
series = {CC 2022}
}
```
