# Installation

SUPERSONIC was tested in Python 3.8, CompilerGym 0.1.8, ray 0.8.6, Ubuntu 18.04.

## Docker（FIX!)

Install docker engine by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

1. Fetch the docker image from docker hub.

```
$ sudo docker pull CGO/SuperSonic:latest
```

Alternatively, you can build the docker image from scratch by the following instruction.

```
$ ./build_docker.sh
```

To check the list of images, run:

```
$ sudo docker images
REPOSITORY                TAG                 IMAGE ID            CREATED             SIZE
CGO/SuperSonic		     latest              d5bc1be66342        2 hours ago         14.2GB
```

1. Run the docker image.

```
$ sudo docker run -it CGO/SuperSonic /bin/bash
$ export PS1="(docker) $PS1"
```

This command will load and run the docker image, and `-it` option attaches you an interactive tty container.

## Building from Source (FIX!)

If you prefer, you may build from source. This requires a modern C++ toolchain
and bazel.

On debian-based linux systems, install the required toolchain using:

```sh
sudo apt install clang-9 libtinfo5 libjpeg-dev patchelf libgl1-mesa-glx
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O bazel
chmod +x bazel && mkdir -p ~/.local/bin && mv -v bazel ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage the remaining build dependencies. First create a conda environment with the required dependencies:

    conda create -n SuperSonic python=3.8 cmake pandoc
    conda activate SuperSonic

Then clone the SuperSonic source code using:

    git clone https://github.com/NWU-NISL-Optimization/SuperSonic.git
    cd SuperSonic
    make init

The `make init` target only needs to be run once on initial setup, or when
pulling remote changes to the SuperSonic repository.

Run the test suite to confirm that everything is working:

    make test

To build and install the `SuperSonic` python package, run:

    make install

**NOTE:** To use the `SuperSonic` package that is installed by `make install`
you must leave the root directory of this repository. Attempting to import
`SuperSonic` while in the root of this repository will cause import errors.

When you are finished, you can deactivate and delete the conda
environment using:

    conda deactivate
    conda env remove -n SuperSonic

## Building FOR 4 Case Studies(FIX!)

### A. *Case Study 1: Superoptimization*

- [ ] **ChengZhang**

### B. *Case Study 2: Optimizing Image Processing Pipelines*

- [ ] **ChengZhang**

### C. *Case Study 3: Neural Network Code Generation*

- [ ] **JiaqIZhao**

### D. *Case Study 4: Phase Ordering for Code Size Reduction*

The environment will be installed with CompilerGym.



**注：**测试过程中的遗漏环境修复问题：python环境扔到[这里](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/521b6f0697779133807a808c07c602fe91bf1102/compiler_gym/requirements.txt)，apt环境增加到这个页面的说明中。

