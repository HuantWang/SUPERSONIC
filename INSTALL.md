# Installation

SUPERSONIC was tested with Python 3.8, CompilerGym 0.1.8, ray 0.8.6 and Ubuntu 18.04.

## DOCKER (TODO.)

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

## Building from Source

## 1.1. Dependences

### 1.1.1 Toolchain and Python

Our environment based on  [CompilerGym v0.1.8](https://github.com/facebookresearch/CompilerGym/tree/v0.1.8), please install CompilerGym at begining. Then using: 

```shell
$ sudo apt install clang-9 libtinfo5 libjpeg-dev patchelf libgl1-mesa-glx libgl1 net-tools
$ pip install -r requirements.txt
# add python environment, 
# e.g. sudo echo /home/SuperSonic >> /root/anaconda3/envs/python3.8/site-packages/SS.pth
$ sudo echo <path_to_supersonic> >> <path_to_site-packages>/SS.pth
$ sudo echo <path_to_supersonic/third_party> >> <path_to_site-packages>/SS.pth
```

### 1.1.2 Acquiring LLVM

Add environment path for `llvm-config` and `clang` (You could download llvm setup file from [here](https://releases.llvm.org/))：

```shell
$ export LLVM_ROOT=<path_to_llvm>/bin/llvm-install  
#If you don't have llvm-install in <path_to_llvm>/bin/, please use this command:
$ export LLVM_ROOT=<path_to_llvm>

$ export LLVM_CONFIG=<path_to_llvm>/bin/llvm-config
$ export CLANG=<path_to_llvm>/bin/clang
```

## 1.2. grpc

We use gRPC to connected our stoke-mcts system, so we have to build the gRPC first. The version of gRPC we used is **v1.12.0**. [Download]<https://github.com/grpc/grpc/releases/tag/v1.12.0> and build the **protoc** and **grpc** with your package.

#### ***Note: using different GCC version to install your GRPC***

|           Use Cases            | GCC Version |
| :----------------------------: | :---------: |
|   Optimizing Image Pipelines   |     7.5     |
| Neural Network Code Generation |     7.5     |
|      Code Size Reduction       |     7.5     |
|       Super-optimization       |     4.9     |

We provide a script `changeGRPC.sh` to help user to change their GCC and GRPC version quickly when they want to start different use cases.  Using our script to control your package version：

```shell
# check your GCC version
$ gcc -v
# change gcc version
$ sudo update-alternatives --config gcc
# change g++ version (optional)
$ sudo update-alternatives --config g++
# recompile GPRC with different gcc version
$ cd <path_to_changeGRPC.sh> && sudo ./changeGRPC.sh <path_to_grpc>
# check the installation
$ protoc --version
$ gcc --version
$ grpc --version
```



## Building for four case studies

## A. Case Study: Optimizing Image Pipelines

### A.1 Dependence

#### A.1.1 GCC and other environment requirements

Install libpng and libjpeg， run：

```shell
#libpng
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb \
  && sudo dpkg -i /tmp/libpng12.deb \
  && rm /tmp/libpng12.deb
#libjpeg
sudo apt-get install libjpeg-turbo8
```

grpc which was compiled with gcc-7.5. Find guidance from [grpc]((#1.2. grpc)).

#### A.1.2 Halide Installation

Our evaluation is based on Halide 10.0.0. Please refer to [Here](https://github.com/halide/Halide/tree/v10.0.0) to install Halide.

Or install with [source code zip](https://github.com/halide/Halide/releases/tag/v10.0.0)  and add halide libHalide.so path to system：

```shell
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_libhalide.so>' >> ~/.bashrc
```

#### A.1.3 RUN

Run with code：

```shell 
$ python supersonic_main.py --task Halide  --mode test
```

## B. Case Study: Neural Network Code Generation

### B.1. Dependence

#### B.1.1. TVM Installation

Our evaluation is based on [TVM 0.8.0](https://tvm.apache.org/download). Please refer to [Here](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github) to install Halide.

grpc which was compiled with gcc-7.5. Find guidance from [grpc]((#1.2. grpc)).

Add TVM path to your system：

```shell
$ export TVM_HOME="/<path_to_tvm>/tvm"
$ export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

#### B.1.2 RUN

Run with code：

```shell 
$ python supersonic_main.py --task Tvm  --mode test
```

## C. Case Study: Code Size Reduction

### C.1 Dependence

#### C.1.1 CompilerGym Installation

Our evaluation is based on CompilerGym v0.1.8. Our framework has insert the environment before, you could also look [here](https://github.com/facebookresearch/CompilerGym/tree/v0.1.8) to install CompilerGym. 

#### C.1.2 Test

Test with code：

```shell 
$ python supersonic_main.py --task CSR  --mode test
```

## D. Case Study: Super-optimization

### D.1 Dependence

#### D.1.1 GCC and other environment requirements

The key to making stoke right is using GCC version 4.9. Below that, the compiler doesn't support enough features to build the code.

```shell
$ sudo apt-get install bison ccache cmake doxygen exuberant-ctags flex g++-4.9  g++-multilib gcc-4.9 ghc git libantlr3c-dev libboost-dev libboost-filesystem-dev libboost-thread-dev libcln-dev libghc-regex-compat-dev libghc-regex-tdfa-dev libghc-split-dev libjsoncpp-dev python subversion libiml-dev libgmp-dev libboost-regex-dev autoconf libtool antlr pccts pkg-config
```

grpc which was compiled with gcc-4.9. Find guidance from [grpc]((#1.2. grpc)).

#### D.1.2 STOKE Installation

We provide a Stoke install package  [here](https://github.com/StanfordPL/stoke). Using our package to install Stoke, run:

```sehll
# Unzip the pre-compiled dependencies
$ cd <path_to_stoke>/src/ext && tar -xzvf x64asm.tar.gz
# Back to the root path and Compile stoke
$ cd ../../ && make
# add environment path
$ export PATH=$PATH:/<path_to_stoke>/bin
# add python environment path
$ echo <grpc_src_path>/ >> <python_site-package>/SS.pth
$ echo <opt>/ >> <python_site-package>/SS.pth
```

#### D.1.3 RUN

Run with code：

```shell 
$ python supersonic_main.py --task Stoke  --mode test
```

