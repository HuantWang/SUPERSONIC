# 1. 安装环境

## 1.1. python, llvm

### 1.1.1 Acquiring LLVM

可以从[llvm](https://releases.llvm.org/)获得安装文件，为了编译halide的代码需要配置`llvm-config`和`clang`的环境变量：

```shell
$ export LLVM_ROOT=<path_to_llvm>/bin/llvm-install
$ export LLVM_CONFIG=<path_to_llvm>/bin/llvm-config
```

## 1.2. grpc

We use gRPC to connected our stoke-mcts system, so we have to build the gRPC first. The version of gRPC we used is **v1.12.0**. We provide the grpc's compressed package and you can download it at<下载链接>. Use our package to build the grpc, run:

- 解压文件
- 编译安装protoc
- 编译安装grpc
- 测试cpp和python

### 1.3 切换GCC编译grpc系统

由于stoke需要在4.9版本的GCC上编译grpc完成对系统的编译，而其他任务需要更高的GCC版本的支持，所以在运行stoke和其他任务之间需要切换GCC重新编译，提供了切换编译的脚本文件`changeGRPC.sh`，使用如下：

```shell
#查看GCC版本
gcc -v
#进行gcc版本切换
sudo update-alternatives --config gcc
#进行g++版本切换
sudo update-alternatives --config g++
#重新编译GPRC
cd <path_to_changeGRPC.sh> && ./changeGRPC.sh <path_to_grpc>
eg. ./changeGRPC.sh /home/grpc/
```



# 2. case1 stoke

## 2.1 必要环境

### 2.1.1 python

见1.1

### 2.1.2 llvm

见1.1

- 下载
- 编译
- 安装  

### 2.1.3 GRPC

见1.2

### 2.1.4 GCC and other environment requirement

The key to making stoke right is using gcc version 4.9. Below that, the compiler doesn't support enough features to build the code. Above that, there are some issues with an ABI change in gcc-5.

```shell
$ sudo apt-get install bison ccache cmake doxygen exuberant-ctags flex g++-4.9  g++-multilib gcc-4.9 ghc git libantlr3c-dev libboost-dev libboost-filesystem-dev libboost-thread-dev libcln-dev libghc-regex-compat-dev libghc-regex-tdfa-dev libghc-split-dev libjsoncpp-dev python subversion libiml-dev libgmp-dev libboost-regex-dev autoconf libtool antlr pccts pkg-config
```

### 2.1.5 stoke

We provide the stoke's source code and you can download it at <下载链接> or you can  download it at [stoke](https://github.com/StanfordPL/stoke).Use our package to build the grpc, run:

```sehll
# Unzip the pre-compiled third-party library
$ cd <path_to_stoke>/src/ext && tar -xzvf x64asm.tar.gz
# Back to the root path
$ cd ../../
# Compile stoke
$ make
# Set environment Path
$ export PATH=$PATH:/<path_to_stoke>/bin
```

### 2.1.6 Python environment path

Before you use the code you have to expand python environment path, run:

```shell
$ echo <grpc_src_path>/ >> <python_site-package>/facebook.pth
$ echo <opt>/ >> <python_site-package>/facebook.pth

```

## 3.2 Code

### 3.2.1 RL

```
opt_test$ tree
.
└── MCTS
    |-- __init__.py
    |-- core
    |   |-- __init__.py
    |   |-- alpha_zero_policy.py
    |   |-- alpha_zero_trainer.py
    |   |-- mcts.py 
    |   └── ranked_rewards.py
    |-- environments
    |   |-- __init__.py
    |   -- stoke_rl_env.py #基于mcts算法的强化学习的底层代码
    |-- examples
    |   |-- __init__.py
    |   |-- train_stoke.py  #File to run RL
    └── models
        |-- __init__.py
        └── custom_torch_models.py
compiler_gym/envs$ tree
.
└── Optimization
|   |-- Stoke.py
|   |-- __init__.py
|-- __init__.py
|-- __pycache__
|-- compiler_env.py
└── llvm
  
```

1. environments/stoke_rl_env.py
   This file defines the environment of mcts, including the main functions of *init, reset, step, set_state, and get_state*. The interaction with stoke is: ① Establish a grpc link with stoke in the init environment ② Accept the state and reward from stoke in the Step call and return the action to stoke at the end of the step to guide its next code mutation.
   
   
   
2. Optimization/Stoke.py
   
   The implementation of the stoke.py is based on the gym framework, and the environment registration is required to complete the registration of the stoke environment, which involves the stoke environment registration code of the init.py file:

   ```python
   from compiler_gym.envs.Optimization.Stoke import StokeEnv
   __all__ = ["StokeEnv"]
   ```
   
   In Optimization/init.py file:
   
   ```python
   from gym.envs.registration import make, register, registry, spec
   register(
       id="Stoke-v0",
       entry_point="compiler_gym.envs.Optimization:StokeEnv",
   )
   
   ```
   
   In Optimization/Stoke.py file:
   
   ```python
   import math
   import gym
   from gym.utils import seeding
   from compiler_gym.mdp_search.action import get_action
   from compiler_gym.mdp_search.observation import get_observation, obs_init
   from compiler_gym.mdp_search.reward import get_reward
   
   class StokeEnv(gym.Env):
       def __init__(self, observation, action, reward):
           ...
       def get_reward(self, method, action, reward, reward_last):
           ...
       def get_obs(self, action,state_code):
           ...
       def step(self, action, state_code, reward,reward_last):
           ...
       def reset(self):
           ...
       def render(self, mode="human"):
           ...
       
   ```
   
   After registering of the stoke environment, we can create an example of the stoke environment through the following code:
   
   ```python
   self.env = gym.make(
               "Stoke-v0",
               observation=xxx,
               action=xxx,
               reward=xxx,
           )
   ```
   
3. **example/train_stoke.py**
   This file using the ray framework to optimize the underlying stoke by calling the environment set by mcts. The ray.tune calling code is as follows, calling the mcts environment in stoke_rl_env.py:
   
   ```python
   tune.run(
               "contrib/AlphaZero",
               stop={"training_iteration": args.training_iteration},
               max_failures=0,
               local_dir=r"/home/stoke/opt_test/MCTS/AutoMDP/model_save/",
               config={
                   "env": mcts,
                   "env_config": {
                       "observation": policy["obs_list"],
                       "action": policy["act_list"],
                       "reward": policy["rew_list"],
                   },
                   "num_workers": args.num_workers,
                   "rollout_fragment_length": 50,
                   "train_batch_size": 50,
                   "sgd_minibatch_size": 64,
                   "lr": 1e-4,
                   "num_sgd_iter": 1,
                   "mcts_config": {
                       "puct_coefficient": 1.5,
                       "num_simulations": 10,
                       "temperature": 1.0,
                       "dirichlet_epsilon": 0.20,
                       "dirichlet_noise": 0.03,
                       "argmax_tree_policy": False,
                       "add_dirichlet_noise": True,
                   },
                   "ranked_rewards": {"enable": True,},
                   "model": {"custom_model": "dense_model",},
               },
           )
   
   ```
   
   

### 3.2.2. 待优化任务端

1. **stoke**
   STOKE is a stochastic optimizer and program synthesizer for the x86-64 instruction set. STOKE uses random search to explore the extremely high-dimensional space of all possible program transformations. Although any one random transformation is unlikely to produce a code sequence that is desirable, the repeated application of millions of transformations is sufficient to produce novel and non-obvious code sequences. STOKE can be used in many different scenarios, such as optimizing code for performance or size, synthesizing an implementation from scratch or to trade accuracy of floating point computations for performance. As a superoptimizer, STOKE has been shown to outperform the code produced by general-purpose and domain-specific compilers, and in some cases expert hand-written code. More information about stoke you can learn from [stoke](https://github.com/StanfordPL/stoke).

2. **MDP构建**

   ***action：***There are 9 search transformations in stoke and we choose them as RL's action:

   |    Name     |                         Description                          |
   | :---------: | :----------------------------------------------------------: |
   |  add_nops   |       Adds one extra nop instruction into the rewrite.       |
   |   delete    |              Deletes one instruction at random.              |
   | instruction |  Replaces an instruction with another one chosen at random.  |
   |   opcode    | Replaces an instruction's opcode with a new one that takes operands of the same type. |
   |   operand   |     Replaces an operand of one instruction with another.     |
   |   rotate    | Formerly "resize". Moves an instruction from one basic block to another, and shifts all the instructions in between. |
   | local_swap  | Takes two instructions in the same basic block and swaps them. |
   | global_swap | Takes two instructions in the entire program and swaps them. |
   |  weighted   |    Selects from among several other transforms at random.    |
   
   ***state：***When search has run, STOKE can get the rewrite and we use doc2vec to embedding it as our state.
   
   ***reward：***Here we use the correctness as our reward which means how "correct" the rewrite's output appear .

## 3.3 Usage

1. There is a demo to use the environment of stoke:

   ```python
   import gym
   import compiler_gym
   
   env = gym.make(
           "Stoke-v0",
           observation="Doc2vec",
           action="transform",
           reward="hamming",
   )
   env.reset()
   env.step(env.action_space.sample(),"._Ztest:blsrl %edi, %eax  retq ",10,12)
   print(env.state)
   ```

##### 在supersonic中运行stoke任务，采取基于以下代码的方式：

```shell
#运行MDP代码
cd <path_to_supersonic>/torchbeastpopart  && python main.py --env BanditStokeEnv-v0
```

## 3.4 Dataset

The data sets we provide for optimization on stoke are hacker, spec, and llvm_testsuit:：

**hacker：**Hacker’s Delight, commonly referred to as “the bible of bittwiddling hacks”, is a collection of techniques for encoding otherwise complex algorithms as small loop-free sequences of bitmanipulating instructions. Gulwani notes this as a source of benchmarks for program synthesis and superoptimization, and identifies a 25 program benchmark which ranges in complexity from turning off the right-most bit in a word, to rounding up to the next highest power of 2, or selecting the upper 32 bits from a 64-bit multiplication. Our implementation of the benchmark uses the C code found in the original text. 

**spec：**

**llvm_testsuit**：


# 3. case2 tvm
## 3.1. 必要环境
### 3.1.1. python
### 3.1.2. llvm
- 下载
- 编译
- 安装


### 3.1.3. grpc
见1.2
### 3.1.4. tvm
- 下载
- 安装
- 编译
- 插入环境变量

## 3.2. 代码介绍

### 3.2.1. RL端

```shell
opt_test$ tree
└── MCTS
    ├── __init__.py
    ├── core
    │   ├── __init__.py
    │   ├── alpha_zero_policy.py
    │   ├── alpha_zero_trainer.py
    │   ├── mcts.py
    │   └── ranked_rewards.py
    ├── environments
    │   ├── __init__.py
    │   └── rltvm_mcts.py # 基于mcts算法的强化学习的底层代码
    ├── examples
    │   ├── __init__.py
    │   ├── rm_port.py # 关闭由于grpc生成的进程
    │   └── train_tvm.py # 强化学习端主要调用的文件
    └── models
        ├── __init__.py
        └── custom_torch_models.py
 
```


1. environments/rltvm_mcts.py
该文件定义了mcts的环境, 包括xxx
与tvm的交互逻辑是
其中: 介绍里面重要的函数, 参数变量和功能
2. 底层代码Stoke.py
    其中: 介绍里面重要的函数, 参数变量和功能
    <!-- 这里介绍底层强化学习的注册和实现 -->
3. **example/train_tvm.py**
    该文件定义了强化学习前端, 通过调用mcts设定好的环境, 用来优化tvm底层
    其中: 介绍里面重要的函数, 参数变量和功能

### 3.2.2. 待优化任务端
1. autotvm/tuner/index_
介绍里面的函数, 参数变量的功能

2. MDP构建
这里介绍reward/ action/ state的具体意义


## 3.3. 具体使用过程
1. 在cg环境的底层修改三个文件用于注册__init_.py, register.py xx
2. 将Opt文件夹放入到cg环境中


3. 配置路径<site-package>/facebook.pth
    ```sh
    <grpc_src_path>/
    <opt>/
    ```

4. 运行强化学习端
    `python `

5. 运行tvm任务
    `python`

最后我们提供了jupyter的结果展示, 通过运行jupyter-notebook在浏览器上查看

# 4. case3 halide

## 4.1 必要环境

### 4.1.1 python

见1.1

### 4.1.2 llvm

对llvm的需求10.0版本以上即可进行编译

见1.1

- 下载
- 编译
- 安装  

### 4.1.3 GRPC

见1.2

### 4.1.4 配置GCC和其他环境依赖

编译halide使用的GCC版本7.5以上即可，运行halide加载图像，需要安装libpng和libjpeg库，安装参考命令：

```shell
#安装libpng库
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb \
  && sudo dpkg -i /tmp/libpng12.deb \
  && rm /tmp/libpng12.deb
#安装libjpeg库
sudo apt-get install libjpeg-turbo8
```

### 4.1.5 安装Halide

Halide的安装文件可以从[halide](https://github.com/halide/Halide/releases)获得，可以参照官网教程源码编译安装或使用二进制文件，之后配置halide中的libHalide.so路径：

```shell
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_to_libhalide.so>' >> ~/.bashrc
```

## 5 使用

halide依赖环境和supersonic环境搭建好后可以通过以下代码运行：

```shell 
#运行MDP代码
cd <path_to_supersonic>/torchbeastpopart && python main.py  --env BanditHalideEnv-v0
```



## 附录

#### 可能存在的问题

```shell
1.bandit等环境注册，涉及到的文件路径:<path_to_compiler_gym>/env/__init__.py、<path_to_compiler_gym>/env/Optimization/__init__.py
2.可能需要自己手动清理端口
3.如果切换GRPC脚本运行出错，按脚本内容一条条自己执行即可
4.可能涉及到一些代码中的绝对路径要修改
```

