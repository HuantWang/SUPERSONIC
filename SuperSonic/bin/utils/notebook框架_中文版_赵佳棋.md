# 1. 安装环境

## 1.1. python, llvm

## 1.2. grpc
- 下载 我们提供压缩包下载链接<>
- 编译安装protoc
- 编译安装grpc
- 测试cpp和python


# 2. case1 stoke


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

(base) zjq@DESKTOP-B5SJ9B4:envs$ tree
├── Optimization
│   ├── __init__.py # 注册tvm环境
│   └── rltvm_env.py # 基于compilerGym底层的强化学习实现
├── __init__.py # 需要在这里添加tvm任务
```


1. environments/rltvm_mcts.py
- 类MsgServicer作为MDP三元组交互的grpc底层类;
- 类mcts是基于mcts算法的强化学习底层, 在这里定义了action, state和reward的关系

该文件定义了mcts的环境, 包括xxx
与tvm的交互逻辑是
其中: 介绍里面重要的函数, 参数变量和功能

2. examples/train_tvm.py
启动强化学习服务器
<!-- 这里介绍底层强化学习的注册和实现 -->

3. Optimization/rltvm_env.py
- 类TVMEnv实现了compilerGym的强化学习, 给mcts作为下层代码, 

### 3.2.2. 待优化任务端
1. autotvm/tuner/index_
介绍里面的函数, 参数变量的功能

2. MDP构建
这里介绍reward/ action/ state的具体意义


## 3.3. 具体使用过程（这里主要是可运行代码demo展示）
### 3.3.1. 底层注册（这一节替换为一个环境注册代码运行示例，像cg的demo那样）
- 在`envs/Optimization/__init__.py`注册环境
```python
from gym.envs.registration import registry, register, make, spec
from compiler_gym.envs.Optimization.rltvm_env import  TVMEnv
register(
    id='tvm-v0',
    entry_point='compiler_gym.envs.Optimization:TVMEnv',
    # max_episode_steps=200,
    # reward_threshold=25.0,
)
```
- 在`envs/__init__.py`添加声明
```python
from compiler_gym.envs.Optimization.rltvm_env import  TVMEnv
__all__ = [
    "**" # other environment
    "TVMEnv",
]
```

- 将文件`envs/Optimization/rltvm_env.py`放到cg环境同名下

- 测试
```python
import gym
import compiler_gym

# compiler_gym.env("Tvm-v0")  # TODO(by zjq):  具体的学长填
```

### 3.3.2. mcts环境（下面配置路径这些都放到环境那里，这一节只演示代码示例）
- 确定`opt_test`文件路径

- 配置路径<site-package>/facebook.pth
    ```sh
    <grpc_src_path>/
    <opt_test_path>/
    ```

- 运行强化学习端
    `python `

### 3.3.3. tvm环境
- 运行tvm任务
    `python`

最后我们提供了jupyter的结果展示, 通过运行jupyter-notebook在浏览器上查看

# 4. case3 halide