# Automating Reinforcement Learning Architecture Design for Code Optimization: Artifact Instructions for Docker Image

The recommended approach for AE is to use the pre-configured, interactive Jupyter notebook with the instructions given in the AE submission. 

The following step-by-step instructions are given for using a  [Docker Image](#docker) running on a local host. Our docker image (40 GB uncompressed) contains the entire execution environment (including python and system dependencies), benchmarks, and source code which includes four optimization tasks: Halide image pipeline optimization, neural network code generation, compiler phase ordering for code size reduction, and superoptimization.  All of our code and data are open-sourced has been developed with extensibility as a primary goal.

*Disclaim:
Although we have worked hard to ensure our AE scripts are robust, our tool remains a *research prototype*. It can still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and what problem you encountered. We will get back to you ASAP. Thank you.*


# Step-by-Step Instructions 

## ★ Main Results 

The main results of our work are presented in Figures 3-6 in the submitted paper, which compares the SuperSonic-tuned RL client against alternative search techniques on each of the four case studies. 

## ★ Docker Image 

We prepare our artifact within a [Docker image](https://zenodo.org/record/4675014) to run "out of the box". 
Our docker image was tested on a host machine running Ubuntu 18.04.

## ★ Artifact Evaluation  

Follow the instructions below to use our AE evaluation scripts.

### 1. Setup

#### 1.1  Load the Docker Image <br id="loaddi">

After downloading the [docker image](#dockerimg), using the following commands to load the docker image (~30 minutes on a laptop) on the host machine:

```
unzip SuperSonic.zip
cd SuperSonic
docker load -i SuperSonic.tar
```

#### 1.2 Setup the Environment

After importing the docker container **and getting into bash** in the container, run the following command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate ss
``````

Then, go to the root directory of our tool:

```
(docker) $ cd /home/sys/SUPERSONIC
```

### 2. Evaluation

The following steps describe how to evaluate each individual case study. In each case study, we first describe how to use SuperSonic to search the RL client architecture. We   then show how to apply the searched client RL to test benchmarks and compare the results against the baselines.

#### 2.1 Task definition

A compiler developer can use the Supersonic API to define the optimization problem. This is done by creating an RL policy interface. The definition includes a list of client RL components for the meta-optimizer to search over.

You can modify elements from  statefs (state functions), rewards, rl_algs and actions to define the optimization problem from ```SuperSonic/policy_search/supersonic_main.py Policys()```.

```python
(python) 
#Candidate state functions
StateFunctions=["Word2vec", "Doc2vec", "Bert", "Actionhistory"]
#Candidate reward functions
RewardFunctions=["relative_measure", "tan", "func", "weight"]
#Candidate RL algorithms
RLAlgorithms=["MCTS", "PPO", "APPO", "A2C", "DQN", "QLearning"
        "MARWIL", "PG", "SimpleQ", "BC"]
#Candidate Action methods
ActionFunctions=["init"]
```



### 2.2. Case Study 1: **Optimizing Image Pipelines**

  The results correspond to Figure 3 of the submitted manuscript. 

#### 2.2.1 Client RL search and testing

(*approximate runtime:  ~12 hours*)

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py  --env BanditHalideEnv-v0   --mode policy --total_steps 70  --datapath "tasks/halide/resource"
```

You can change the following parameters:

```--env ``` The task environment that passed to RL client (Include 4 cases: BanditStokeEnv-v0, BanditTvmEnv-v0, BanditCSREnv-v0, BanditHalideEnv-v0). In this case, we set to BanditHalideEnv-v0. 

```--datapath``` Data path ( Change data path to use different benchmarks to support RL policy search). In this case, we set to the halide benchmarks' path "tasks/halide/resource". 

 ```--mode ``` "policy" - An automatic process includes RL Policy Search, and deploy the policy as well as parameters to the task; "config" - Parameters Tuning;  "deploy" - Deploy Policy and Parameter; We set it to "policy" to do the entire process.

```--total_steps``` to set the number of trials spent on client RL search. ( It should be set to > 70 )

#### 2.2.2 Testing the tuned RL client

To measure the runtime of the resulting binary, we run each benchmark at least 100 times on an unloaded machine. Using ```get_runtime.py``` to run data's result and get a speedup. Here are some parameter setting to calculate the halide' result.

`--task`: The case to be calculated. In this case, we set it to halide.

`--data`: Representing different halide data . Options: _harris, interpolate, hist, max_filter, unsharp, nl_mean, lens_blur, local_laplacian, conv_layer, st_chain_.

`--log_file`: The log file corresponds to the data is generated after the system-SuperSonic run. This  default log file path is in *<SUPERSONIC-root-path/SuperSonic/utils/result>*.

`--result_path`: The path to save results. The default result path is*<SUPERSONIC-root-path/tasks/get_runtime_save/>*.

Notes: Make sure the environment can compiler the halide binary. The GCC version is 7.5.

```shell
# computer running time 
#The default log file is in <SUPERSONIC-root-path/SuperSonic/utils/result>.
#The caculate time shell is in <SUPERSONIC-root-path>.
(docker) $ cd <SUPERSONIC-root-path>
(docker) $ python tasks/get_runtime.py  --task halide --data <halide_data> --log_file <log_file_path> --result_path <result_to_save>
#demo:python tasks/get_runtime.py  --task halide --data 'interpolate' --log_file  "tasks/get_runtime_save/interpolate_result.csv" --result_path 'tasks/get_runtime_save/'. You can find csv file in 'tasks/get_runtime_save/'.
```


### 2.3. Case Study 2: **Neural Network Code Generation**

  The results correspond to Figure 4 of the submitted manuscript. 

#### 2.3.1 Find the best policy and testing

(*approximate runtime:  ~3 hours*)

Notes: Make sure the environment can import the TVM. The GCC version is 7.5.

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py  --env BanditTvmEnv-v0 --datapath tasks/tvm/zjq/benchmark/
```

#### 2.3.2 Testing the tuned RL client

```shell
(docker) $ cd tasks/tvm/zjq/benchmark/ && python model_optimization.py --opt "rl" --do "test"
```

### 2.4. Case Study 3: **Code Size Reduction**

The results correspond to Figure 5 of the submitted manuscript. 

#### 2.4.1 Client RL search and testing

(*approximate runtime:  ~12 hours*)

Notes: Make sure the environment can import the compiler_gym.

```shell
(docker) $ python  SuperSonic/policy_search/supersonic_main.py --env BanditCSREnv-v0 --datapath "../../tasks/CSR/DATA" --mode policy --steps 1 --total_steps 70
```

### 2.5. Case Study 4: **Superoptimization**

The results correspond to Figure 6 of the submitted manuscript. 

#### 2.5.1 Client RL search and testing

(*approximate runtime:  ~12 hours*)

Notes: Make sure the environment can compiler the Stoke. (Following [this instruction](https://github.com/HuantWang/SUPERSONIC/edit/master/INSTALL.md#grpc) to rebuild grpc with gcc v4.9. )

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode policy --steps 1 --total_steps 70 ```
```                      

#### 2.5.2 Testing the tuned RL client

#modified by lzy
Using ```get_runtime.py``` to run data's running time. Here are some parameter setting to calculate the stoke' result.

`--task`: The case to be calculated. In this case, we set it to stoke.

`--hacker_number`: ID of the hack benchmark dataset, taking values between 1 and 25

```shell
#computer runnintg time
#The caculate time shell is in <SUPERSONIC-root-path>.
(docker) $ python tasks/get_runtime.py --task stoke --hacker_number 1`  
#demo:python tasks/get_runtime.py  --task stoke --hacker_number 1 
#Now you have calculated the running time of data p01.
