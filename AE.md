# Automating Reinforcement Learning Architecture Design for Code Optimization: Artifact Instructions for Docker Image

The recommended approach for AE is to use the pre-configured, interactive Jupyter notebook with the instructions given in the AE submission. 

The following step-by-step instructions are given for using a  [Docker Image](#docker) running on a local host. Our docker image (40 GB uncompressed) contains the entire execution environment (including python and system dependencies), benchmarks, and source code which includes four optimization tasks: Halide image pipeline optimization, neural network code generation, compiler phase ordering for code size reduction, and superoptimization.  All of our code and data are open-sourced has been developed with extensibility as a primary goal.

*Disclaim:
Although we have worked hard to ensure our AE scripts are robust, our tool remains a *research prototype*. It can still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and what problem you encountered. We will get back to you ASAP. Thank you.*


# Step-by-Step Instructions <br id = "docker">

## ★ Main Results <span id = "bug-list">

The main results of our work are presented in Figures 3-6 in the submitted paper, which compare the SuperSonic-tuned RL client against alternative search techniques on each of the four case studies. 

## ★ Docker Image <br id = "dockerimg">

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

#### 1.2 Setup environmental parameters:

After importing the docker container **and getting into bash** in the container, make sure you run the below command to select the conda environment, before using any of the AE scripts:

`````` shell
$ conda activate ss
``````

Then go to the root path of our framework:

```
(docker) $ cd /home/sys/SUPERSONIC
```

### 2. Evaluation
The following steps describe how to evaluate each individual case study. In each case study, we first describe how to use SuperSonic to search the RL client architecture. We   then show how to apply the searched client RL to test benchmarks and compare the results against the baseline.
  
#### Case Study 1: **Optimizing Image Pipelines**
  
  The results correspond to Figure 3 of the submitted manuscript. 
  
##### 2.1 Client RL search and deployment

(*approximate runtime:  ~12 hours*)

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py  --env BanditHalideEnv-v0   --mode policy --steps 1 --total_steps 70  --datapath "tasks/halide/resource"
```

(set ```--env BanditStokeEnv-v0``` to evaluate the Stoke, set ```--datapath "tasks/stoke/example/hacker"``` to change the benchmark to hacker, ```--mode policy``` to start our engine to find the best policy and deploy to benchmark. ```--total_steps``` to set the number of trials spent on client RL searching)

##### 2.2 Computing running time and speedup for each result

```shell
# computer running time 
#The default log file is in <SUPERSONIC-root-path/SuperSonic/utils/result>.
#The caculate time shell is in <SUPERSONIC-root-path/SuperSonic/tasks/halide>.
(docker) $ cd <SUPERSONIC-root-path/SuperSonic/tasks/halide>
(docker) $ python HalideShell.py  --algorithm_id <data_id> --log_file <log_file_path> 
#The default result path is in <SUPERSONIC-root-path/tasks/halide/result>.You can find csv file in it.
```

Using ```stoke replace``` to replace original benchmark with optimized one. Using ```RunTime.py``` to run ```hacker/pxx``` ```100000000``` times. Using ```CalculateTime.py``` to get a speedup.

### 3. Case Study 2: **Neural Network Code Generation**

  The results correspond to Figure 4 of the submitted manuscript. 

##### 3.1 Client RL search and deployment

(*approximate runtime:  ~3 hours*)

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py  --env BanditTvmEnv-v0 --datapath tasks/tvm/zjq/benchmark/
```

##### 3.2 Computing running time for each result

```shell
(docker) $ cd tasks/tvm/zjq/benchmark/ && python model_optimization.py --opt "rl" --do "test"
```

### 4. Case Study 3: **Code Size Reduction**
  
    The results correspond to Figure 5 of the submitted manuscript. 

##### 4.1 Client RL search and deployment

(*approximate runtime:  ~12 hours*)

```shell
(docker) $ python  SuperSonic/policy_search/supersonic_main.py --env BanditCSREnv-v0 --datapath "../../tasks/CSR/DATA" --mode policy --steps 1 --total_steps 70
```

(set ```--env BanditStokeEnv-v0``` to evaluate the Stoke, set ```--datapath "tasks/stoke/example/hacker"``` to set the benchmark to hacker, ```--mode policy``` to start our engine to find the best policy and deploy to benchmark. ```--total_steps``` to set the number of trials spent on client RL searching)：

### Case Study 4: **Superoptimization**

      The results correspond to Figure 6 of the submitted manuscript. 

  
##### 5.1 Setup environmental parameters:

Following [this instruction](https://github.com/HuantWang/SUPERSONIC/edit/master/INSTALL.md#grpc) to rebuild grpc with gcc v4.9.

##### 5.2 Client RL search and deployment

(*approximate runtime:  ~12 hours*)

```shell
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode policy --steps 1 --total_steps 70 ```
```

(set ```--env BanditStokeEnv-v0``` to evaluate the Stoke, set ```--datapath "tasks/stoke/example/hacker"``` to set the benchmark to hacker, ```--mode policy``` to start our engine to find the best policy and deploy it to test benchmarks. ```--total_steps``` to set the number of trials spent on client RL searching)：

##### 5.3 Computing running time for each result

```shell
# Concatenate the result file with the original data file
(docker) $ cd tasks/stoke/example/hacker/pxx
(docker) $ stoke replace --config replace.conf

# Computing running time
(docker) $ cd ../../
(docker) $ python RunTime.py hacker/pxx 100000000
(docker) $ python CalculateTime.py speedup
```
