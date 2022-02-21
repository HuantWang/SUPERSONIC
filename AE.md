# Automating Reinforcement Learning Architecture Design for Code Optimization: Artifact Instructions for Docker Image

The recommended approach for AE is to use the pre-configured, interactive Jupyter notebook with the instructions given in the AE submission. 

The following step-by-step instructions are given for using a  Docker Image running on a local host. Our docker image (40 GB uncompressed) contains the entire execution environment (including python and system dependencies), benchmarks, and source code which includes four optimization tasks: Halide image pipeline optimization, neural network code generation, compiler phase ordering for code size reduction, and superoptimization.  All of our code and data are open-sourced has been developed with extensibility as a primary goal.

Check
    <a href="http://1.14.76.177:7033/index.html">the website</a>
    for documents and more information of Supersonic. 

*Disclaim:
Although we have worked hard to ensure our AE scripts are robust, our tool remains a *research prototype*. It can still have glitches when used in complex, real-life settings. If you discover any bugs, please raise an issue, describing how you ran the program and what problem you encountered. We will get back to you ASAP. Thank you.*


# Step-by-Step Instructions 

## ★ Main Results 

The main results of our work are presented in Figures 3-6 in the submitted paper, which compares the SuperSonic-tuned RL client against alternative search techniques on each of the four case studies. 

## ★ Docker Image 

We prepare our artifact within a Docker image to run "out of the box". 
Our docker image was tested on a host machine running Ubuntu 18.04.

## ★ Artifact Evaluation  

Follow the instructions below to use our AE evaluation scripts.

### 1. Setup
Install Docker by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/). The following instructions assume the host OS runs Linux.

#### 1.1  Fetch the Docker Image
Fetch the docker image from docker hub.

```
$ sudo docker pull nwussimage/supersonic_0.1
```

To check the list of images, run:

```
$ sudo docker images
REPOSITORY                                   TAG                 IMAGE ID            CREATED             SIZE
nwussimage/supersonic_0.1		     latest              ac6b624d06de        2 hours ago         41.8GB
```

Run the docker image.

```
$ docker run -dit -P --name=supersonic nwussimage/supersonic_0.1 /bin/bash
$ docker start supersonic 
$ docker exec -it supersonic /bin/bash
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

The following steps describe how to evaluate each individual case study. In each case study, we first describe how to use SuperSonic to search the RL client architecture. We then show how to apply the searched client RL to test benchmarks and compare the results against the baselines.


### 2.1. Case Study 1: **Optimizing Image Pipelines**

This task aims to improve the optimization heuristic of the Halide compiler framework. Halide is a domain-specific language and compiler for image processing pipelines (or graphs) with multiple computation stages. A Halide program separates the expression of the computation kernels and the application processing pipeline from the pipeline’s schedule. Here, the schedule defines the order of execution and placement of data on the hardware. The goal of this task is to automatically synthesize schedules to minimize the execution time of the benchmark.

***Note:***

 **Make sure gcc-7 is chosen**. 

```shell
(ss)root@2d2dbe667d18:SUPERSONIC# update-alternatives --config gcc

There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path              Priority   Status
------------------------------------------------------------
  0            /usr/bin/gcc-7     100       auto mode
* 1            /usr/bin/gcc-4.9   80        manual mode
  2            /usr/bin/gcc-7     100       manual mode
Press <enter> to keep the current choice[*], or type selection number: 0
update-alternatives: using /usr/bin/gcc-7 to provide /usr/bin/gcc (gcc) in manual mode

(ss)root@2d2dbe667d18:SUPERSONIC# cd /home/sys/SUPERSONIC/third_party && ./changeGRPC.sh /home/sys/SUPERSONIC/third_party/grpc/
```

#### 2.1.1 Client RL search and testing

(*approximate runtime:  **~ 30 minutes***)

##### Note:

You may encounter an error of failed tests. This is because we reduce the RL client search steps to make the search time manageable for the demo. Such failure did occur during our full-scale evaluation.

```shell
#Client RL search
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditHalideEnv-v0  --datapath "tasks/halide/resource" --mode policy --total_steps 10 2>/dev/null
#Client RL Parameter Tuning
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditHalideEnv-v0  --datapath "tasks/halide/resource" --mode config --iterations 10 --task Halide 2>/dev/null
#Client RL Deployment
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditHalideEnv-v0  --datapath "tasks/halide/resource" --mode deploy --training_iterations 50 --task Halide 2>/dev/null
```

You can change the following parameters:

```--env ``` The task environment that passed to RL client (Include 4 cases: BanditStokeEnv-v0, BanditTvmEnv-v0, BanditCSREnv-v0, BanditHalideEnv-v0). In this case, we set to BanditHalideEnv-v0. 

```--datapath``` Data path ( Change data path to use different benchmarks to support RL policy search). In this case, we set to the halide benchmarks' path "tasks/halide/resource". 

 ```--mode ``` "policy" - An automatic process includes RL Policy Search, and deploy the policy as well as parameters to the task; "config" - Parameters Tuning;  "deploy" - Deploy Policy and Parameter; We set it to "policy" to do the entire process.

```--total_steps``` to set the number of trials spent on client RL search. 

#### 2.1.2 Performance evaluation on benchmarks

The results correspond to Figure 3 of the submitted manuscript.

*approximate runtime ~60 minutes for one benchmark*

```shell
(docker) $ python tasks/halide/run.py
# download the PDF from docker container to the host machine
(docker) $ sz /home/sys/SUPERSONIC/AE/halide/graph/*
```


### 2.2. Case Study 2: **Neural Network Code Generation**

This task targets DNN back-end code generation to find a good schedule. e.g., instruction orders and data placement to reduce execution time on a multi-core CPU.

This demo corresponds to Figure 4 of the submitted manuscript.

***Note:***

**Like case study 1, choose gcc-7 for this demo**. 

You may encounter an error of failed tests. This is because we reduce the RL client search steps to make the search time manageable for the demo. Such failure did occur during our full-scale evaluation.

#### 2.2.1 Find the best policy and testing

(*approximate runtime:  **~ 30 minutes***)

```shell
#Client RL search
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditTvmEnv-v0  --datapath "tasks/tvm/zjq/benchmark/" --mode policy --total_steps 10 2>/dev/null
#Client RL Parameter Tuning
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditTvmEnv-v0  --datapath "tasks/tvm/zjq/benchmark/" --mode config --iterations 10 --task Tvm 2>/dev/null
#Client RL Deployment
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditTvmEnv-v0  --datapath "tasks/tvm/zjq/benchmark/" --mode deploy --training_iterations 50 --task Tvm 2>/dev/null
```

#### 2.2.2 Performance evaluation on benchmarks

This demo corresponds to Figure 4 of the submitted manuscript.

*approximate runtime = 10 minutes for one benchmark*

```shell
(docker) $ python tasks/tvm/run.py
# download the PDF from docker container to the host machine
(docker) $ sz /home/sys/SUPERSONIC/AE/tvm/graph/*
```

### 2.3. Case Study 3: **Code Size Reduction**

This task is concerned with determining the LLVM passes and their order to minimize the code size.

This demo corresponds to Figure 5 of the submitted manuscript.

***Note:***

Make sure the environment can import the compiler_gym.

You may encounter an error of failed tests. This is because we reduce the RL client search steps to make the search time manageable for the demo. Such failure did occur during our full-scale evaluation.

#### 2.3.1 Client RL search and testing

This demo shows how to apply the saved client RL to optimize a test program for Code Size Reduction.

*approximate runtime ~ 20 minutes*

```shell
#Client RL search
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditCSREnv-v0  --datapath "tasks/CSR/DATA" --mode policy --total_steps 10 2>/dev/null
#Client RL Parameter Tuning
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditCSREnv-v0  --datapath "tasks/CSR/DATA" --mode config --iterations 10 --task CSR 2>/dev/null
#Client RL Deployment
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditCSREnv-v0  --datapath "tasks/CSR/DATA" --mode deploy --training_iterations 50 --task CSR 2>/dev/null
```

#### 2.3.2 Performance evaluation on benchmarks

The results correspond to Figure 5 of the submitted manuscript.

*approximate runtime ~30 minutes for one benchmark*

```shell
(docker) $ python tasks/CSR/run.py
# download the PDF from docker container to the host machine
(docker) $ sz /home/sys/SUPERSONIC/AE/csr/graph/*
```



### 2.4. Case Study 4: **Superoptimization**

This classical compiler optimization task finds a valid code sequence to maximize the performance of a loop-free sequence of instructions. Superoptimizaiton is an expensive optimization technique as the number of possible configurations grows exponentially as the instruction count to be optimized increases.

This demo corresponds to Figure 5 of the submitted manuscript.

***Notes***:

Follow the following instructions to choose *gcc v4.9 and an other gRPC* for this demo as the stoke implementation only works for this setting.  


```shell
(ss)root@2d2dbe667d18:SUPERSONIC# update-alternatives --config gcc

There are 2 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path              Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-7     100       auto mode
  1            /usr/bin/gcc-4.9   80        manual mode
  2            /usr/bin/gcc-7     100       manual mode
Press <enter> to keep the current choice[*], or type selection number: 1
update-alternatives: using /usr/bin/gcc-4.9 to provide /usr/bin/gcc (gcc) in manual mode

(ss)root@2d2dbe667d18:SUPERSONIC# cd /home/sys/SUPERSONIC/third_party && ./changeGRPC.sh /home/sys/SUPERSONIC/third_party/grpc/
```

#### 2.4.1 Client RL search and testing

This demo shows how to apply the saved client RL to optimize a test program for Superoptimization.

*approximate runtime ~ 30 minutes*

```shell
#Client RL search
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode policy --total_steps 10 2>/dev/null
#Client RL Parameter Tuning
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode config --iterations 10 --task Stoke 2>/dev/null
#Client RL Deployment
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker/p20" --mode deploy --training_iterations 50 --task Stoke 2>/dev/null
```

#### 2.4.2 Performance evaluation on benchmarks

The results correspond to Figure 6 of the submitted manuscript.

*approximate runtime ~15 minutes for one benchmark*

```shell
(docker) $ python tasks/stoke/run.py
# download the PDF from docker container to the host machine
(docker) $ sz /home/sys/SUPERSONIC/AE/stoke/graph/*
```
