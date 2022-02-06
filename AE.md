# SUPERSONIC Artifact and Evaluation

We provide a pre-configured live server with a [Python Jupyter Notebook]() and a [Docker Image](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/INSTALL.md#dockerfix) to support artifact for our paper (SUPERSONIC) on CC 2022 paper.

## Contents

1. [Build and Run the Artifact](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/artifact_evaluation/README.md#section-1-build-and-run-the-artifact)
2. [Jupyter Experimental Evaluation](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/artifact_evaluation/README.md#section-2-jupyter---experimental-evaluation)
3. [Docker Experimental Evaluation](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/artifact_evaluation/README.md#section-3-docker---experimental-evaluation)

# Section 1: Build and Run the Artifact

For convenience, we have provided a pre-configured live server with a [Python Jupyter Notebook]() to work through our techiques (Please see the ReadMe document on the AE submission website on how to access the Notebook).

See [INSTALL.md](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/INSTALL.md) to build SUPERSONIC. While it is possible to create your own copy of our Jupyter Notebook from [source code](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/INSTALL.md#building-from-source-fix), we recommend to use the [docker container](https://github.com/NWU-NISL-Optimization/SuperSonic/blob/AE/INSTALL.md#dockerfix) we provided.

# Section 2: Jupyter - Experimental Evaluation

You could refer [here]() for detailed instructions to reproduce the results with Python Jupyter Notebook.

# Section 3: Docker - Experimental Evaluation

After you have successfully run the docker image, you can go the path:

```
(docker) $ cd /usr/src/xx
```

This directory contains the entire setup of our tool. This section provides details on how to evaluate the results section in our paper.

## Evaluation: Section V.A

Run the script to reproduce the results in Section V.A.

- **For AMD:**

```shell
# Get experimental results
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode policy --steps 1 --total_steps 70

# Concatenate the result file with the original data file
(docker) $ cd tasks/stoke/example/hacker/pxx
(docker) $ stoke replace --config replace.conf

# computer running time
(docker) $ cd ..
(docker) $ python RunTime.py pxx/ 100000000
(docker) $ python CalculateTime.py speedup
```

Estimated time: xx minutes

- **For Intel:**

```shell
# Get experimental results
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditStokeEnv-v0  --datapath "tasks/stoke/example/hacker" --mode policy --steps 1 --total_steps 70

# Concatenate the result file with the original data file
(docker) $ cd tasks/stoke/example/hacker/pxx
(docker) $ stoke replace --config replace.conf

# computer running time
(docker) $ cd ..
(docker) $ python RunTime.py pxx/ 100000000
(docker) $ python CalculateTime.py speedup
```

Estimated time: xx minutes


## Evaluation: Section V.B

- **For AMD:**

```shell
# Get experimental results
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditHalideEnv-v0 --mode policy --steps 1 --total_steps 70

# computer running time
#The default log file is in <SUPERSONIC-root-path/SuperSonic/utils/result>.
(docker) $ cd <SUPERSONIC-root-path/SuperSonic/utils/result>
#The default caculate_time file for halide is in <SUPERSONIC-root-path/tasks/halide/app-halide>.
(docker) $ cd <SUPERSONIC-root-path/tasks/halide/app-halide>
#You can find the schedules in log file then add it into caculate_time file and compile.You can add after the comment ' //xx schedules'
```

Estimated time: xx minutes

- **For Intel:**

```shell
# Get experimental results
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/supersonic_main.py --env BanditHalideEnv-v0 --mode policy --steps 1 --total_steps 70

# computer running time
#The default log file is in <SUPERSONIC-root-path/SuperSonic/utils/result>.
(docker) $ cd <SUPERSONIC-root-path/SuperSonic/utils/result>
#The default caculate_time file for halide is in <SUPERSONIC-root-path/tasks/halide/app-halide>.
(docker) $ cd <SUPERSONIC-root-path/tasks/halide/app-halide>
#You can find the schedules in log file then add it into caculate_time file and compile.You can add after the comment ' //xx schedules'
```

Estimated time: xx minutes

- [ ] ### **ChengZhang**

## Evaluation: Section `V.C`

- **`For AMD`:**

```shell
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/tvm_main.py  --env BanditTvmEnv-v0
(docker) $ python SuperSonic/policy_search/tvm_test.py
```

Estimated time: 180 minutes

- **`For Intel`:**

```shell
(docker) $ cd <SUPERSONIC-root-path>/
(docker) $ python SuperSonic/policy_search/tvm_main.py  --env BanditTvmEnv-v0
(docker) $ python SuperSonic/policy_search/tvm_test.py
```

Estimated time: 180 minutes


## Evaluation: Section V.D

Run the script to reproduce the results in Section V.D.

```
(docker) $ cd python SuperSonic/policy_search/ supersonic_main.py --env BanditCSREnv-v0 --datapath "../../tasks/CSR/DATA" --mode policy --steps 1 --total_steps 70
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- [ ] ### **YamengLu**

## Evaluation: Section V.E

Run the script to reproduce the results in Section V.E.

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes
