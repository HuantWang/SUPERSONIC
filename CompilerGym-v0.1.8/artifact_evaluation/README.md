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

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- **For Intel:**

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- [ ] ### **ChengZhang**

## Evaluation: Section V.B

- **For AMD:**

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- **For Intel:**

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- [ ] ### **ChengZhang**

## Evaluation: Section V.C

- **For AMD:**

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- **For Intel:**

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
(docker) $ ./test_precision.sh
```

Estimated time: xx minutes

- [ ] ### **JiaqiZhao**

## Evaluation: Section V.D

Run the script to reproduce the results in Section V.D.

```
(docker) $ cd /usr/src/artifact-cgo/precision/test
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

- [ ] ### **HuantingWang**
