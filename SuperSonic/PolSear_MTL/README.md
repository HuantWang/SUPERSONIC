# TorchBeastPopArt
[PopArt](https://arxiv.org/abs/1809.04474) extension to [TorchBeast](https://github.com/facebookresearch/torchbeast), the PyTorch implementation of [IMPALA](https://github.com/deepmind/scalable_agent).

# Experiments
The PopArt extension was used to train a multi-task agent for six Atari games (AirRaid, Carnival, DemonAttack, Pong, SpaceInvaders, all with the NoFrameskip-v4 variant) and compared to the corresponding single-task agents and to a simpler mulit-task agent without PopArt normalisation. More details on these experiments can be found in the [report](results/report.pdf).

## Movies

Single-task:  
![AirRaid (Single-task clipped)](movies/AirRaid_050009600_AirRaidNoFrameskip-v4.gif)
![Carnival (Single-task clipped)](movies/Carnival_050002560_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Single-task clipped)](movies/DemonAttack_050001280_DemonAttackNoFrameskip-v4.gif)
![Pong (Single-task clipped)](movies/Pong_050013440_PongNoFrameskip-v4.gif)
![SpaceInvaders (Single-task clipped)](movies/SpaceInvaders_050001280_SpaceInvadersNoFrameskip-v4.gif)
  
Multi-task (clipped):  
![AirRaid (Multi-task clipped)](movies/MultiTask_300014720_AirRaidNoFrameskip-v4.gif)
![Carnival (Multi-task clipped)](movies/MultiTask_300014720_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Multi-task clipped)](movies/MultiTask_300014720_DemonAttackNoFrameskip-v4.gif)
![Pong (Multi-task clipped)](movies/MultiTask_300014720_PongNoFrameskip-v4.gif)
![SpaceInvaders (Multi-task clipped)](movies/MultiTask_300014720_SpaceInvadersNoFrameskip-v4.gif)
  
Multi-task PopArt:  
![AirRaid (Multi-task PopArt)](movies/MultiTaskPopart_300010240_AirRaidNoFrameskip-v4.gif)
![Carnival (Multi-task PopArt)](movies/MultiTaskPopart_300010240_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Multi-task PopArt)](movies/MultiTaskPopart_300010240_DemonAttackNoFrameskip-v4.gif)
![Pong (Multi-task PopArt)](movies/MultiTaskPopart_300010240_PongNoFrameskip-v4.gif)
![SpaceInvaders (Multi-task PopArt)](movies/MultiTaskPopart_300010240_SpaceInvadersNoFrameskip-v4.gif)

The different games plans learned by these three models, can be illustrated with the help of saliency maps (here red is the policy saliency and green is the baseline saliency). More details on these experiments can be found in the [report](results/report.pdf).

Saliency:  
![AirRaid](movies/Saliency_AirRaidNoFrameskip-v4.gif)
![Carnival](movies/Saliency_CarnivalNoFrameskip-v4.gif)
![DemonAttack](movies/Saliency_DemonAttackNoFrameskip-v4.gif)
![Pong](movies/Saliency_PongNoFrameskip-v4.gif)
![SpaceInvaders)](movies/Saliency_SpaceInvadersNoFrameskip-v4.gif)


## Trained models
The following trained models can be downloaded from the [models](models/) directory:

| Name | Environments (NoFrameskip-v4) | Steps (millions) |
| ---- |------------- | ---------------- |
| [AirRaid](models/AirRaid) | AirRaid | 50 |
| [Carnival](models/Carnival) | Carnival  | 50 |
| [DemonAttack](models/DemonAttack) | DemonAttack | 50 |
| [NameThisGame](models/NameThisGame) | NameThisGame | 50 |
| [Pong](models/Pong) | Pong | 50 |
| [SpaceInvaders](models/SpaceInvaders) | SpaceInvaders | 50 |
| [MultiTask](models/MultiTask) | AirRaid, Carnival, DemonAttack, NameThisGame, Pong, SpaceInvaders | 300 |
| [MultiTaskPopArt](models/MultiTaskPopArt) | AirRaid, Carnival, DemonAttack, NameThisGame, Pong, SpaceInvaders | 300 |


# Running the code
## Preparation
For our experiments we used the faster [PolyBeast](https://github.com/facebookresearch/torchbeast#faster-version-polybeast) implementation of TorchBeast and refer the reader to the installation instructions in the original repository. However, since we have encountered problems getting this version to work, we also added multi-task training functionality and PopArt to the [MonoBeast](https://github.com/facebookresearch/torchbeast#getting-started-monobeast) implementation of TorchBeast. However, some of the testing functionality is not implemented for this version, but PolyBeast can be used for this if the imports for `nest` and `libtorchbeast` are commented out.

Since it is more convenient to get PolyBeast to run, these are the platforms on which we managed to install and use it:
- Ubuntu 18.04
- MacOS (CPU only)
- Google Cloud Platform (Standard machine with NVIDIA Tesla P100 GPUs)

## Training a model
```bash
python -m torchbeast.polybeast --mode train --xpid MultiTaskPopArt --env AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4,NameThisGameNoFrameskip-v4,PongNoFrameskip-v4,SpaceInvadersNoFrameskip-v4 --total_steps 300000000 --use_popart
```
There are the following additional flags, as compared to the original TorchBeast implementation:
- `use_popart`, to enable to PopArt extension
- `save_model_every_nsteps`, to save intermediate models during training

### With MonoBeast
```bash
python -m torchbeast.monobeast --mode train --xpid MultiTaskPopArt --env AirRaidNoFrameskip-v4,CarnivalNoFrameskip-v4,DemonAttackNoFrameskip-v4,NameThisGameNoFrameskip-v4,PongNoFrameskip-v4,SpaceInvadersNoFrameskip-v4 --total_steps 300000000 --use_popart
```

In addition MonoBeast can also be used to run two other models: a small CNN (optionally with an LSTM) and an [Attention-Augmented Agent](https://arxiv.org/abs/1906.02500) (models selected with the flag `agent_type`). Unfortunately we did not get this model to train properly, but for the sake of completeness and possible future reference, here are the additional flags that can be used with this model:
- `frame_height` and `frame_width`, which set the dimensions to which frames are rescaled (in the original paper the original size is used as opposed to the rescaling done in TorchBeast)
- `aaa_input_format` (with choices `gray_stack`, `rgb_last`, `rgb_stack`), which decides how frames are formatted as input for the network (where `rgb_last` only feeds one of every four frames in RGB, as is done in the original paper)

## Testing a model
```bash
python -m torchbeast.polybeast --mode test --xpid MultiTaskPopArt --env PongNoFrameskip-v4 --savedir=./models
python -m torchbeast.polybeast --mode test_render --xpid MultiTaskPopArt --env PongNoFrameskip-v4 --savedir=./models
```

## Saliency
```bash
python -m torchbeast.saliency --xpid MultiTaskPopArt --env PongNoFrameskip-v4 --first_frame 0 --num_frames 100 --savedir=./models
```
Note that compared to the original [saliency code](https://github.com/greydanus/visualize_atari), the extension does not produce a movie directly, but saves the frames as individual images. Animated gifs can subsequently be produced with a [Jupyter notebook](results/movies.ipynb).

## CNN filter comparisons
**NOTE:** it is assumed that a) intermediate model checkpoints have been saved (flag `save_model_every_nsteps`) and b) the results for all models are saved in the same parent directory and have the exact names used in our experiments (see in the [table](https://github.com/aluscher/torchbeastpopart#trained-models))
```bash
python -m torchbeast.analysis.analyze_resnet --model_load_path /path/to/directory --mode filter_comp --comp_num_models 10
```
The different comparisons presented in the [report](results/report.pdf) can be set with the flag `comp_between`. By default the only comparisons done are between the vanilla multi-task model and the multi-task PopArt model, as well as between each of these models and all single-task models.

For plotting the following command can be used (saving the figures in the same directory that the data generated by the previous command was loaded from):
```bash
python -m torchbeast.analysis.analyze_resnet --load_path /path/to/directory --mode filter_comp_plot --save_figures
```
For more options to the data generation and plotting, the help texts can be consulted.

# References
TorchBeast
```
@article{torchbeast2019,
  title={{TorchBeast: A PyTorch Platform for Distributed RL}},
  author={Heinrich K\"{u}ttler and Nantas Nardelli and Thibaut Lavril and Marco Selvatici and Viswanath Sivakumar and Tim Rockt\"{a}schel and Edward Grefenstette},
  year={2019},
  journal={arXiv preprint arXiv:1910.03552},
  url={https://github.com/facebookresearch/torchbeast},
}
```

PopArt
```
@inproceedings{hessel2019,
  title={Multi-task deep reinforcement learning with popart},
  author={Hessel, Matteo and Soyer, Hubert and Espeholt, Lasse and Czarnecki, Wojciech and Schmitt, Simon and van Hasselt, Hado},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3796--3803},
  year={2019}
}
```

Saliency
```
@article{greydanus2017visualizing,
  title={Visualizing and Understanding Atari Agents},
  author={Greydanus, Sam and Koul, Anurag and Dodge, Jonathan and Fern, Alan},
  journal={arXiv preprint arXiv:1711.00138},
  year={2017},
  url={https://github.com/greydanus/visualize_atari},
}
```
