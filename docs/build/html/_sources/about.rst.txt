About
=====

SuperSonic is a library to allow compiler developers to integrate
reinforcement learning (RL) into compilers easily, regardless of their RL expertise. SuperSonic supports
customizable RL architecture compositions to target a wide range of optimization
tasks. A key feature of SuperSonic is the use of deep RL and multi-task learning
techniques to develop a meta-optimizer to automatically find and tune the right RL
architecture from training benchmarks.

.. contents:: Overview:
    :local:

Motivation
-----------

Efforts have been made to provide out-of-the box RL algorithms
and high-level APIs for action definitions, models for program state
representation, etc. While these recent works have lowered the barrier
for integrating RL techniques into compilers, compiler engineers still
face a major hurdle.

As the right combination of RL exploration algorithms and their state, reward and
transition functions and parameters highly depend on the optimization task,
developers must carefully choose the RL architecture by finding the right
RL component composition and their parameters from a large pool of candidate
RL algorithms, machine-learning models and functions. This process currently
requires testing and manually analyzing a large combination of RL components.
Experience in the field of neural architecture search shows that doing this
by hand is an expensive and non-trivial process.

Our Vision
-----------

We present SuperSonic and aim to lower the barrier to automate the RL
architecture search and parameter tuning process by building a playground
that allows anyone to integrate RL into compilers, without knowing RL knowledge.
We make the following contributions:

1, Present a generic framework to automatically choose and tune a suitable RL architecture for code optimization tasks.

2, Demonstrate how deep RL can be used as a meta-optimizer to support the integration of RL into performance tuners.

3, Provide a large study validating the effectiveness of RL in four code optimization tasks.
