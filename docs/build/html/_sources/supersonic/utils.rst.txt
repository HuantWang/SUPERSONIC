SuperSonic.utils
=====================

SuperSonic uses
A :class:`TaskEngine<SuperSonic.utils.engine.tasks_engine.TaskEngine>` to call run() to deployment specific policy to
 objective task.
A :class:`Halide<SuperSonic.utils.engine.tasks_engine.Halide>`,
to start agent and environment. To apply a tuned RL,
SuperSonic creates a session to apply a standard RL loop to optimize the input program by using the chosen RL
exploration algorithms to select an action for a given state.
Using :class:`stoke_rl<SuperSonic.utils.engine.environments.stoke_rl>`,
to register a RL environment.

.. contents:: SuperSonic Tasks Policy Search:
  :local:

.. currentmodule:: SuperSonic.utils.engine.tasks_engine

TaskEngine
-------
.. autoclass:: TaskEngine
  :members:

Optimizing Image Pipelines
-------
.. autoclass:: Halide
  :members:


Neural Network Code Generation Reduction
-------
.. autoclass:: Tvm
  :members:

Code Size
-------
.. autoclass:: CSR
  :members:

Superoptimization
-------
.. autoclass:: Stoke
  :members:

Optimizing Image Pipelines RL Environments
-------
.. currentmodule:: SuperSonic.utils.environments.Halide_env
.. autoclass:: halide_rl
  :members:

Neural Network Code Generation Reduction RL Environments
-------
.. currentmodule:: SuperSonic.utils.environments.AutoTvm_env
.. autoclass:: autotvm_rl
  :members:

Code Size Reduction RL Environments
-------
.. currentmodule:: SuperSonic.utils.environments.CSR_env
.. autoclass:: csr_rl
  :members:

Superoptimization RL Environments
-------
.. currentmodule:: SuperSonic.utils.environments.Stoke_env
.. autoclass:: stoke_rl
  :members: