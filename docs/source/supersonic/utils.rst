SuperSonic.utils
=====================

SuperSonic uses
A :class:`TaskEngine<SuperSonic.utils.engine.tasks_engine.TaskEngine>` to call run() to deployment specific policy to
 objective task.
A :class:`Stoke<SuperSonic.utils.engine.tasks_engine.Stoke>` to start agent and environment. To apply a tuned RL,
SuperSonic creates a session to apply a standard RL loop to optimize the input program by using the chosen RL
exploration algorithms to select an action for a given state.
A :class:`stoke_rl<SuperSonic.utils.engine.environments.stoke_rl>` to register a RL environment.

.. contents:: SuperSonic Tasks Policy Search:
  :local:

.. currentmodule:: SuperSonic.utils.engine.tasks_engine

TaskEngine
-------
.. autoclass:: TaskEngine
  :members:

Stoke
-------
.. autoclass:: Stoke
  :members:

stoke_environments
-------
.. currentmodule:: SuperSonic.utils.environments.stoke_env
.. autoclass:: stoke_rl
  :members: