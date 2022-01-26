Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about Supersonic. Have a
question not answered here? File an issue on the `GitHub issue tracker
<https://github.com/HuantWang/SUPERSONIC/issues>`_.

.. contents:: Questions:
    :local:

What can I do with this?
------------------------

This projects lets you integrate RL into compilers easily, regardless of
their RL expertise. Currently, it includes four code optimization problems.
To use SuperSonic, the compiler developer provides the action list according
to the problem being tackled and a measurement interface to report metrics
like code size or speedup. SuperSonic then automatically assembles an RL
architecture for the targeting optimization from an extensible set of
built-in RL components.

I found a bug. How do I report it?
----------------------------------

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/HuantWang/SUPERSONIC/issues>`_.  See
:doc:`contributing` for more details.


Do I have to use reinforcement learning?
----------------------------------------

No. We think that the the gym and ray provide useful abstraction for sequential
decision making. You may use any technique you wish to explore the optimization
space.


When does a compiler enviornment consider an episode “done”?
------------------------------------------------------------

The compiler itself doesn't have a signal for termination. Actions are like
rewrite rules, it is up to the user to decide when no more improvement can be
achieved from further rewrites.
The only exception is if the compiler crashes, or the code ends up in an
unexpected state - we have to abort. This happens.


How do I run this on my own program?
------------------------------------




I want to add a new program representation / reward signal. How do I do that?
-----------------------------------------------------------------------------




Should I always try different MDP strategies?
--------------------------------------

