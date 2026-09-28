Platform
=======================

Class architecture
~~~~~~~~~~~~~~~~~~~~~~~~~
The following code is typical lines in main file:

.. code-block:: python

    runner = SequentialRunner(
        settings=setting_dict_or_json,
        prng=random.Random(seed),
        logger=MarketStepPrintLogger(),
    )
    runner.main()


As this code representing, the simulation is fully controlled by a runner, in this case, :class:`SequentialRunner` .
:class:`Runner` usually accepts setting dictionary, a pseudo random number generator, and a logger.

The following figure shows the class architecture.

.. image:: ../static/class-architecture.png
   :scale: 100%

Main usually has a logger and generating runner.
As a runner, Parallel Runner or Sequential Runner is available.
Runner will generate simulator and some other components such as markets, agents, and events.
Then, runner control all the components and simulator for it rule.
On the other hand, simulator provide API interface for runners to define the procedure controlling simulation.
Moreover, simulator and its components generate logs and push it to logger.

The key concept of this platform is that runner is replaceable to realize more complex simulator controlling.

Parallel runners
~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`pams.runners.MultiThreadAgentParallelRunner` and :class:`pams.runners.MultiProcessAgentParallelRunner` are experimental runners
that parallelize the order submissions of normal (non high-frequency) agents in each step.
They are drop-in replacements of :class:`pams.runners.SequentialRunner`, and the simulation results are the same as
:class:`pams.runners.SequentialRunner` with the same settings and the same seed.
The number of workers can be set by ``simulation.numParallel`` in the config, but the number of agents that can be
parallelized in each step is limited by ``maxNormalOrders`` of the session.

.. code-block:: python

    runner = MultiThreadAgentParallelRunner(
        settings=setting_dict_or_json,
        prng=random.Random(seed),
        logger=MarketStepPrintLogger(),
    )
    runner.main()

Because of the GIL of python, :class:`pams.runners.MultiThreadAgentParallelRunner` is beneficial only when
the order submission of agents waits for I/O (e.g., external models or services).
:class:`pams.runners.MultiProcessAgentParallelRunner` copies agents and markets to worker processes for each order submission,
so it is much slower and the user-defined classes must be picklable and importable from the worker processes.




